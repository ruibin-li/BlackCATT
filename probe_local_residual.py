import os
import re
import glob
import copy
import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
import pandas as pd

import blackcatt.wm_config as C
from blackcatt.task import load_data


# =========================
# 1. Build model
# =========================
def build_model():
    """
    Try to construct the model used by BlackCATT.
    If your repo uses a different model constructor, modify this function.
    """
    try:
        from blackcatt.task import Net
        return Net()
    except Exception as e1:
        pass

    try:
        from blackcatt.models import Net
        return Net()
    except Exception as e2:
        pass

    raise RuntimeError(
        "Cannot find model class automatically. "
        "Please edit build_model() and return your model."
    )


# =========================
# 2. Load checkpoint
# =========================
def find_checkpoint(round_id):
    patterns = [
        f"**/*round*{round_id}*.pt",
        f"**/*round*{round_id}*.pth",
        f"**/*snapshot*{round_id}*.pt",
        f"**/*snapshot*{round_id}*.pth",
        f"**/*{round_id}*.pt",
        f"**/*{round_id}*.pth",
    ]

    candidates = []
    for p in patterns:
        candidates.extend(glob.glob(p, recursive=True))

    candidates = sorted(set(candidates))
    if not candidates:
        raise FileNotFoundError(
            f"Cannot auto-find checkpoint for round {round_id}. "
            f"Please pass --ckpt /path/to/checkpoint.pt"
        )

    print("[Auto checkpoint candidates]")
    for i, path in enumerate(candidates[:20]):
        print(f"  [{i}] {path}")

    print(f"\nUsing: {candidates[0]}")
    return candidates[0]


def extract_state_dict(obj, model):
    """
    Support several common checkpoint formats.
    """
    if isinstance(obj, dict):
        for key in [
            "model_state_dict",
            "state_dict",
            "model",
            "net",
            "net_state_dict",
            "server_model",
            "global_model",
        ]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]

        # Sometimes the checkpoint itself is already a state_dict.
        model_keys = set(model.state_dict().keys())
        obj_keys = set(obj.keys())
        overlap = len(model_keys.intersection(obj_keys))
        if overlap > 0:
            return obj

    raise RuntimeError(
        "Unknown checkpoint format. "
        "Print torch.load(ckpt).keys() and adjust extract_state_dict()."
    )


def clean_state_dict_keys(sd):
    """
    Remove common prefixes such as module. or model.
    """
    new_sd = {}
    for k, v in sd.items():
        nk = k
        for prefix in ["module.", "model.", "net."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        new_sd[nk] = v
    return new_sd


def load_global_model(ckpt_path, device):
    model = build_model().to(device)
    obj = torch.load(ckpt_path, map_location=device)
    sd = extract_state_dict(obj, model)
    sd = clean_state_dict_keys(sd)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("\n[Checkpoint load]")
    print("missing keys:", missing[:10], "..." if len(missing) > 10 else "")
    print("unexpected keys:", unexpected[:10], "..." if len(unexpected) > 10 else "")

    model.eval()
    return model


# =========================
# 3. Basic train/eval helpers
# =========================
def get_xy(batch, device):
    """
    Your current batch keys are:
      img, label
    """
    x = batch["img"].to(device)
    y = batch["label"].to(device)
    return x, y


def local_train_one_client(model, trainloader, device, epochs=1, lr=0.01, wd=0.0, max_batches=None):
    model.train()
    opt = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=wd,
    )

    total_loss = 0.0
    total_n = 0
    seen_batches = 0

    for _ in range(epochs):
        for batch in trainloader:
            x, y = get_xy(batch, device)

            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()

            bs = y.numel()
            total_loss += loss.item() * bs
            total_n += bs
            seen_batches += 1

            if max_batches is not None and seen_batches >= max_batches:
                return total_loss / max(total_n, 1)

    return total_loss / max(total_n, 1)


@torch.no_grad()
def per_class_loss(model, loader, device, num_classes=10, max_batches=None):
    model.eval()

    loss_sum = torch.zeros(num_classes, device=device)
    count = torch.zeros(num_classes, device=device)

    seen_batches = 0

    for batch in loader:
        x, y = get_xy(batch, device)
        logits = model(x)

        losses = F.cross_entropy(logits, y, reduction="none")

        for c in range(num_classes):
            mask = y == c
            if mask.any():
                loss_sum[c] += losses[mask].sum()
                count[c] += mask.sum()

        seen_batches += 1
        if max_batches is not None and seen_batches >= max_batches:
            break

    avg = loss_sum / count.clamp_min(1)
    return avg.detach().cpu(), count.detach().cpu()


def flatten_delta(local_model, global_model):
    vecs = []

    gs = global_model.state_dict()
    ls = local_model.state_dict()

    for k in gs.keys():
        if k not in ls:
            continue

        g = gs[k]
        l = ls[k]

        if not torch.is_floating_point(g):
            continue

        vecs.append((l.detach().float().cpu() - g.detach().float().cpu()).flatten())

    if not vecs:
        raise RuntimeError("No floating tensors found for delta.")

    return torch.cat(vecs)


def cosine(a, b, eps=1e-12):
    return float(torch.dot(a, b) / (a.norm() * b.norm() + eps))


# =========================
# 4. Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--snapshot-round", type=int, default=500)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)

    parser.add_argument("--out", type=str, default="local_residual_probe.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[Config]")
    print("dataset =", C.dataset)
    print("partition_mode =", C.partition_mode)
    print("num_classes_per_client =", getattr(C, "num_classes_per_client", None))
    print("n_users =", C.n_users)
    print("device =", device)

    ckpt = args.ckpt if args.ckpt is not None else find_checkpoint(args.snapshot_round)

    global_model = load_global_model(ckpt, device)

    # Use client 0 only to get testloader.
    # In many Flower examples, testloader is the shared/global test split.
    _, testloader = load_data(0, C.n_users)

    global_class_loss, global_class_count = per_class_loss(
        global_model,
        testloader,
        device,
        num_classes=10,
        max_batches=args.max_eval_batches,
    )

    print("\n[Global per-class loss]")
    for c in range(10):
        print(f"class {c}: loss={global_class_loss[c].item():.6f}, count={int(global_class_count[c].item())}")

    rows = []
    deltas = {}

    for cid in range(C.n_users):
        print(f"\n===== Client {cid} =====")

        trainloader, _ = load_data(cid, C.n_users)

        local_model = copy.deepcopy(global_model).to(device)

        train_loss = local_train_one_client(
            local_model,
            trainloader,
            device,
            epochs=args.epochs,
            lr=args.lr,
            wd=args.wd,
            max_batches=args.max_train_batches,
        )

        delta = flatten_delta(local_model, global_model)
        deltas[cid] = delta

        delta_norm = float(delta.norm())
        global_norm = float(
            torch.cat([
                p.detach().float().cpu().flatten()
                for p in global_model.state_dict().values()
                if torch.is_floating_point(p)
            ]).norm()
        )

        local_class_loss, local_class_count = per_class_loss(
            local_model,
            testloader,
            device,
            num_classes=10,
            max_batches=args.max_eval_batches,
        )

        residual = local_class_loss - global_class_loss

        print(f"local train loss = {train_loss:.6f}")
        print(f"delta norm       = {delta_norm:.6e}")
        print(f"relative norm    = {delta_norm / max(global_norm, 1e-12):.6e}")

        print("class residual loss = local_loss - global_loss")
        for c in range(10):
            print(f"  class {c}: {residual[c].item():+.6f}")

        row = {
            "client": cid,
            "local_train_loss": train_loss,
            "delta_norm": delta_norm,
            "global_norm": global_norm,
            "relative_delta_norm": delta_norm / max(global_norm, 1e-12),
        }

        for c in range(10):
            row[f"global_loss_c{c}"] = float(global_class_loss[c])
            row[f"local_loss_c{c}"] = float(local_class_loss[c])
            row[f"residual_loss_c{c}"] = float(residual[c])

        rows.append(row)

    # Pairwise cosine similarity between local updates.
    print("\n[Pairwise delta cosine similarity]")
    for i in range(C.n_users):
        for j in range(i + 1, C.n_users):
            sim = cosine(deltas[i], deltas[j])
            print(f"cos(delta_{i}, delta_{j}) = {sim:+.6f}")

    # Save summary.
    df = pd.DataFrame(rows)

    # Add compact nearest-neighbor info by delta cosine.
    nn_client = []
    nn_cos = []
    for i in range(C.n_users):
        best_j = None
        best_sim = -999.0
        for j in range(C.n_users):
            if i == j:
                continue
            sim = cosine(deltas[i], deltas[j])
            if sim > best_sim:
                best_sim = sim
                best_j = j
        nn_client.append(best_j)
        nn_cos.append(best_sim)

    df["nearest_delta_client"] = nn_client
    df["nearest_delta_cosine"] = nn_cos

    df.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()