import os
import re
import glob
import copy
import pickle
import argparse

import torch
import torch.nn.functional as F
import pandas as pd

import blackcatt.wm_config as C
from blackcatt.task import load_data


# ============================================================
# 1. Build model
# ============================================================
def build_model():
    """
    You may need to edit this part if automatic model construction fails.

    The saved parameter keys look like a ResNet-style model:
      conv1.0.weight
      layer1.0.left.0.weight
      ...
    """

    # Try project-specific common constructors.
    candidates = []

    try:
        import blackcatt.task as task
        for name in [
            "Net",
            "ResNet18",
            "ResNet183x3",
            "ResNet18_3x3",
            "get_model",
            "load_model",
            "create_model",
            "init_model",
        ]:
            if hasattr(task, name):
                candidates.append(("blackcatt.task", name, getattr(task, name)))
    except Exception:
        pass

    try:
        import blackcatt.models as models
        for name in [
            "Net",
            "ResNet18",
            "ResNet183x3",
            "ResNet18_3x3",
            "get_model",
            "create_model",
        ]:
            if hasattr(models, name):
                candidates.append(("blackcatt.models", name, getattr(models, name)))
    except Exception:
        pass

    errors = []

    for module_name, name, obj in candidates:
        try:
            if name in ["get_model", "load_model", "create_model", "init_model"]:
                # Try several common signatures.
                for args in [
                    (),
                    (C.dataset,),
                    (C.dataset, 10),
                    (10,),
                ]:
                    try:
                        model = obj(*args)
                        print(f"[build_model] success: {module_name}.{name}{args}")
                        return model
                    except Exception as e:
                        errors.append(f"{module_name}.{name}{args}: {repr(e)}")
            else:
                for args in [
                    (),
                    (10,),
                ]:
                    try:
                        model = obj(*args)
                        print(f"[build_model] success: {module_name}.{name}{args}")
                        return model
                    except Exception as e:
                        errors.append(f"{module_name}.{name}{args}: {repr(e)}")
        except Exception as e:
            errors.append(f"{module_name}.{name}: {repr(e)}")

    print("\n[build_model failed]")
    print("Tried candidates:")
    for e in errors[:50]:
        print(" ", e)

    raise RuntimeError(
        "Cannot build model automatically. "
        "Open blackcatt/task.py or eval_response_degree.py and replace build_model()."
    )


# ============================================================
# 2. Load / average saved client parameters
# ============================================================
def load_status_params(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if not isinstance(obj, dict):
        raise RuntimeError(f"{path} is not a dict")

    if "parameters" not in obj:
        raise RuntimeError(f"{path} does not contain key 'parameters'. Keys={obj.keys()}")

    params = obj["parameters"]

    out = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().clone()
        else:
            out[k] = torch.as_tensor(v).detach().cpu().clone()

    return out


def find_status_files(model_dir, round_id):
    pattern = os.path.join(model_dir, f"{round_id}_client_status_*.pkl")
    files = sorted(glob.glob(pattern))

    def get_cid(path):
        m = re.search(rf"{round_id}_client_status_(\d+)\.pkl$", path)
        if m is None:
            return 10**9
        return int(m.group(1))

    files = sorted(files, key=get_cid)

    if not files:
        raise FileNotFoundError(f"No files found with pattern: {pattern}")

    return files


def average_params(param_list):
    keys = list(param_list[0].keys())
    avg = {}

    for k in keys:
        vals = [p[k] for p in param_list]

        if torch.is_floating_point(vals[0]):
            stacked = torch.stack([v.float() for v in vals], dim=0)
            avg[k] = stacked.mean(dim=0).to(dtype=vals[0].dtype)
        else:
            # For integer buffers such as num_batches_tracked.
            avg[k] = vals[0].clone()

    return avg


def load_params_into_model(model, params, device):
    """
    Load saved dict into model.
    """
    model_sd = model.state_dict()

    clean = {}
    for k, v in params.items():
        nk = k
        for prefix in ["module.", "model.", "net."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]

        if nk in model_sd:
            clean[nk] = v.to(dtype=model_sd[nk].dtype)
        else:
            clean[nk] = v

    missing, unexpected = model.load_state_dict(clean, strict=False)

    print("[load_state_dict]")
    print("  missing keys:", missing[:10], "..." if len(missing) > 10 else "")
    print("  unexpected keys:", unexpected[:10], "..." if len(unexpected) > 10 else "")

    model.to(device)
    model.eval()
    return model


# ============================================================
# 3. Evaluation
# ============================================================
def get_xy(batch, device):
    # Your current batch keys are img / label.
    x = batch["img"].to(device)
    y = batch["label"].to(device)
    return x, y


def get_logits(out):
    # Some models return logits directly; some return tuple/list.
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


@torch.no_grad()
def eval_clean(model, loader, device, num_classes=10, max_batches=None):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_n = 0

    class_loss_sum = torch.zeros(num_classes, dtype=torch.double)
    class_correct = torch.zeros(num_classes, dtype=torch.long)
    class_count = torch.zeros(num_classes, dtype=torch.long)

    seen_batches = 0

    for batch in loader:
        x, y = get_xy(batch, device)

        logits = get_logits(model(x))
        losses = F.cross_entropy(logits, y, reduction="none")
        pred = logits.argmax(dim=1)

        bs = y.numel()
        total_loss += float(losses.sum().detach().cpu())
        total_correct += int((pred == y).sum().detach().cpu())
        total_n += bs

        yc = y.detach().cpu()
        lossc = losses.detach().cpu()
        predc = pred.detach().cpu()

        for c in range(num_classes):
            mask = yc == c
            if mask.any():
                class_loss_sum[c] += lossc[mask].double().sum()
                class_correct[c] += int((predc[mask] == yc[mask]).sum())
                class_count[c] += int(mask.sum())

        seen_batches += 1
        if max_batches is not None and seen_batches >= max_batches:
            break

    overall_loss = total_loss / max(total_n, 1)
    overall_acc = total_correct / max(total_n, 1)

    class_loss = class_loss_sum / class_count.clamp_min(1).double()
    class_acc = class_correct.double() / class_count.clamp_min(1).double()

    return {
        "loss": overall_loss,
        "acc": overall_acc,
        "class_loss": class_loss,
        "class_acc": class_acc,
        "class_count": class_count,
        "n": total_n,
    }


def client_class_distribution(cid, num_classes=10):
    trainloader, _ = load_data(cid, C.n_users)
    count = torch.zeros(num_classes, dtype=torch.long)

    for batch in trainloader:
        y = batch["label"].detach().cpu()
        for c in range(num_classes):
            count[c] += int((y == c).sum())

    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--round", type=int, default=500)
    parser.add_argument("--out", type=str, default="saved_client_clean_functional_residual.csv")
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--skip-client-dist", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[Config]")
    print("dataset =", C.dataset)
    print("partition_mode =", C.partition_mode)
    print("num_classes_per_client =", getattr(C, "num_classes_per_client", None))
    print("n_users =", C.n_users)
    print("device =", device)

    files = find_status_files(args.model_dir, args.round)

    print("\n[Found files]")
    for f in files:
        print(" ", f)

    client_ids = []
    param_list = []

    for f in files:
        m = re.search(rf"{args.round}_client_status_(\d+)\.pkl$", f)
        cid = int(m.group(1))
        client_ids.append(cid)
        param_list.append(load_status_params(f))

    print("\nLoaded clients:", client_ids)

    pseudo_global_params = average_params(param_list)

    # Shared/global testloader.
    _, testloader = load_data(0, C.n_users)

    # Evaluate pseudo-global baseline.
    print("\n===== Evaluate pseudo_global =====")
    base_model = build_model()
    base_model = load_params_into_model(base_model, pseudo_global_params, device)
    base_metrics = eval_clean(
        base_model,
        testloader,
        device,
        num_classes=10,
        max_batches=args.max_eval_batches,
    )

    print(f"pseudo_global clean loss = {base_metrics['loss']:.6f}")
    print(f"pseudo_global clean acc  = {base_metrics['acc']:.6f}")

    rows = []

    # Optional: local class distribution.
    local_dists = {}
    if not args.skip_client_dist:
        print("\n[Compute local train class distributions]")
        for cid in client_ids:
            local_dists[cid] = client_class_distribution(cid, num_classes=10)
            active = [c for c in range(10) if int(local_dists[cid][c]) > 0]
            print(f"client {cid} active classes:", active)

    for cid, params in zip(client_ids, param_list):
        print(f"\n===== Evaluate client {cid} =====")

        model = build_model()
        model = load_params_into_model(model, params, device)

        m = eval_clean(
            model,
            testloader,
            device,
            num_classes=10,
            max_batches=args.max_eval_batches,
        )

        row = {
            "client": cid,
            "base_clean_loss": base_metrics["loss"],
            "client_clean_loss": m["loss"],
            "delta_clean_loss": m["loss"] - base_metrics["loss"],
            "base_clean_acc": base_metrics["acc"],
            "client_clean_acc": m["acc"],
            "delta_clean_acc": m["acc"] - base_metrics["acc"],
            "eval_n": m["n"],
        }

        print(f"client clean loss = {m['loss']:.6f}  delta = {row['delta_clean_loss']:+.6f}")
        print(f"client clean acc  = {m['acc']:.6f}  delta = {row['delta_clean_acc']:+.6f}")

        for c in range(10):
            row[f"base_loss_c{c}"] = float(base_metrics["class_loss"][c])
            row[f"client_loss_c{c}"] = float(m["class_loss"][c])
            row[f"delta_loss_c{c}"] = float(m["class_loss"][c] - base_metrics["class_loss"][c])

            row[f"base_acc_c{c}"] = float(base_metrics["class_acc"][c])
            row[f"client_acc_c{c}"] = float(m["class_acc"][c])
            row[f"delta_acc_c{c}"] = float(m["class_acc"][c] - base_metrics["class_acc"][c])

            if not args.skip_client_dist:
                row[f"local_train_count_c{c}"] = int(local_dists[cid][c])

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print(f"\nSaved: {args.out}")

    print("\n[Compact summary]")
    cols = [
        "client",
        "delta_clean_loss",
        "delta_clean_acc",
    ]
    print(df[cols].to_string(index=False))

    print("\n[Per-class loss residual columns]")
    print("delta_loss_c0 ... delta_loss_c9")
    print("Negative means client model is better than pseudo_global on that class.")


if __name__ == "__main__":
    main()