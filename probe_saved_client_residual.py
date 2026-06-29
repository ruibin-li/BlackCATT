import os
import re
import glob
import pickle
import argparse

import torch
import pandas as pd


def load_status_params(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if not isinstance(obj, dict):
        raise RuntimeError(f"{path} is not a dict.")

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
            # For integer buffers, e.g. num_batches_tracked, just keep client 0's value.
            avg[k] = vals[0].clone()

    return avg


def flatten_delta(params, baseline):
    vecs = []

    for k in baseline.keys():
        if k not in params:
            continue

        a = params[k]
        b = baseline[k]

        if not torch.is_floating_point(a):
            continue

        # use double precision for reliable cosine / norm
        vecs.append((a.double() - b.double()).flatten())

    if not vecs:
        raise RuntimeError("No floating tensors found.")

    return torch.cat(vecs)


def flatten_params(params):
    vecs = []
    for k, v in params.items():
        if torch.is_floating_point(v):
            vecs.append(v.double().flatten())
    return torch.cat(vecs)


def cosine(a, b, eps=1e-12):
    a = a.double()
    b = b.double()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + eps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--round", type=int, default=500)
    parser.add_argument("--out", type=str, default="saved_client_residual_probe.csv")
    parser.add_argument("--cos-out", type=str, default="saved_client_delta_cosine.csv")
    args = parser.parse_args()

    files = find_status_files(args.model_dir, args.round)

    print("[Found client status files]")
    for f in files:
        print(" ", f)

    param_list = []
    client_ids = []

    for f in files:
        m = re.search(rf"{args.round}_client_status_(\d+)\.pkl$", f)
        cid = int(m.group(1))
        client_ids.append(cid)
        param_list.append(load_status_params(f))

    print("\nLoaded clients:", client_ids)

    baseline = average_params(param_list)
    baseline_vec = flatten_params(baseline)
    baseline_norm = float(baseline_vec.norm())

    rows = []
    deltas = {}

    for cid, params in zip(client_ids, param_list):
        delta = flatten_delta(params, baseline)
        deltas[cid] = delta

        param_vec = flatten_params(params)

        row = {
            "client": cid,
            "param_norm": float(param_vec.norm()),
            "pseudo_global_norm": baseline_norm,
            "delta_norm_vs_pseudo_global": float(delta.norm()),
            "relative_delta_norm": float(delta.norm() / max(baseline_norm, 1e-12)),
            "cos_param_with_pseudo_global": cosine(param_vec, baseline_vec),
        }

        rows.append(row)

        print(f"\nclient {cid}")
        print(f"  param norm                 = {row['param_norm']:.6e}")
        print(f"  delta norm vs pseudo global = {row['delta_norm_vs_pseudo_global']:.6e}")
        print(f"  relative delta norm         = {row['relative_delta_norm']:.6e}")
        print(f"  cos(param, pseudo global)   = {row['cos_param_with_pseudo_global']:.9f}")

    # Pairwise delta cosine matrix.
    cos_rows = []
    print("\n[Pairwise residual cosine]")
    for i in client_ids:
        line = []
        for j in client_ids:
            sim = cosine(deltas[i], deltas[j])
            line.append(sim)
            cos_rows.append({
                "client_i": i,
                "client_j": j,
                "delta_cosine": sim,
            })
        print("client", i, ":", " ".join(f"{x:+.4f}" for x in line))

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    cdf = pd.DataFrame(cos_rows)
    cdf.to_csv(args.cos_out, index=False)

    print(f"\nSaved summary: {args.out}")
    print(f"Saved cosine matrix long table: {args.cos_out}")

    # Also print nearest neighbor by residual direction.
    print("\n[Nearest client by residual cosine]")
    for i in client_ids:
        best_j = None
        best_sim = -999
        for j in client_ids:
            if i == j:
                continue
            sim = cosine(deltas[i], deltas[j])
            if sim > best_sim:
                best_sim = sim
                best_j = j
        print(f"client {i} nearest residual client = {best_j}, cosine = {best_sim:+.6f}")


if __name__ == "__main__":
    main()