"""
Evaluate class-wise response degree for BlackCATT collusion checkpoints.

This is designed as a drop-in analysis script next to eval_auc_short.py.
It does NOT change training. It reuses existing triggers/checkpoints and exports
per-user and per-class response metrics for IID and PATH5 runs.

Usage examples:
  IID_DIR=/path/to/iid/output P5_DIR=/path/to/path5/output python eval_response_degree.py

Optional:
  python eval_response_degree.py --runs IID=/path/to/iid PATH5=/path/to/path5 --snapshot-round 500
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import blackcatt.wm_config as C
from blackcatt.models import ResNet, ResidualBlock
from blackcatt.wm_task import load_collusion, _normalize_triggers


def arr(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def align_code_and_p() -> Tuple[np.ndarray, np.ndarray]:
    """Return V with shape [n_users, m] and P with shape [m, n_classes]."""
    V = arr(C.clients_tardos_q).astype(int)
    P = arr(C.p_secret).astype(float)

    if V.shape[0] == C.n_users:
        pass
    elif V.shape[1] == C.n_users:
        V = V.T
    elif V.shape[0] > C.n_users:
        print(f"[info] code has {V.shape[0]} users; using first C.n_users={C.n_users}")
        V = V[: C.n_users, :]
    else:
        raise ValueError(f"Cannot align clients_tardos_q shape {V.shape} with C.n_users={C.n_users}")

    m = V.shape[1]
    if P.shape[0] == m:
        pass
    elif P.shape[1] == m:
        P = P.T
    elif P.shape[0] > m:
        print(f"[info] p_secret has {P.shape[0]} rows; using first m={m}, consistent with wm_config.py")
        P = P[:m, :]
    else:
        raise ValueError(f"Cannot align p_secret shape {P.shape} with code length m={m}")

    return V, P


def score_vec(y: np.ndarray, V: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Final and prefix-max Tardos scores for all users."""
    m = V.shape[1]
    p = P[np.arange(m), y]
    p = np.clip(p, 1e-12, 1 - 1e-12)
    plus = np.sqrt((1 - p) / p)
    minus = -np.sqrt(p / (1 - p))
    S = np.where(V == y[None, :], plus[None, :], minus[None, :])
    return S.sum(axis=1), S.cumsum(axis=1).max(axis=1)


def build_model(device: str):
    # Keep this identical to eval_auc_short.py. If you later use another model,
    # change it here in the same way as the old analysis script.
    net = ResNet(ResidualBlock).to(device).eval()
    return net


def predict_on_triggers(model, triggers: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        logits = model(triggers)
        probs = torch.softmax(logits, dim=1)
        y = logits.argmax(dim=1)
    return y.detach().cpu().numpy(), probs.detach().cpu().numpy()


def response_profiles(
    y: np.ndarray,
    probs: np.ndarray,
    V: np.ndarray,
    n_classes: int,
    min_count: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each candidate user u and class c, measure whether the model responds as c
    on trigger positions where user u's Tardos symbol is c.

    hard_raw[u,c]      = P(model output = c | V[u,t] = c)
    hard_center[u,c]   = hard_raw[u,c] - P(model output = c | V[u,t] != c)
    hard_z[u,c]        = two-sample z-style normalization of hard_center
    soft_raw[u,c]      = E[softmax_c | V[u,t] = c]
    soft_center[u,c]   = soft_raw[u,c] - E[softmax_c | V[u,t] != c]

    The centered version is the main one: it corrects for the model simply liking
    a class globally.
    """
    n_users, m = V.shape
    eps = 1e-12

    class_rows: List[Dict] = []
    user_rows: List[Dict] = []

    for u in range(n_users):
        hard_raws, hard_centers, hard_zs = [], [], []
        soft_raws, soft_centers = [], []
        weights = []

        for c in range(n_classes):
            pos = V[u] == c
            neg = ~pos
            n_pos = int(pos.sum())
            n_neg = int(neg.sum())

            if n_pos < min_count or n_neg == 0:
                hard_raw = hard_neg = hard_center = hard_z = np.nan
                soft_raw = soft_neg = soft_center = np.nan
            else:
                hard_raw = float(np.mean(y[pos] == c))
                hard_neg = float(np.mean(y[neg] == c))
                hard_center = hard_raw - hard_neg

                # Approximate standard error for difference of proportions.
                var = hard_raw * (1 - hard_raw) / max(n_pos, 1) + hard_neg * (1 - hard_neg) / max(n_neg, 1)
                hard_z = hard_center / np.sqrt(var + eps)

                soft_raw = float(np.mean(probs[pos, c]))
                soft_neg = float(np.mean(probs[neg, c]))
                soft_center = soft_raw - soft_neg

                hard_raws.append(hard_raw)
                hard_centers.append(hard_center)
                hard_zs.append(hard_z)
                soft_raws.append(soft_raw)
                soft_centers.append(soft_center)
                weights.append(n_pos)

            class_rows.append(
                {
                    "user": u,
                    "class": c,
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "hard_raw": hard_raw,
                    "hard_neg": hard_neg,
                    "hard_center": hard_center,
                    "hard_z": hard_z,
                    "soft_raw": soft_raw,
                    "soft_neg": soft_neg,
                    "soft_center": soft_center,
                }
            )

        weights_arr = np.asarray(weights, dtype=float)
        if weights_arr.size > 0 and weights_arr.sum() > 0:
            weights_arr = weights_arr / weights_arr.sum()
            hard_raw_score = float(np.sum(weights_arr * np.asarray(hard_raws)))
            hard_center_score = float(np.sum(weights_arr * np.asarray(hard_centers)))
            hard_z_score = float(np.sum(weights_arr * np.asarray(hard_zs)))
            soft_raw_score = float(np.sum(weights_arr * np.asarray(soft_raws)))
            soft_center_score = float(np.sum(weights_arr * np.asarray(soft_centers)))
        else:
            hard_raw_score = hard_center_score = hard_z_score = np.nan
            soft_raw_score = soft_center_score = np.nan

        # This equals the overall codeword agreement rate and is useful as a sanity check.
        agreement = float(np.mean(y == V[u]))

        user_rows.append(
            {
                "user": u,
                "agreement": agreement,
                "hard_raw_score": hard_raw_score,
                "hard_center_score": hard_center_score,
                "hard_z_score": hard_z_score,
                "soft_raw_score": soft_raw_score,
                "soft_center_score": soft_center_score,
            }
        )

    return pd.DataFrame(user_rows), pd.DataFrame(class_rows)


def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() == 0:
        return np.nan
    aa, bb = a[mask], b[mask]
    denom = np.linalg.norm(aa) * np.linalg.norm(bb)
    if denom <= 1e-12:
        return np.nan
    return float(np.dot(aa, bb) / denom)


def safe_l2(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() == 0:
        return np.nan
    return float(np.linalg.norm(a[mask] - b[mask]))


def load_case_prediction(
    folder: Path,
    user_idxs: Iterable[int],
    triggers: torch.Tensor,
    device: str,
    snapshot_round: int,
) -> Tuple[np.ndarray, np.ndarray]:
    net = build_model(device)
    net = load_collusion(
        list(user_idxs),
        net,
        folder=str(folder) + "/",
        before_cid=f"{snapshot_round}_client_status_",
        after_cid=".pkl",
        device=device,
    )
    return predict_on_triggers(net, triggers)


def build_self_references(
    run_name: str,
    folder: Path,
    triggers: torch.Tensor,
    V: np.ndarray,
    n_classes: int,
    device: str,
    snapshot_round: int,
    min_count: int,
) -> pd.DataFrame:
    """Single-client reference response profile for each user's own model."""
    rows = []
    for u in range(C.n_users):
        y, probs = load_case_prediction(folder, [u], triggers, device, snapshot_round)
        _, class_df = response_profiles(y, probs, V, n_classes, min_count=min_count)
        self_df = class_df[class_df["user"] == u].copy()
        self_df.insert(0, "run", run_name)
        self_df.rename(
            columns={
                "hard_raw": "ref_hard_raw",
                "hard_center": "ref_hard_center",
                "hard_z": "ref_hard_z",
                "soft_raw": "ref_soft_raw",
                "soft_center": "ref_soft_center",
            },
            inplace=True,
        )
        keep = [
            "run",
            "user",
            "class",
            "n_pos",
            "ref_hard_raw",
            "ref_hard_center",
            "ref_hard_z",
            "ref_soft_raw",
            "ref_soft_center",
        ]
        rows.append(self_df[keep])
    return pd.concat(rows, ignore_index=True)


def evaluate_run(
    run_name: str,
    folder: Path,
    V: np.ndarray,
    P: np.ndarray,
    device: str,
    snapshot_round: int,
    min_count: int,
    with_reference: bool,
    m_eval: int | None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    triggers = _normalize_triggers(np.load(folder / "triggers.npy"), device)

    if m_eval is not None:
        if m_eval <= 0:
            raise ValueError(f"m_eval must be positive, got {m_eval}")
        if m_eval > V.shape[1]:
            raise ValueError(f"m_eval={m_eval} exceeds code length {V.shape[1]}")
        if m_eval > triggers.shape[0]:
            raise ValueError(f"m_eval={m_eval} exceeds trigger count {triggers.shape[0]}")
        V = V[:, :m_eval]
        P = P[:m_eval, :]
        triggers = triggers[:m_eval]

    n_classes = int(P.shape[1])

    print(f"[{run_name}] V shape={V.shape}, P shape={P.shape}, triggers={tuple(triggers.shape)}, n_classes={n_classes}, m_eval={m_eval}")
    print(f"[{run_name}] folder={folder}")

    ref_df = None
    ref_center = None
    ref_soft_center = None
    if with_reference:
        print(f"[{run_name}] building single-client reference profiles...")
        ref_df = build_self_references(run_name, folder, triggers, V, n_classes, device, snapshot_round, min_count)
        ref_center = np.full((C.n_users, n_classes), np.nan, dtype=float)
        ref_soft_center = np.full((C.n_users, n_classes), np.nan, dtype=float)
        for u, c, ref_hard_center, ref_soft_center_val in ref_df[
            ["user", "class", "ref_hard_center", "ref_soft_center"]
        ].itertuples(index=False, name=None):
            ref_center[int(u), int(c)] = float(ref_hard_center)
            ref_soft_center[int(u), int(c)] = float(ref_soft_center_val)
    else:
        ref_df = pd.DataFrame()

    user_rows = []
    class_rows = []
    case_rows = []

    # Same two-colluder sweep as eval_auc_short.py: (cid, cid+1 mod n_users).
    for cid in range(C.n_users):
        col = (cid + 1) % C.n_users
        y, probs = load_case_prediction(folder, [cid, col], triggers, device, snapshot_round)

        sf, sp = score_vec(y, V, P)
        user_df, class_df = response_profiles(y, probs, V, n_classes, min_count=min_count)

        guilty = np.zeros(C.n_users, dtype=int)
        guilty[[cid, col]] = 1

        user_df.insert(0, "run", run_name)
        user_df.insert(1, "cid", cid)
        user_df.insert(2, "col", col)
        user_df["guilty"] = guilty[user_df["user"].values]
        user_df["score_final"] = sf[user_df["user"].values]
        user_df["score_prefix"] = sp[user_df["user"].values]

        if with_reference and ref_center is not None:
            cos_hard, l2_hard, cos_soft, l2_soft = [], [], [], []
            # Reconstruct collusion response matrix from class_df.
            coll_center = np.full((C.n_users, n_classes), np.nan, dtype=float)
            coll_soft_center = np.full((C.n_users, n_classes), np.nan, dtype=float)
            for u, c, hard_center, soft_center in class_df[
                ["user", "class", "hard_center", "soft_center"]
            ].itertuples(index=False, name=None):
                coll_center[int(u), int(c)] = float(hard_center)
                coll_soft_center[int(u), int(c)] = float(soft_center)
            for u in user_df["user"].values:
                cos_hard.append(safe_cosine(coll_center[int(u)], ref_center[int(u)]))
                l2_hard.append(safe_l2(coll_center[int(u)], ref_center[int(u)]))
                cos_soft.append(safe_cosine(coll_soft_center[int(u)], ref_soft_center[int(u)]))
                l2_soft.append(safe_l2(coll_soft_center[int(u)], ref_soft_center[int(u)]))
            user_df["cos_self_ref_hard_center"] = cos_hard
            user_df["l2_self_ref_hard_center"] = l2_hard
            user_df["cos_self_ref_soft_center"] = cos_soft
            user_df["l2_self_ref_soft_center"] = l2_soft

        class_df.insert(0, "run", run_name)
        class_df.insert(1, "cid", cid)
        class_df.insert(2, "col", col)
        class_df["guilty"] = guilty[class_df["user"].values]

        # Multi-av test from the old script: output not matching either colluder symbol.
        mav = float(np.mean([y[t] not in V[[cid, col], t] for t in range(V.shape[1])]))

        metric_cols = [
            "score_final",
            "score_prefix",
            "agreement",
            "hard_center_score",
            "hard_z_score",
            "soft_center_score",
        ]
        if with_reference:
            metric_cols += ["cos_self_ref_hard_center", "cos_self_ref_soft_center"]

        case = {"run": run_name, "cid": cid, "col": col, "mav": mav}
        for metric in metric_cols:
            if metric not in user_df.columns:
                continue
            # Higher is better for all listed metrics.
            top_user = int(user_df.sort_values(metric, ascending=False).iloc[0]["user"])
            top2_users = user_df.sort_values(metric, ascending=False).head(2)["user"].astype(int).tolist()
            case[f"top1_{metric}"] = top_user
            case[f"top1_hit_{metric}"] = int(top_user in [cid, col])
            case[f"top2_recall_{metric}"] = int(cid in top2_users) + int(col in top2_users)
        case_rows.append(case)

        user_rows.append(user_df)
        class_rows.append(class_df)

    return (
        pd.concat(user_rows, ignore_index=True),
        pd.concat(class_rows, ignore_index=True),
        pd.DataFrame(case_rows),
        ref_df,
    )


def parse_runs(args_runs: List[str] | None) -> Dict[str, Path]:
    if args_runs:
        out = {}
        for item in args_runs:
            if "=" not in item:
                raise ValueError(f"Run spec must look like NAME=/path/to/folder, got: {item}")
            name, path = item.split("=", 1)
            out[name] = Path(path)
        return out

    runs = {}
    if os.environ.get("IID_DIR"):
        runs["IID"] = Path(os.environ["IID_DIR"])
    if os.environ.get("P5_DIR"):
        runs["PATH5"] = Path(os.environ["P5_DIR"])
    if not runs:
        raise RuntimeError("Provide --runs NAME=/path or set IID_DIR/P5_DIR environment variables.")
    return runs


def auc_table(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    rows = []
    for run, g in df.groupby("run"):
        y_true = g["guilty"].values
        for metric in metrics:
            if metric not in g.columns:
                continue
            vals = g[metric].values.astype(float)
            mask = np.isfinite(vals)
            if mask.sum() == 0 or len(np.unique(y_true[mask])) < 2:
                auc = np.nan
            else:
                auc = roc_auc_score(y_true[mask], vals[mask])
            rows.append({"run": run, "metric": metric, "auc": auc})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", help="Run specs like IID=/path/to/iid PATH5=/path/to/path5")
    parser.add_argument("--out", default="analysis_response_degree")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--snapshot-round", type=int, default=500)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--m-eval", type=int, default=None, help="Use only the first m_eval triggers/queries at evaluation time")
    parser.add_argument("--no-reference", action="store_true", help="Skip single-client reference similarity metrics")
    args = parser.parse_args()

    V, P = align_code_and_p()
    runs = parse_runs(args.runs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_user, all_class, all_case, all_ref = [], [], [], []
    for run_name, folder in runs.items():
        user_df, class_df, case_df, ref_df = evaluate_run(
            run_name=run_name,
            folder=folder,
            V=V,
            P=P,
            device=args.device,
            snapshot_round=args.snapshot_round,
            min_count=args.min_count,
            with_reference=not args.no_reference,
            m_eval=args.m_eval,
        )
        all_user.append(user_df)
        all_class.append(class_df)
        all_case.append(case_df)
        if len(ref_df) > 0:
            all_ref.append(ref_df)

    user_all = pd.concat(all_user, ignore_index=True)
    class_all = pd.concat(all_class, ignore_index=True)
    case_all = pd.concat(all_case, ignore_index=True)
    ref_all = pd.concat(all_ref, ignore_index=True) if all_ref else pd.DataFrame()

    user_all.to_csv(out / "response_user_scores.csv", index=False)
    class_all.to_csv(out / "response_class_profile.csv", index=False)
    case_all.to_csv(out / "response_case_topk.csv", index=False)
    if len(ref_all) > 0:
        ref_all.to_csv(out / "single_client_reference_profile.csv", index=False)

    metrics = [
        "score_final",
        "score_prefix",
        "agreement",
        "hard_center_score",
        "hard_z_score",
        "soft_center_score",
        "cos_self_ref_hard_center",
        "cos_self_ref_soft_center",
    ]
    auc = auc_table(user_all, metrics)
    auc.to_csv(out / "response_auc.csv", index=False)

    # Case-level top-k summary.
    topk_rows = []
    for run, g in case_all.groupby("run"):
        row = {"run": run, "n_cases": len(g), "mav_mean": g["mav"].mean()}
        for col in g.columns:
            if col.startswith("top1_hit_"):
                row[col.replace("top1_hit_", "top1_hit_rate_")] = g[col].mean()
            if col.startswith("top2_recall_"):
                row[col.replace("top2_recall_", "top2_recall_mean_")] = g[col].mean()
        topk_rows.append(row)
    topk = pd.DataFrame(topk_rows)
    topk.to_csv(out / "response_topk_summary.csv", index=False)

    print("\nAUC summary:")
    print(auc.sort_values(["run", "auc"], ascending=[True, False]).to_string(index=False))
    print("\nTop-k summary:")
    print(topk.to_string(index=False))
    print(f"\nSaved CSV files under: {out}")


if __name__ == "__main__":
    main()
