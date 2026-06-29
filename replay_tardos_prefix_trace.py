import argparse
import os
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

import blackcatt.wm_config as C
from blackcatt.models import ResNet, ResidualBlock
from blackcatt.wm_task import load_collusion, _normalize_triggers


def arr(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def align_code_and_psecret():
    V = arr(C.clients_tardos_q).astype(int)
    P = arr(C.p_secret).astype(float)

    # Force V to [n_users, m]
    if V.shape[0] == C.n_users:
        pass
    elif V.shape[1] == C.n_users:
        V = V.T
    elif V.shape[0] > C.n_users:
        print(f"[info] code has {V.shape[0]} users; using first C.n_users={C.n_users}")
        V = V[:C.n_users, :]
    else:
        raise ValueError(f"Cannot align clients_tardos_q shape {V.shape} with C.n_users={C.n_users}")

    m = V.shape[1]

    # Force P to [m, n_classes]
    if P.shape[0] == m:
        pass
    elif P.shape[1] == m:
        P = P.T
    elif P.shape[0] > m:
        print(f"[info] p_secret has {P.shape[0]} rows; using first m={m}")
        P = P[:m, :]
    else:
        raise ValueError(f"Cannot align p_secret shape {P.shape} with code length m={m}")

    return V, P


def score_increment_matrix(y, V, P):
    """
    Return increment matrix S with shape [n_users, m].
    This matches Tardos scoring used in eval_auc_short.py.
    """
    m = V.shape[1]
    p = P[np.arange(m), y]
    p = np.clip(p, 1e-12, 1 - 1e-12)

    plus = np.sqrt((1 - p) / p)
    minus = -np.sqrt(p / (1 - p))

    S = np.where(V == y[None, :], plus[None, :], minus[None, :])
    return S


def threshold_by_prefix(m, tau, pfp):
    """
    Reproduce threshold Z in blackcatt/wm_task.py::tardos_accusation.
    """
    Zs = np.full(m, np.nan, dtype=float)

    for mi in range(m):
        a = 1 / (2 * np.log(pfp))
        b = 1 / (3 * np.sqrt(tau))
        c = mi + 1

        disc = b * b - 4 * a * c

        if disc >= 0:
            x1 = (-b + np.sqrt(disc)) / (2 * a)
            x2 = (-b - np.sqrt(disc)) / (2 * a)

            candidates = [x for x in [x1, x2] if x > 0]
            if candidates:
                Zs[mi] = min(candidates)

    return Zs


def summarize_prefix(run_name, cid, col, prefix_len, scores, Z, guilty):
    coll_scores = scores[guilty == 1]
    inn_scores = scores[guilty == 0]

    accused = int(np.argmax(scores)) if np.isfinite(Z) and scores.max() > Z else -1

    row = {
        "run": run_name,
        "cid": cid,
        "col": col,
        "prefix_len": prefix_len,
        "threshold": Z,
        "accused": accused,
        "accused_is_colluder": int(accused in [cid, col]) if accused != -1 else 0,
        "any_accused": int(accused != -1),
    }

    row.update({
        "colluder_min": float(np.min(coll_scores)),
        "colluder_median": float(np.median(coll_scores)),
        "colluder_mean": float(np.mean(coll_scores)),
        "colluder_max": float(np.max(coll_scores)),
        "colluder_min_margin": float(np.min(coll_scores) - Z),
        "colluder_median_margin": float(np.median(coll_scores) - Z),
        "colluder_mean_margin": float(np.mean(coll_scores) - Z),
        "colluder_max_margin": float(np.max(coll_scores) - Z),
        "n_colluder_above_threshold": int(np.sum(coll_scores > Z)),
        "threshold_percentile_among_colluders": float(np.mean(coll_scores <= Z)),
    })

    row.update({
        "innocent_max": float(np.max(inn_scores)),
        "innocent_mean": float(np.mean(inn_scores)),
        "innocent_max_margin": float(np.max(inn_scores) - Z),
        "n_innocent_above_threshold": int(np.sum(inn_scores > Z)),
    })

    return row


def run_one_setting(run_name, folder, before_cid, after_cid, pfp, device):
    folder = Path(folder)

    trig = _normalize_triggers(np.load(folder / "triggers.npy"), device)
    V, P = align_code_and_psecret()
    m = V.shape[1]
    Zs = threshold_by_prefix(m, tau=C.tau, pfp=pfp)

    print(f"\n== {run_name} ==")
    print("folder:", folder)
    print("V shape:", V.shape, "P shape:", P.shape)
    print("tau:", C.tau, "pfp:", pfp)
    print("checkpoint pattern:", before_cid + "<cid>" + after_cid)

    prefix_rows = []
    case_rows = []

    for cid in range(C.n_users):
        col = (cid + 1) % C.n_users

        net = ResNet(ResidualBlock).to(device).eval()
        net = load_collusion(
            [cid, col],
            net,
            folder=str(folder) + "/",
            before_cid=before_cid,
            after_cid=after_cid,
            device=device,
        )

        with torch.no_grad():
            y = net(trig).argmax(1).detach().cpu().numpy()

        S = score_increment_matrix(y, V, P)
        CUM = np.cumsum(S, axis=1)

        guilty = np.zeros(C.n_users, dtype=int)
        guilty[[cid, col]] = 1

        mav = np.mean([y[t] not in V[[cid, col], t] for t in range(m)])

        first_accused = -1
        first_prefix = np.nan
        first_accused_is_colluder = 0

        for mi in range(m):
            prefix_len = mi + 1
            scores = CUM[:, mi]
            Z = Zs[mi]

            row = summarize_prefix(run_name, cid, col, prefix_len, scores, Z, guilty)
            prefix_rows.append(row)

            if first_accused == -1 and row["any_accused"] == 1:
                first_accused = row["accused"]
                first_prefix = prefix_len
                first_accused_is_colluder = row["accused_is_colluder"]

        final_scores = CUM[:, -1]
        prefix_max_scores = CUM.max(axis=1)

        coll_final = final_scores[guilty == 1]
        inn_final = final_scores[guilty == 0]

        coll_prefix_max = prefix_max_scores[guilty == 1]
        inn_prefix_max = prefix_max_scores[guilty == 0]

        # Original-like FN/FP for this replay case:
        # FN if nobody accused, FP if first accused is not a colluder.
        fn_original_like = int(first_accused == -1)
        fp_original_like = int(first_accused != -1 and first_accused not in [cid, col])

        case_rows.append({
            "run": run_name,
            "cid": cid,
            "col": col,
            "mav": float(mav),

            "first_accused": int(first_accused),
            "first_prefix": first_prefix,
            "first_accused_is_colluder": int(first_accused_is_colluder),
            "fn_original_like": fn_original_like,
            "fp_original_like": fp_original_like,

            "final_threshold": float(Zs[-1]),
            "final_colluder_max": float(np.max(coll_final)),
            "final_colluder_median": float(np.median(coll_final)),
            "final_innocent_max": float(np.max(inn_final)),
            "final_colluder_max_margin": float(np.max(coll_final) - Zs[-1]),
            "final_innocent_max_margin": float(np.max(inn_final) - Zs[-1]),

            "prefix_max_colluder_max": float(np.max(coll_prefix_max)),
            "prefix_max_innocent_max": float(np.max(inn_prefix_max)),
            "prefix_max_gap": float(np.max(coll_prefix_max) - np.max(inn_prefix_max)),
        })

        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return prefix_rows, case_rows


def make_plots(prefix_df, out_dir):
    plot_dir = Path(out_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    avg = (
        prefix_df
        .groupby(["run", "prefix_len"])
        .agg(
            threshold=("threshold", "mean"),
            colluder_max=("colluder_max", "mean"),
            colluder_median=("colluder_median", "mean"),
            innocent_max=("innocent_max", "mean"),
            colluder_max_margin=("colluder_max_margin", "mean"),
            colluder_median_margin=("colluder_median_margin", "mean"),
            innocent_max_margin=("innocent_max_margin", "mean"),
            n_colluder_above_threshold=("n_colluder_above_threshold", "mean"),
            threshold_percentile_among_colluders=("threshold_percentile_among_colluders", "mean"),
        )
        .reset_index()
    )

    avg.to_csv(Path(out_dir) / "per_prefix_average_over_cases.csv", index=False)

    for run, sub in avg.groupby("run"):
        sub = sub.sort_values("prefix_len")

        plt.figure(figsize=(9, 5))
        plt.plot(sub["prefix_len"], sub["threshold"], label="threshold")
        plt.plot(sub["prefix_len"], sub["colluder_max"], label="max colluder score")
        plt.plot(sub["prefix_len"], sub["colluder_median"], label="median colluder score")
        plt.plot(sub["prefix_len"], sub["innocent_max"], label="max innocent score")
        plt.xlabel("prefix length")
        plt.ylabel("score")
        plt.title(f"{run}: threshold vs scores by prefix")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"{run}_threshold_vs_scores_by_prefix.png", dpi=200)
        plt.close()

        plt.figure(figsize=(9, 5))
        plt.plot(sub["prefix_len"], sub["colluder_max_margin"], label="max colluder margin")
        plt.plot(sub["prefix_len"], sub["colluder_median_margin"], label="median colluder margin")
        plt.plot(sub["prefix_len"], sub["innocent_max_margin"], label="max innocent margin")
        plt.axhline(0, linestyle="--", linewidth=1)
        plt.xlabel("prefix length")
        plt.ylabel("score - threshold")
        plt.title(f"{run}: margin to dynamic threshold")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"{run}_margins_by_prefix.png", dpi=200)
        plt.close()

        plt.figure(figsize=(9, 5))
        plt.plot(
            sub["prefix_len"],
            sub["threshold_percentile_among_colluders"],
            label="threshold percentile among colluders",
        )
        plt.xlabel("prefix length")
        plt.ylabel("fraction of colluder scores <= threshold")
        plt.title(f"{run}: threshold percentile among colluder scores")
        plt.ylim(-0.05, 1.05)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"{run}_threshold_percentile_among_colluders.png", dpi=200)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iid-dir", default=os.environ.get("IID_DIR"))
    parser.add_argument("--path5-dir", default=os.environ.get("P5_DIR"))
    parser.add_argument("--out", default="analysis_tardos_prefix_replay")

    parser.add_argument("--before-cid", default="500_client_status_")
    parser.add_argument("--after-cid", default=".pkl")
    parser.add_argument("--pfp", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    if not args.iid_dir or not args.path5_dir:
        raise ValueError("Please set IID_DIR and P5_DIR or pass --iid-dir and --path5-dir.")

    Path(args.out).mkdir(parents=True, exist_ok=True)

    all_prefix_rows = []
    all_case_rows = []

    for run_name, folder in [("IID", args.iid_dir), ("PATH5", args.path5_dir)]:
        prefix_rows, case_rows = run_one_setting(
            run_name=run_name,
            folder=folder,
            before_cid=args.before_cid,
            after_cid=args.after_cid,
            pfp=args.pfp,
            device=args.device,
        )
        all_prefix_rows.extend(prefix_rows)
        all_case_rows.extend(case_rows)

    prefix_df = pd.DataFrame(all_prefix_rows)
    case_df = pd.DataFrame(all_case_rows)

    prefix_df.to_csv(Path(args.out) / "per_case_per_prefix_trace.csv", index=False)
    case_df.to_csv(Path(args.out) / "per_case_original_like_summary.csv", index=False)

    overall = (
        case_df
        .groupby("run")
        .agg(
            n_cases=("cid", "count"),
            fn_original_like_rate=("fn_original_like", "mean"),
            fp_original_like_rate=("fp_original_like", "mean"),
            mean_mav=("mav", "mean"),
            mean_first_prefix=("first_prefix", "mean"),
            median_first_prefix=("first_prefix", "median"),
            mean_final_threshold=("final_threshold", "mean"),
            mean_final_colluder_max=("final_colluder_max", "mean"),
            mean_final_innocent_max=("final_innocent_max", "mean"),
            mean_final_colluder_max_margin=("final_colluder_max_margin", "mean"),
            median_final_colluder_max_margin=("final_colluder_max_margin", "median"),
            min_final_colluder_max_margin=("final_colluder_max_margin", "min"),
            mean_final_innocent_max_margin=("final_innocent_max_margin", "mean"),
            mean_prefix_max_gap=("prefix_max_gap", "mean"),
        )
        .reset_index()
    )

    overall.to_csv(Path(args.out) / "overall_original_like_summary.csv", index=False)

    make_plots(prefix_df, args.out)

    print("\n=== Overall original-like summary ===")
    print(overall.to_string(index=False))

    print("\n=== Per-case summary ===")
    print(case_df.to_string(index=False))

    print(f"\nSaved outputs to {args.out}/")
    print("Main files:")
    print("  - per_case_per_prefix_trace.csv")
    print("  - per_case_original_like_summary.csv")
    print("  - overall_original_like_summary.csv")
    print("  - per_prefix_average_over_cases.csv")
    print("  - plots/*.png")


if __name__ == "__main__":
    main()