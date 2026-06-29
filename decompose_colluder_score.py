import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch

import blackcatt.wm_config as C
from blackcatt.models import ResNet, ResidualBlock
from blackcatt.wm_task import load_collusion, _normalize_triggers
from replay_tardos_prefix_trace import align_code_and_psecret, score_increment_matrix, threshold_by_prefix


def run_setting(run_name, folder, before_cid, device="cuda"):
    folder = Path(folder)
    trig = _normalize_triggers(np.load(folder / "triggers.npy"), device)

    V, P = align_code_and_psecret()
    m = V.shape[1]
    Z_final = threshold_by_prefix(m, C.tau, 1e-6)[-1]

    rows = []
    case_rows = []

    for cid in range(C.n_users):
        col = (cid + 1) % C.n_users

        net = ResNet(ResidualBlock).to(device).eval()
        net = load_collusion(
            [cid, col],
            net,
            folder=str(folder) + "/",
            before_cid=before_cid,
            after_cid=".pkl",
            device=device,
        )

        with torch.no_grad():
            y = net(trig).argmax(1).detach().cpu().numpy()

        S = score_increment_matrix(y, V, P)
        final_scores = S.sum(axis=1)

        colluders = [cid, col]
        winner = colluders[int(np.argmax(final_scores[colluders]))]
        winner_score = final_scores[winner]

        pos_mass = 0.0
        neg_penalty = 0.0
        hit_count = 0
        pair_hit_count = 0

        for t in range(m):
            p = float(P[t, y[t]])
            p = min(max(p, 1e-12), 1 - 1e-12)

            plus = float(np.sqrt((1 - p) / p))
            penalty = float(np.sqrt(p / (1 - p)))

            matched = int(V[winner, t] == y[t])
            pair_hit = int(y[t] in V[colluders, t])

            inc = float(S[winner, t])

            if inc >= 0:
                pos_mass += inc
            else:
                neg_penalty += -inc

            hit_count += matched
            pair_hit_count += pair_hit

            rows.append({
                "run": run_name,
                "cid": cid,
                "col": col,
                "winner": winner,
                "t": t,
                "y": int(y[t]),
                "winner_code": int(V[winner, t]),
                "matched": matched,
                "pair_hit": pair_hit,
                "p_y": p,
                "plus_weight": plus,
                "penalty_weight": penalty,
                "increment": inc,
            })

        case_rows.append({
            "run": run_name,
            "cid": cid,
            "col": col,
            "winner": winner,
            "winner_score": float(winner_score),
            "threshold": float(Z_final),
            "margin": float(winner_score - Z_final),
            "hit_rate": hit_count / m,
            "pair_hit_rate": pair_hit_count / m,
            "positive_mass": float(pos_mass),
            "negative_penalty": float(neg_penalty),
            "net_score_check": float(pos_mass - neg_penalty),
        })

        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return rows, case_rows


def main():
    IID_DIR = os.environ["IID_DIR"]
    P5_DIR = os.environ["P5_DIR"]

    out_dir = Path("analysis_colluder_score_decomposition_500")
    out_dir.mkdir(exist_ok=True)

    all_rows = []
    all_cases = []

    for run_name, folder in [("IID", IID_DIR), ("PATH5", P5_DIR)]:
        rows, cases = run_setting(
            run_name,
            folder,
            before_cid="500_client_status_",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        all_rows.extend(rows)
        all_cases.extend(cases)

    seg = pd.DataFrame(all_rows)
    cases = pd.DataFrame(all_cases)

    # Weight bins based on plus_weight over all rows.
    seg["weight_bin"] = pd.qcut(seg["plus_weight"], 5, labels=False, duplicates="drop")

    by_bin = (
        seg.groupby(["run", "weight_bin"])
        .agg(
            n=("increment", "count"),
            hit_rate=("matched", "mean"),
            pair_hit_rate=("pair_hit", "mean"),
            mean_plus_weight=("plus_weight", "mean"),
            positive_mass=("increment", lambda x: x[x > 0].sum()),
            negative_penalty=("increment", lambda x: -x[x < 0].sum()),
            net_contribution=("increment", "sum"),
        )
        .reset_index()
    )

    overall = (
        cases.groupby("run")
        .agg(
            mean_winner_score=("winner_score", "mean"),
            mean_margin=("margin", "mean"),
            median_margin=("margin", "median"),
            mean_hit_rate=("hit_rate", "mean"),
            mean_pair_hit_rate=("pair_hit_rate", "mean"),
            mean_positive_mass=("positive_mass", "mean"),
            mean_negative_penalty=("negative_penalty", "mean"),
        )
        .reset_index()
    )

    seg.to_csv(out_dir / "per_trigger_contribution.csv", index=False)
    cases.to_csv(out_dir / "per_case_decomposition.csv", index=False)
    by_bin.to_csv(out_dir / "decomposition_by_weight_bin.csv", index=False)
    overall.to_csv(out_dir / "overall_decomposition.csv", index=False)

    print("\n=== Overall decomposition ===")
    print(overall.to_string(index=False))

    print("\n=== By weight bin ===")
    print(by_bin.to_string(index=False))

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()