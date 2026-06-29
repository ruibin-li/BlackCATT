import argparse
import itertools
import numpy as np
import pandas as pd


def cosine(a, b, eps=1e-12):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="saved_client_clean_functional_residual.csv")
    parser.add_argument("--out", type=str, default="proxy_clean_collusion_match.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv).sort_values("client")
    clients = df["client"].astype(int).tolist()

    # Per-class clean residual vector.
    # Use both loss and acc residual, or only loss.
    loss_cols = [f"delta_loss_c{c}" for c in range(10)]
    acc_cols = [f"delta_acc_c{c}" for c in range(10)]

    X_loss = {
        int(row["client"]): row[loss_cols].to_numpy(dtype=float)
        for _, row in df.iterrows()
    }
    X_acc = {
        int(row["client"]): row[acc_cols].to_numpy(dtype=float)
        for _, row in df.iterrows()
    }

    rows = []

    for a, b in itertools.combinations(clients, 2):
        # Proxy collusion residual vector.
        coll_loss = 0.5 * (X_loss[a] + X_loss[b])
        coll_acc = 0.5 * (X_acc[a] + X_acc[b])

        scores = []
        for u in clients:
            sim_loss = cosine(coll_loss, X_loss[u])
            sim_acc = cosine(coll_acc, X_acc[u])

            # Combined score. Loss and acc are separate views.
            score = 0.5 * sim_loss + 0.5 * sim_acc

            scores.append({
                "colluder_a": a,
                "colluder_b": b,
                "candidate": u,
                "is_guilty": int(u in [a, b]),
                "sim_loss": sim_loss,
                "sim_acc": sim_acc,
                "score": score,
            })

        scores_sorted = sorted(scores, key=lambda x: x["score"], reverse=True)

        for rank, item in enumerate(scores_sorted, start=1):
            item["rank"] = rank
            rows.append(item)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)

    # Summary: top-2 hit rate.
    pair_rows = []
    for (a, b), g in out.groupby(["colluder_a", "colluder_b"]):
        top2 = g.sort_values("rank").head(2)
        hit_count = int(top2["is_guilty"].sum())
        top1_hit = int(top2.iloc[0]["is_guilty"])

        pair_rows.append({
            "colluder_a": a,
            "colluder_b": b,
            "top1_hit": top1_hit,
            "top2_hit_count": hit_count,
            "top2_both_hit": int(hit_count == 2),
        })

    summary = pd.DataFrame(pair_rows)

    print("\n[Proxy clean collusion matching]")
    print("num pairs =", len(summary))
    print("top1 hit rate =", summary["top1_hit"].mean())
    print("top2 average guilty count =", summary["top2_hit_count"].mean())
    print("top2 both-hit rate =", summary["top2_both_hit"].mean())

    print("\nSaved:", args.out)


if __name__ == "__main__":
    main()