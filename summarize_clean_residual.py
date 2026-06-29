import argparse
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default="saved_client_clean_functional_residual.csv",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="clean_residual_active_inactive_summary.csv",
    )
    parser.add_argument(
        "--top-out",
        type=str,
        default="clean_residual_top_classes.csv",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    rows = []
    top_rows = []

    for _, row in df.iterrows():
        cid = int(row["client"])

        active = []
        inactive = []

        for c in range(10):
            cnt_col = f"local_train_count_c{c}"
            if cnt_col not in df.columns:
                raise RuntimeError(
                    f"Missing column {cnt_col}. "
                    "Re-run eval_saved_client_clean_residual.py without --skip-client-dist."
                )

            if row[cnt_col] > 0:
                active.append(c)
            else:
                inactive.append(c)

        active_delta_loss = np.array([row[f"delta_loss_c{c}"] for c in active], dtype=float)
        inactive_delta_loss = np.array([row[f"delta_loss_c{c}"] for c in inactive], dtype=float)

        active_delta_acc = np.array([row[f"delta_acc_c{c}"] for c in active], dtype=float)
        inactive_delta_acc = np.array([row[f"delta_acc_c{c}"] for c in inactive], dtype=float)

        active_mean_loss = active_delta_loss.mean()
        inactive_mean_loss = inactive_delta_loss.mean()

        active_mean_acc = active_delta_acc.mean()
        inactive_mean_acc = inactive_delta_acc.mean()

        rows.append({
            "client": cid,
            "active_classes": str(active),
            "inactive_classes": str(inactive),

            "delta_clean_loss": row["delta_clean_loss"],
            "delta_clean_acc": row["delta_clean_acc"],

            "active_mean_delta_loss": active_mean_loss,
            "inactive_mean_delta_loss": inactive_mean_loss,
            "active_minus_inactive_delta_loss": active_mean_loss - inactive_mean_loss,

            "active_mean_delta_acc": active_mean_acc,
            "inactive_mean_delta_acc": inactive_mean_acc,
            "active_minus_inactive_delta_acc": active_mean_acc - inactive_mean_acc,
        })

        # Per-class ranking for inspection.
        class_items = []
        for c in range(10):
            class_items.append({
                "client": cid,
                "class": c,
                "is_active": c in active,
                "local_train_count": row[f"local_train_count_c{c}"],
                "delta_loss": row[f"delta_loss_c{c}"],
                "delta_acc": row[f"delta_acc_c{c}"],
            })

        class_items_sorted_loss = sorted(class_items, key=lambda x: x["delta_loss"])
        class_items_sorted_acc = sorted(class_items, key=lambda x: x["delta_acc"], reverse=True)

        for rank, item in enumerate(class_items_sorted_loss[:3], start=1):
            top_rows.append({
                **item,
                "rank_type": "best_loss_decrease",
                "rank": rank,
            })

        for rank, item in enumerate(class_items_sorted_loss[-3:], start=1):
            top_rows.append({
                **item,
                "rank_type": "worst_loss_increase",
                "rank": rank,
            })

        for rank, item in enumerate(class_items_sorted_acc[:3], start=1):
            top_rows.append({
                **item,
                "rank_type": "best_acc_increase",
                "rank": rank,
            })

        for rank, item in enumerate(class_items_sorted_acc[-3:], start=1):
            top_rows.append({
                **item,
                "rank_type": "worst_acc_decrease",
                "rank": rank,
            })

    out = pd.DataFrame(rows)
    top = pd.DataFrame(top_rows)

    out.to_csv(args.out, index=False)
    top.to_csv(args.top_out, index=False)

    print("\n[Active vs inactive summary]")
    print(out.to_string(index=False))

    print("\n[Aggregate means]")
    print("mean active_minus_inactive_delta_loss =",
          out["active_minus_inactive_delta_loss"].mean())
    print("mean active_minus_inactive_delta_acc  =",
          out["active_minus_inactive_delta_acc"].mean())

    print("\nSaved:")
    print(" ", args.out)
    print(" ", args.top_out)

    print("\nInterpretation:")
    print("active_minus_inactive_delta_loss < 0:")
    print("  active classes are relatively better than inactive classes.")
    print("active_minus_inactive_delta_acc > 0:")
    print("  active classes are relatively better than inactive classes.")


if __name__ == "__main__":
    main()