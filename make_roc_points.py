import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

df = pd.read_csv("analysis_auc_short/raw_scores.csv")

rows = []
summary = []

for run, g in df.groupby("run"):
    for score_col in ["score_final", "score_prefix"]:
        y_true = g["guilty"].values
        y_score = g[score_col].values

        fpr, tpr, thr = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)

        summary.append({
            "run": run,
            "score": score_col,
            "auc": auc,
            "n_points": len(g),
            "n_guilty": int(y_true.sum()),
            "n_innocent": int((1-y_true).sum()),
        })

        for a,b,c in zip(fpr,tpr,thr):
            rows.append({
                "run": run,
                "score": score_col,
                "fpr": a,
                "tpr": b,
                "threshold": c,
            })

pd.DataFrame(rows).to_csv("analysis_auc_short/roc_points.csv", index=False)
pd.DataFrame(summary).to_csv("analysis_auc_short/roc_auc_summary.csv", index=False)

print(pd.DataFrame(summary))
print("\nSaved:")
print("analysis_auc_short/roc_points.csv")
print("analysis_auc_short/roc_auc_summary.csv")
