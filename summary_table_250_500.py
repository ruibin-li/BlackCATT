import pandas as pd
from pathlib import Path

items = [
    ("250", "analysis_tardos_prefix_replay_250/overall_original_like_summary.csv"),
    ("500", "analysis_tardos_prefix_replay_500/overall_original_like_summary.csv"),
]

rows = []
for snap, path in items:
    df = pd.read_csv(path)
    df.insert(0, "snapshot", snap)
    rows.append(df)

out = pd.concat(rows, ignore_index=True)

cols = [
    "snapshot", "run",
    "fn_original_like_rate",
    "fp_original_like_rate",
    "mean_mav",
    "mean_first_prefix",
    "median_first_prefix",
    "mean_final_threshold",
    "mean_final_colluder_max",
    "mean_final_innocent_max",
    "mean_final_colluder_max_margin",
    "median_final_colluder_max_margin",
    "min_final_colluder_max_margin",
    "mean_prefix_max_gap",
]

print(out[cols].to_string(index=False))
out[cols].to_csv("analysis_tardos_prefix_replay_snapshot_summary.csv", index=False)

# Also compute IID -> PATH5 deltas per snapshot.
wide = out[cols].pivot(index="snapshot", columns="run")
delta_rows = []

for snap in ["250", "500"]:
    if snap not in wide.index:
        continue
    row = {"snapshot": snap}
    for metric in [
        "fn_original_like_rate",
        "mean_mav",
        "mean_final_colluder_max",
        "mean_final_innocent_max",
        "mean_final_colluder_max_margin",
        "median_final_colluder_max_margin",
        "mean_prefix_max_gap",
    ]:
        iid = wide.loc[snap, (metric, "IID")]
        p5 = wide.loc[snap, (metric, "PATH5")]
        row[f"{metric}_IID"] = iid
        row[f"{metric}_PATH5"] = p5
        row[f"{metric}_delta_PATH5_minus_IID"] = p5 - iid
    delta_rows.append(row)

delta = pd.DataFrame(delta_rows)
delta.to_csv("analysis_tardos_prefix_replay_snapshot_deltas.csv", index=False)

print("\nSaved:")
print("analysis_tardos_prefix_replay_snapshot_summary.csv")
print("analysis_tardos_prefix_replay_snapshot_deltas.csv")