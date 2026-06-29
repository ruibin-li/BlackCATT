import argparse
import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def normalize_name(x):
    return str(x).strip().lower()


def read_csv_smart(path):
    """
    Read csv with or without header.
    If header looks uninformative, reread as headerless.
    """
    df1 = pd.read_csv(path)
    cols = [normalize_name(c) for c in df1.columns]
    useful = any(
        any(k in c for k in ["run", "setting", "score", "guilty", "label", "threshold", "round", "client", "user"])
        for c in cols
    )

    if useful:
        df = df1.copy()
    else:
        df = pd.read_csv(path, header=None)

        n = df.shape[1]
        if n == 7:
            df.columns = [
                "setting",
                "user",
                "maybe_guilty",
                "aux1",
                "aux2",
                "score_final",
                "score_prefix",
            ]
        elif n == 6:
            df.columns = [
                "setting",
                "user",
                "maybe_guilty",
                "aux1",
                "score_final",
                "score_prefix",
            ]
        else:
            df.columns = [f"col{i}" for i in range(n)]

    df.columns = [str(c).strip() for c in df.columns]
    return df


def guess_setting_col(df):
    candidates = []
    for c in df.columns:
        s = df[c].astype(str).str.upper()
        hit = s.str.contains("IID|PATH|P5", regex=True, na=False).mean()
        if hit > 0.2:
            candidates.append((hit, c))
    if candidates:
        return sorted(candidates, reverse=True)[0][1]

    for name in ["setting", "run", "split", "condition", "case", "dataset"]:
        for c in df.columns:
            if name in normalize_name(c):
                return c

    return None


def guess_score_col(df, user_score_col=None):
    if user_score_col and user_score_col in df.columns:
        return user_score_col

    priority = [
        "score_final",
        "final_score",
        "score",
        "final",
        "score_prefix",
        "prefix_score",
    ]

    lower_map = {normalize_name(c): c for c in df.columns}
    for p in priority:
        if p in lower_map:
            return lower_map[p]

    for c in df.columns:
        lc = normalize_name(c)
        if "score" in lc and "threshold" not in lc:
            return c

    numeric_cols = []
    for c in df.columns:
        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().mean() > 0.8:
            numeric_cols.append(c)

    if not numeric_cols:
        raise ValueError("Cannot find score column. Please pass --score-col.")

    # fallback: use last numeric column
    return numeric_cols[-1]


def guess_guilty_col(df, score_col, user_guilty_col=None):
    if user_guilty_col and user_guilty_col in df.columns:
        return user_guilty_col

    for name in ["is_guilty", "guilty", "is_colluder", "colluder", "label", "y"]:
        for c in df.columns:
            if normalize_name(c) == name:
                return c

    score = pd.to_numeric(df[score_col], errors="coerce")
    candidates = []

    for c in df.columns:
        if c == score_col:
            continue

        x = pd.to_numeric(df[c], errors="coerce")
        vals = set(x.dropna().unique())

        # binary-like column
        if vals.issubset({0, 1, 0.0, 1.0}) and len(vals) == 2:
            m1 = score[x == 1].mean()
            m0 = score[x == 0].mean()
            gap = m1 - m0
            candidates.append((gap, c, m1, m0))

    if not candidates:
        raise ValueError("Cannot find guilty label column. Please pass --guilty-col.")

    # choose the binary column where value=1 has largest positive score separation
    candidates = sorted(candidates, reverse=True)
    best_gap, best_col, m1, m0 = candidates[0]

    print(f"[auto] guessed guilty column = {best_col}, mean(score|1)={m1:.4f}, mean(score|0)={m0:.4f}, gap={best_gap:.4f}")
    return best_col


def guess_user_col(df):
    for name in ["user", "user_id", "client", "client_id", "cid", "id"]:
        for c in df.columns:
            if normalize_name(c) == name:
                return c

    # fallback: first integer-like column with several unique values
    for c in df.columns:
        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().mean() > 0.8:
            unique = x.nunique()
            if unique >= 3:
                return c

    return None


def guess_threshold_col(df, user_threshold_col=None):
    if user_threshold_col and user_threshold_col in df.columns:
        return user_threshold_col

    for c in df.columns:
        lc = normalize_name(c)
        if any(k in lc for k in ["threshold", "thres", "tau", "bound"]):
            return c

    return None


def binary_auc_manual(y_true, scores):
    """
    AUC via rank statistic.
    Handles ties by average ranks.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)

    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]

    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)

    if n_pos == 0 or n_neg == 0:
        return np.nan

    ranks = pd.Series(s).rank(method="average").values
    sum_pos = ranks[y == 1].sum()

    auc = (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def qstats(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    if len(x) == 0:
        return {
            "n": 0,
            "min": np.nan,
            "q05": np.nan,
            "q25": np.nan,
            "median": np.nan,
            "mean": np.nan,
            "q75": np.nan,
            "q95": np.nan,
            "max": np.nan,
            "std": np.nan,
        }

    return {
        "n": int(len(x)),
        "min": float(x.min()),
        "q05": float(x.quantile(0.05)),
        "q25": float(x.quantile(0.25)),
        "median": float(x.median()),
        "mean": float(x.mean()),
        "q75": float(x.quantile(0.75)),
        "q95": float(x.quantile(0.95)),
        "max": float(x.max()),
        "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
    }


def make_score_tables(df, setting_col, guilty_col, score_col):
    rows = []
    gap_rows = []

    df = df.copy()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df[guilty_col] = pd.to_numeric(df[guilty_col], errors="coerce").astype("Int64")

    for setting, sub in df.groupby(setting_col):
        sub = sub.dropna(subset=[score_col, guilty_col])
        guilty = sub[sub[guilty_col] == 1][score_col]
        innocent = sub[sub[guilty_col] == 0][score_col]

        gs = qstats(guilty)
        ins = qstats(innocent)

        rows.append({"setting": setting, "group": "guilty", **gs})
        rows.append({"setting": setting, "group": "innocent", **ins})

        if len(guilty) and len(innocent):
            mean_gap = guilty.mean() - innocent.mean()
            top_gap = guilty.max() - innocent.max()
            hard_margin = guilty.min() - innocent.max()
            auc = binary_auc_manual(sub[guilty_col].astype(int), sub[score_col])

            gap_rows.append({
                "setting": setting,
                "guilty_mean": float(guilty.mean()),
                "innocent_mean": float(innocent.mean()),
                "mean_gap": float(mean_gap),
                "max_guilty": float(guilty.max()),
                "max_innocent": float(innent_max := innocent.max()),
                "top_gap_max_guilty_minus_max_innocent": float(top_gap),
                "min_guilty": float(guilty.min()),
                "hard_margin_min_guilty_minus_max_innocent": float(hard_margin),
                "auc": float(auc),
                "n_guilty": int(len(guilty)),
                "n_innocent": int(len(innocent)),
            })

    return pd.DataFrame(rows), pd.DataFrame(gap_rows)


def load_thresholds(args, df, setting_col, threshold_col):
    """
    Return dict: setting -> threshold.
    Priority:
    1. threshold column in raw df
    2. threshold csv
    3. command line --threshold-iid / --threshold-path5
    """
    thresholds = {}

    if threshold_col is not None:
        tmp = df[[setting_col, threshold_col]].copy()
        tmp[threshold_col] = pd.to_numeric(tmp[threshold_col], errors="coerce")
        tmp = tmp.dropna()
        if len(tmp):
            for setting, sub in tmp.groupby(setting_col):
                thresholds[str(setting)] = float(sub[threshold_col].iloc[-1])

    if args.threshold_csv:
        th = pd.read_csv(args.threshold_csv)
        th.columns = [str(c).strip() for c in th.columns]

        th_setting = None
        th_value = None

        for c in th.columns:
            if normalize_name(c) in ["setting", "run", "condition", "case"]:
                th_setting = c
            if any(k in normalize_name(c) for k in ["threshold", "thres", "tau"]):
                th_value = c

        if th_setting is None or th_value is None:
            raise ValueError("threshold csv needs columns like setting/run and threshold/tau.")

        for _, row in th.iterrows():
            thresholds[str(row[th_setting])] = float(row[th_value])

    if args.threshold_iid is not None:
        thresholds["IID"] = float(args.threshold_iid)

    if args.threshold_path5 is not None:
        thresholds["Path5"] = float(args.threshold_path5)
        thresholds["PATH5"] = float(args.threshold_path5)
        thresholds["P5"] = float(args.threshold_path5)

    return thresholds


def setting_to_threshold(setting, thresholds):
    s = str(setting)
    su = s.upper()

    if s in thresholds:
        return thresholds[s]
    if su in thresholds:
        return thresholds[su]

    if "IID" in su and "IID" in thresholds:
        return thresholds["IID"]
    if ("PATH" in su or "P5" in su) and "Path5" in thresholds:
        return thresholds["Path5"]
    if ("PATH" in su or "P5" in su) and "PATH5" in thresholds:
        return thresholds["PATH5"]

    return np.nan


def make_margin_table(df, setting_col, guilty_col, score_col, thresholds):
    if not thresholds:
        return None, None

    tmp = df.copy()
    tmp[score_col] = pd.to_numeric(tmp[score_col], errors="coerce")
    tmp[guilty_col] = pd.to_numeric(tmp[guilty_col], errors="coerce").astype("Int64")
    tmp["threshold_used"] = tmp[setting_col].apply(lambda x: setting_to_threshold(x, thresholds))
    tmp["margin"] = tmp[score_col] - tmp["threshold_used"]
    tmp["above_threshold"] = tmp["margin"] >= 0

    rows = []
    count_rows = []

    for (setting, group_value), sub in tmp.groupby([setting_col, guilty_col]):
        group_name = "guilty" if int(group_value) == 1 else "innocent"
        stats = qstats(sub["margin"])
        rows.append({
            "setting": setting,
            "group": group_name,
            "threshold": setting_to_threshold(setting, thresholds),
            **stats,
        })

        count_rows.append({
            "setting": setting,
            "group": group_name,
            "threshold": setting_to_threshold(setting, thresholds),
            "n": int(len(sub)),
            "n_above_threshold": int(sub["above_threshold"].sum()),
            "rate_above_threshold": float(sub["above_threshold"].mean()) if len(sub) else np.nan,
        })

    return pd.DataFrame(rows), pd.DataFrame(count_rows)


def plot_score_hist(df, setting_col, guilty_col, score_col, thresholds, out_dir):
    plot_dir = os.path.join(out_dir, "plots")
    ensure_dir(plot_dir)

    df = df.copy()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df[guilty_col] = pd.to_numeric(df[guilty_col], errors="coerce").astype("Int64")

    for setting, sub in df.groupby(setting_col):
        sub = sub.dropna(subset=[score_col, guilty_col])
        guilty = sub[sub[guilty_col] == 1][score_col]
        innocent = sub[sub[guilty_col] == 0][score_col]

        plt.figure(figsize=(8, 5))
        plt.hist(innocent, bins=20, alpha=0.6, label="innocent")
        plt.hist(guilty, bins=20, alpha=0.6, label="guilty")

        th = setting_to_threshold(setting, thresholds) if thresholds else np.nan
        if np.isfinite(th):
            plt.axvline(th, linestyle="--", linewidth=2, label=f"threshold={th:.3f}")

        plt.title(f"{setting}: {score_col} distribution")
        plt.xlabel(score_col)
        plt.ylabel("count")
        plt.legend()
        plt.tight_layout()

        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(setting))
        plt.savefig(os.path.join(plot_dir, f"{safe}_{score_col}_hist.png"), dpi=200)
        plt.close()


def plot_margin_hist(df, setting_col, guilty_col, score_col, thresholds, out_dir):
    if not thresholds:
        return

    plot_dir = os.path.join(out_dir, "plots")
    ensure_dir(plot_dir)

    tmp = df.copy()
    tmp[score_col] = pd.to_numeric(tmp[score_col], errors="coerce")
    tmp[guilty_col] = pd.to_numeric(tmp[guilty_col], errors="coerce").astype("Int64")
    tmp["threshold_used"] = tmp[setting_col].apply(lambda x: setting_to_threshold(x, thresholds))
    tmp["margin"] = tmp[score_col] - tmp["threshold_used"]

    for setting, sub in tmp.groupby(setting_col):
        sub = sub.dropna(subset=["margin", guilty_col])
        if not len(sub):
            continue

        guilty = sub[sub[guilty_col] == 1]["margin"]
        innocent = sub[sub[guilty_col] == 0]["margin"]

        plt.figure(figsize=(8, 5))
        plt.hist(innocent, bins=20, alpha=0.6, label="innocent margin")
        plt.hist(guilty, bins=20, alpha=0.6, label="guilty margin")
        plt.axvline(0, linestyle="--", linewidth=2, label="decision boundary")

        plt.title(f"{setting}: margin = {score_col} - threshold")
        plt.xlabel("margin")
        plt.ylabel("count")
        plt.legend()
        plt.tight_layout()

        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(setting))
        plt.savefig(os.path.join(plot_dir, f"{safe}_{score_col}_margin_hist.png"), dpi=200)
        plt.close()


def maybe_crossing_analysis(df, setting_col, guilty_col, score_col, thresholds, out_dir):
    """
    If the csv is long-format and has round + threshold,
    compute first crossing round per user.
    """
    round_col = None
    for c in df.columns:
        if normalize_name(c) in ["round", "t", "step", "prefix", "prefix_len", "iteration"]:
            round_col = c
            break

    user_col = guess_user_col(df)

    if round_col is None or user_col is None:
        return None

    th_col = guess_threshold_col(df)

    tmp = df.copy()
    tmp[score_col] = pd.to_numeric(tmp[score_col], errors="coerce")
    tmp[guilty_col] = pd.to_numeric(tmp[guilty_col], errors="coerce").astype("Int64")
    tmp[round_col] = pd.to_numeric(tmp[round_col], errors="coerce")

    if th_col is not None:
        tmp["threshold_used"] = pd.to_numeric(tmp[th_col], errors="coerce")
    elif thresholds:
        tmp["threshold_used"] = tmp[setting_col].apply(lambda x: setting_to_threshold(x, thresholds))
    else:
        return None

    tmp["above_threshold"] = tmp[score_col] >= tmp["threshold_used"]

    rows = []

    for (setting, user), sub in tmp.groupby([setting_col, user_col]):
        sub = sub.sort_values(round_col)
        is_guilty = int(pd.to_numeric(sub[guilty_col], errors="coerce").dropna().iloc[0])
        crossed = sub[sub["above_threshold"]]

        if len(crossed):
            first_cross = float(crossed[round_col].iloc[0])
            final_margin = float((sub[score_col] - sub["threshold_used"]).iloc[-1])
            crossed_flag = True
        else:
            first_cross = np.nan
            final_margin = float((sub[score_col] - sub["threshold_used"]).iloc[-1])
            crossed_flag = False

        rows.append({
            "setting": setting,
            "user": user,
            "is_guilty": is_guilty,
            "crossed": crossed_flag,
            "first_crossing_round": first_cross,
            "final_margin": final_margin,
        })

    cross = pd.DataFrame(rows)
    cross.to_csv(os.path.join(out_dir, "crossing_by_user.csv"), index=False)

    summary = (
        cross
        .groupby(["setting", "is_guilty"])
        .agg(
            n=("user", "count"),
            n_crossed=("crossed", "sum"),
            crossing_rate=("crossed", "mean"),
            median_first_crossing_round=("first_crossing_round", "median"),
            mean_final_margin=("final_margin", "mean"),
            min_final_margin=("final_margin", "min"),
        )
        .reset_index()
    )

    summary["group"] = summary["is_guilty"].map({1: "guilty", 0: "innocent"})
    summary.to_csv(os.path.join(out_dir, "crossing_summary.csv"), index=False)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, default="analysis_auc_short/raw_scores.csv")
    parser.add_argument("--out", type=str, default="analysis_threshold_scores")

    parser.add_argument("--setting-col", type=str, default=None)
    parser.add_argument("--guilty-col", type=str, default=None)
    parser.add_argument("--score-col", type=str, default=None)
    parser.add_argument("--threshold-col", type=str, default=None)

    parser.add_argument("--threshold-csv", type=str, default=None)
    parser.add_argument("--threshold-iid", type=float, default=None)
    parser.add_argument("--threshold-path5", type=float, default=None)

    args = parser.parse_args()
    ensure_dir(args.out)

    df = read_csv_smart(args.raw)

    setting_col = args.setting_col or guess_setting_col(df)
    if setting_col is None:
        raise ValueError("Cannot find setting/run column. Please pass --setting-col.")

    score_col = guess_score_col(df, args.score_col)
    guilty_col = guess_guilty_col(df, score_col, args.guilty_col)
    threshold_col = guess_threshold_col(df, args.threshold_col)

    print("\n=== Column choices ===")
    print("raw file      :", args.raw)
    print("setting_col   :", setting_col)
    print("guilty_col    :", guilty_col)
    print("score_col     :", score_col)
    print("threshold_col :", threshold_col)

    score_summary, gap_table = make_score_tables(df, setting_col, guilty_col, score_col)
    score_summary.to_csv(os.path.join(args.out, "score_distribution_summary.csv"), index=False)
    gap_table.to_csv(os.path.join(args.out, "gap_table.csv"), index=False)

    thresholds = load_thresholds(args, df, setting_col, threshold_col)
    print("\n=== Thresholds ===")
    if thresholds:
        for k, v in thresholds.items():
            print(f"{k}: {v:.6f}")
    else:
        print("No threshold found/provided. Margin analysis will be skipped.")

    margin_summary, threshold_counts = make_margin_table(df, setting_col, guilty_col, score_col, thresholds)

    if margin_summary is not None:
        margin_summary.to_csv(os.path.join(args.out, "margin_distribution_summary.csv"), index=False)
        threshold_counts.to_csv(os.path.join(args.out, "threshold_counts.csv"), index=False)

    plot_score_hist(df, setting_col, guilty_col, score_col, thresholds, args.out)
    plot_margin_hist(df, setting_col, guilty_col, score_col, thresholds, args.out)

    crossing_summary = maybe_crossing_analysis(df, setting_col, guilty_col, score_col, thresholds, args.out)

    print("\n=== Gap table ===")
    print(gap_table.to_string(index=False))

    if threshold_counts is not None:
        print("\n=== Threshold crossing counts ===")
        print(threshold_counts.to_string(index=False))

    if crossing_summary is not None:
        print("\n=== Dynamic crossing summary ===")
        print(crossing_summary.to_string(index=False))

    print(f"\nSaved outputs to: {args.out}/")
    print("Main files:")
    print("  - score_distribution_summary.csv")
    print("  - gap_table.csv")
    print("  - margin_distribution_summary.csv, if threshold exists")
    print("  - threshold_counts.csv, if threshold exists")
    print("  - crossing_summary.csv, if dynamic/prefix columns exist")
    print("  - plots/*.png")


if __name__ == "__main__":
    main()