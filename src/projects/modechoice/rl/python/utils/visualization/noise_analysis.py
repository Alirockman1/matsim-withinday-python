"""
How big is the randomisation noise in the step scores, compared with the differences between modes?

Input : agent_tracking.csv  (columns: iteration, agent_id, experienced_modes, trip_scores, ...)
        one row per agent and iteration; experienced_modes and trip_scores hold one ";"-separated
        value per trip (step) of the day.
Output: noise_analysis.png          the figure
        noise_step_mode_table.csv   mean / spread per step and mode
        noise_variance_split.csv    how much of each step's variance comes from what
        noise_mode_pairs.csv        gap between every pair of modes versus the noise

Usage : python noise_analysis.py agent_tracking.csv [--out-dir OUT] [--agent AGENT_ID]
                                 [--step-labels "home>work,work>lunch,lunch>work,work>home"]

Definitions (for every step k, using the mode sequences chosen in the iterations)
  history        the modes chosen at the earlier steps (they change arrival times, and with them the
                 activity part of the step score)
  pure noise     spread between iterations that had the SAME modes up to and including step k.
                 Nothing else differs between them, so this is the randomisation of the simulation.
                 Estimated as the pooled within-group standard deviation (groups with 2+ iterations).
  variance split share of the step's score variance explained by (a) the mode chosen at this step,
                 (b) the history, (c) randomisation noise.
  adjusted gap   difference between two modes at step k after removing the effect of the history
                 (least squares with one set of mode indicators per step up to k).
                 gap_over_noise = |gap| / pure noise.  2+ = clear, 1-2 = marginal, below 1 = blurred.
"""
import argparse
import itertools
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

NOISE_FLOOR = 0.01       # avoids dividing by exactly zero noise (deterministic modes)
RATIO_CAP = 1000.0       # largest ratio shown in the figure
MODE_COLORS = {"car": "#4C72B0", "pt": "#55A868", "bike": "#C44E52", "pedestrian": "#8172B2"}


# ------------------------------------------------------------------------------------ loading
def load_steps(path, agent=None):
    """One row per (iteration, step) with the chosen mode, the score and the mode history."""
    df = pd.read_csv(path)
    if agent is not None:
        df = df[df["agent_id"].astype(str) == str(agent)]
    if df.empty:
        raise SystemExit("No rows found (check --agent).")
    if df["agent_id"].nunique() > 1:
        print(f"Note: {df['agent_id'].nunique()} agents in the file; pass --agent to analyse one. Using all rows pooled.")

    rows = []
    for record in df.itertuples(index=False):
        modes = str(record.experienced_modes).split(";")
        scores = [float(value) for value in str(record.trip_scores).split(";")]
        if len(modes) != len(scores):
            raise SystemExit(f"Iteration {record.iteration}: {len(modes)} modes but {len(scores)} scores.")
        for step, (mode, score) in enumerate(zip(modes, scores)):
            rows.append({
                "iteration": record.iteration,
                "agent_id": record.agent_id,
                "step": step,
                "mode": mode,
                "history": ">".join(modes[:step]),          # earlier modes only
                "prefix": ">".join(modes[:step + 1]),       # earlier modes and this step's mode
                "modes_so_far": modes[:step + 1],
                "score": score,
            })
    return pd.DataFrame(rows), df


# ------------------------------------------------------------------------------------ statistics
def pooled_noise(frame):
    """Pooled within-group std over groups of iterations with identical prefix; (std, degrees of freedom)."""
    sizes = frame.groupby("prefix")["score"].size()
    variances = frame.groupby("prefix")["score"].var(ddof=1)
    usable = sizes >= 2
    degrees = int((sizes[usable] - 1).sum())
    if degrees == 0:
        return float("nan"), 0
    return math.sqrt(float(((sizes[usable] - 1) * variances[usable]).sum()) / degrees), degrees


def step_mode_table(steps):
    table = steps.groupby(["step", "mode"])["score"].agg(n="size", mean="mean", std_total="std").reset_index()
    noise_rows = []
    for (step, mode), frame in steps.groupby(["step", "mode"]):
        noise, degrees = pooled_noise(frame)
        noise_rows.append({"step": step, "mode": mode, "pure_noise_std": noise, "noise_df": degrees})
    return table.merge(pd.DataFrame(noise_rows), on=["step", "mode"])


def variance_split(steps):
    rows = []
    for step, frame in steps.groupby("step"):
        scores = frame["score"].to_numpy()
        n = len(scores)
        total_ss = float(((scores - scores.mean()) ** 2).sum())
        if total_ss == 0 or n < 2:
            rows.append({"step": step, "total_std": 0.0, "pure_noise_std": 0.0, "noise_df": 0,
                         "share_mode": 0.0, "share_history": 0.0, "share_noise": 0.0})
            continue
        mode_ss = float(sum(len(g) * (g["score"].mean() - scores.mean()) ** 2 for _, g in frame.groupby("mode")))
        noise, degrees = pooled_noise(frame)
        total_var = total_ss / (n - 1)
        share_noise = min(1.0, (noise ** 2) / total_var) if not math.isnan(noise) else float("nan")
        share_mode = min(1.0, mode_ss / total_ss)
        share_history = max(0.0, 1.0 - share_mode - (0.0 if math.isnan(share_noise) else share_noise))
        rows.append({"step": step, "total_std": math.sqrt(total_var), "pure_noise_std": noise, "noise_df": degrees,
                     "share_mode": share_mode, "share_history": share_history, "share_noise": share_noise})
    return pd.DataFrame(rows)


def adjusted_pairs(steps, noise_by_step):
    """Mode-vs-mode gaps at each step, adjusted for the earlier modes (additive least squares)."""
    rows = []
    for step, frame in steps.groupby("step"):
        frame = frame.reset_index(drop=True)
        modes_matrix = [m for m in frame["modes_so_far"]]            # list of lists, length step+1 each
        levels = [sorted({m[j] for m in modes_matrix}) for j in range(step + 1)]

        # design matrix: intercept + indicators for every level but the first, per position j
        columns = []                                                   # (position, level)
        for j in range(step + 1):
            for level in levels[j][1:]:
                columns.append((j, level))
        x = np.zeros((len(frame), 1 + len(columns)))
        x[:, 0] = 1.0
        for row, m in enumerate(modes_matrix):
            for c, (j, level) in enumerate(columns, start=1):
                x[row, c] = 1.0 if m[j] == level else 0.0
        y = frame["score"].to_numpy()

        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        residual = y - x @ beta
        dof = max(len(y) - np.linalg.matrix_rank(x), 1)
        covariance = (float(residual @ residual) / dof) * np.linalg.pinv(x.T @ x)

        def contrast(level):
            vector = np.zeros(1 + len(columns))
            if (step, level) in columns:
                vector[1 + columns.index((step, level))] = 1.0
            return vector

        noise = noise_by_step.get(step, float("nan"))
        for a, b in itertools.combinations(levels[step], 2):
            c = contrast(a) - contrast(b)
            gap = float(c @ beta)
            se = math.sqrt(max(float(c @ covariance @ c), 0.0))
            denominator = max(noise, NOISE_FLOOR) if not math.isnan(noise) else float("nan")
            ratio = abs(gap) / denominator if not math.isnan(denominator) else float("nan")
            verdict = ("clear" if ratio >= 2 else "marginal" if ratio >= 1 else "blurred by noise") \
                if not math.isnan(ratio) else "noise unknown"
            rows.append({"step": step, "mode_a": a, "mode_b": b, "adjusted_gap": gap, "gap_std_error": se,
                         "pure_noise_std": noise, "gap_over_noise": ratio, "verdict": verdict})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------- figure
def make_figure(steps, split, pairs, labels, out_path, title):
    n_steps = steps["step"].nunique()
    fig = plt.figure(figsize=(4.2 * n_steps, 10.5))
    grid = fig.add_gridspec(2, n_steps, height_ratios=[1.15, 1.0], hspace=0.38, wspace=0.28)
    rng = np.random.default_rng(1)

    # ---- top row: score of every iteration, per mode, per step
    for step in range(n_steps):
        axis = fig.add_subplot(grid[0, step])
        frame = steps[steps["step"] == step]
        order = list(frame.groupby("mode")["score"].mean().sort_values(ascending=False).index)
        for position, mode in enumerate(order):
            values = frame.loc[frame["mode"] == mode, "score"].to_numpy()
            jitter = rng.uniform(-0.18, 0.18, size=len(values))
            axis.scatter(position + jitter, values, s=16, alpha=0.55, color=MODE_COLORS.get(mode, "grey"), linewidths=0)
            axis.errorbar(position, values.mean(), yerr=values.std(ddof=1) if len(values) > 1 else 0.0,
                          fmt="D", color="black", markersize=5, capsize=4, linewidth=1.3, zorder=5)
            axis.text(position, 0.02, f"n={len(values)}", transform=axis.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
        axis.set_xticks(range(len(order)))
        axis.set_xticklabels(order, fontsize=9)
        axis.set_xlim(-0.6, len(order) - 0.4)
        axis.set_title(labels[step], fontsize=11, fontweight="bold")
        axis.grid(axis="y", alpha=0.25)
        if step == 0:
            axis.set_ylabel("step score (dots: iterations, diamond: mean +/- std)")
        row = split[split["step"] == step].iloc[0]
        noise_text = "n/a" if math.isnan(row["pure_noise_std"]) else f"{row['pure_noise_std']:.2f}"
        axis.text(0.5, 1.0, f"total std {row['total_std']:.1f} | pure noise {noise_text}", transform=axis.transAxes,
                  ha="center", va="top", fontsize=8, bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))

    # ---- bottom left: where does the variance of each step come from?
    half = max(n_steps // 2, 1)
    axis = fig.add_subplot(grid[1, :half])
    labels_short = [labels[s] for s in split["step"]]
    ypos = np.arange(len(split))[::-1]
    parts = [("share_mode", "mode chosen at this step", "#4C72B0"),
             ("share_history", "earlier modes (history)", "#DD8452"),
             ("share_noise", "randomisation noise", "#C44E52")]
    left = np.zeros(len(split))
    for column, name, color in parts:
        widths = np.nan_to_num(split[column].to_numpy()) * 100
        axis.barh(ypos, widths, left=left, color=color, label=name, height=0.6)
        for y, x0, w in zip(ypos, left, widths):
            if w >= 6:
                axis.text(x0 + w / 2, y, f"{w:.0f}%", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        left += widths
    axis.set_yticks(ypos)
    axis.set_yticklabels(labels_short, fontsize=9)
    axis.set_xlim(0, 100)
    axis.set_xlabel("share of the step's score variance (%)")
    axis.set_title("What makes the step score vary?", fontsize=11, fontweight="bold")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8, frameon=False)

    # ---- bottom right: gaps between modes versus the noise
    axis = fig.add_subplot(grid[1, half:])
    verdict_style = {"clear": ("#2E8B57", "o"), "marginal": ("#E0A800", "s"), "blurred by noise": ("#C0392B", "X"),
                     "noise unknown": ("grey", "d")}
    for verdict, (color, marker) in verdict_style.items():
        subset = pairs[pairs["verdict"] == verdict]
        if subset.empty:
            continue
        xs = subset["step"].to_numpy() + rng.uniform(-0.12, 0.12, size=len(subset))
        ys = np.clip(subset["gap_over_noise"].to_numpy(), 0.05, RATIO_CAP)
        axis.scatter(xs, ys, s=46, color=color, marker=marker, label=f"{verdict} ({len(subset)})", alpha=0.9, edgecolors="white", linewidths=0.5)
    axis.set_yscale("log")
    axis.axhline(2, color="#2E8B57", linestyle="--", linewidth=1)
    axis.axhline(1, color="#C0392B", linestyle="--", linewidth=1)
    axis.set_xticks(range(n_steps))
    axis.set_xticklabels([labels[s] for s in range(n_steps)], fontsize=8)
    axis.set_ylabel("|gap between two modes| / noise  (log scale)")
    axis.set_title("Can the agent tell two modes apart?  (one marker per mode pair)", fontsize=11, fontweight="bold")
    axis.legend(fontsize=8, loc="best")
    axis.grid(alpha=0.25)
    axis.text(0.99, 0.02, f"values capped at {RATIO_CAP:.0f}", transform=axis.transAxes, ha="right", fontsize=7, color="grey")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------------------ main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--agent", default=None)
    parser.add_argument("--step-labels", default=None, help='comma-separated, e.g. "home>work,work>lunch,lunch>work,work>home"')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    steps, raw = load_steps(args.csv, args.agent)
    n_steps = steps["step"].nunique()
    if args.step_labels:
        names = [name.strip() for name in args.step_labels.split(",")]
        if len(names) != n_steps:
            raise SystemExit(f"--step-labels needs {n_steps} entries, got {len(names)}.")
        labels = [f"Step {i}: {name}" for i, name in enumerate(names)]
    else:
        labels = [f"Step {i}" for i in range(n_steps)]

    table = step_mode_table(steps)
    split = variance_split(steps)
    noise_by_step = dict(zip(split["step"], split["pure_noise_std"]))
    pairs = adjusted_pairs(steps, noise_by_step)

    table.to_csv(out_dir / "noise_step_mode_table.csv", index=False)
    split.to_csv(out_dir / "noise_variance_split.csv", index=False)
    pairs.to_csv(out_dir / "noise_mode_pairs.csv", index=False)

    agent_text = args.agent if args.agent else ", ".join(map(str, raw["agent_id"].unique()[:3]))
    title = f"Step-score noise vs mode differences  |  agent {agent_text}, {raw['iteration'].nunique()} iterations, random mode choice"
    make_figure(steps, split, pairs, labels, out_dir / "noise_analysis.png", title)

    pd.set_option("display.width", 200)
    print("\n=== Mean and spread per step and mode (std_total includes the effect of earlier modes) ===")
    print(table.round(3).to_string(index=False))
    print("\n=== What makes each step's score vary ===")
    shown = split.copy()
    for column in ["share_mode", "share_history", "share_noise"]:
        shown[column] = (shown[column] * 100).round(1)
    print(shown.round(3).to_string(index=False))
    print("\n=== Mode pairs: adjusted gap versus pure noise ===")
    print(pairs.round(3).to_string(index=False))
    print(f"\nWrote {out_dir / 'noise_analysis.png'} and 3 CSV files.")


if __name__ == "__main__":
    main()
