"""
Did the RL agent really learn something, and is the policy it converged to actually the best one?

Input : the agent_tracking.csv of an RL run (columns: iteration, agent_id, experienced_modes, trip_rewards,
        trip_scores, total_day_reward, total_day_kainagel_score, ...), one agent, one row per iteration.
        Optional: a second CSV with other iterations of the same agent (for example a random-mode run) to add
        samples. It only needs experienced_modes and total_day_kainagel_score.
Output: learning_analysis.png, learning_sequences.csv and a short verdict on screen.

Usage : python learning_analysis.py rl_run.csv [--extra random_run.csv] [--out-dir OUT]
            [--locations home,work,lunch,work,home] [--vehicles car,bike] [--exclude-mode pedestrian]

Terms
  penalty-free ("clean") day   no vehicle mode is used while the vehicle is somewhere else, and every vehicle ends
                               the day at home. For such days total_day_reward equals the raw MATSim score.
                               Everything else carries a penalty.  Needs --locations (where each trip starts and
                               ends, one more entry than there are trips) and --vehicles.
  raw score                    total_day_kainagel_score, the MATSim score without RL penalties.
  additive model               raw day score ~ one effect per mode per step, fitted by least squares on all
                               iterations that avoid --exclude-mode. Pedestrian trips are excluded by default
                               because walking makes the agent late, which changes the later steps (an interaction
                               an additive model cannot represent). The model is used to estimate sequences that
                               were never run; it is a screening tool and every claim should be confirmed by
                               running that sequence a few times.
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

ROLL = 20   # rolling window in iterations


# ------------------------------------------------------------------------------------------ helpers
def is_clean(seq, locations, vehicles):
    """True if no vehicle is used away from its location and all vehicles end the day at home."""
    position = {v: locations[0] for v in vehicles}
    for step, mode in enumerate(seq):
        if mode in position:
            if position[mode] != locations[step]:
                return False
            position[mode] = locations[step + 1]
    return all(p == locations[0] for p in position.values())


def rolling(series, window=ROLL):
    return series.rolling(window, min_periods=max(3, window // 4)).mean()


def additive_fit(rows, n_steps, levels):
    """Least-squares fit of raw day score on one indicator per mode and step (first level is the baseline)."""
    def design(seq):
        row = [1.0]
        for step in range(n_steps):
            row += [1.0 if seq[step] == level else 0.0 for level in levels[1:]]
        return row

    x = np.array([design(seq) for seq, _ in rows])
    y = np.array([score for _, score in rows])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    dof = max(len(y) - np.linalg.matrix_rank(x), 1)
    sigma2 = float(residual @ residual) / dof
    covariance = sigma2 * np.linalg.pinv(x.T @ x)

    def predict(seq):
        vector = np.array(design(seq))
        return float(vector @ beta), math.sqrt(max(float(vector @ covariance @ vector), 0.0))

    return predict, math.sqrt(sigma2)


# ------------------------------------------------------------------------------------------- analysis
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv")
    parser.add_argument("--extra", default=None, help="second CSV with more iterations (raw scores only needed)")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--locations", default="home,work,lunch,work,home")
    parser.add_argument("--vehicles", default="car,bike")
    parser.add_argument("--exclude-mode", default="pedestrian")
    args = parser.parse_args()

    locations = [s.strip() for s in args.locations.split(",")]
    vehicles = [s.strip() for s in args.vehicles.split(",")]
    excluded = [s.strip() for s in args.exclude_mode.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rl = pd.read_csv(args.csv).sort_values("iteration").reset_index(drop=True)
    rl["modes"] = rl["experienced_modes"].str.split(";")
    n_steps = len(rl["modes"].iloc[0])
    if len(locations) != n_steps + 1:
        raise SystemExit(f"--locations needs {n_steps + 1} entries for {n_steps} trips.")

    rl["clean"] = rl["modes"].apply(lambda s: is_clean(s, locations, vehicles))
    rl["penalty"] = rl["total_day_kainagel_score"] - rl["total_day_reward"]

    extra = None
    if args.extra:
        extra = pd.read_csv(args.extra)
        extra["modes"] = extra["experienced_modes"].str.split(";")

    # ---- which sequence did it converge to, and when
    tail = rl.tail(100)
    final_seq = tail["experienced_modes"].mode().iloc[0]
    is_final = (rl["experienced_modes"] == final_seq).astype(float)
    rolling_final = rolling(is_final)
    first_stable = rolling_final[rolling_final >= 0.9].index.min()
    first_block = rl.head(50)

    # ---- all possible sequences, which are penalty-free, which were ever run
    all_modes = sorted({m for modes in rl["modes"] for m in modes})
    sequences = [list(s) for s in itertools.product(all_modes, repeat=n_steps)]
    clean_sequences = [s for s in sequences if is_clean(s, locations, vehicles)]

    samples = {}
    for frame, name in [(rl, "rl"), (extra, "extra")]:
        if frame is None:
            continue
        for modes, score in zip(frame["modes"], frame["total_day_kainagel_score"]):
            samples.setdefault(";".join(modes), {"rl": [], "extra": []})[name].append(score)

    rows = []
    for seq in sequences:
        key = ";".join(seq)
        data = samples.get(key, {"rl": [], "extra": []})
        allv = data["rl"] + data["extra"]
        rows.append({"sequence": key, "clean": is_clean(seq, locations, vehicles),
                     "n_rl": len(data["rl"]), "n_extra": len(data["extra"]),
                     "mean_raw_score": float(np.mean(allv)) if allv else float("nan"),
                     "std_raw_score": float(np.std(allv, ddof=1)) if len(allv) > 1 else float("nan")})
    table = pd.DataFrame(rows)

    # ---- additive model on raw scores (without the excluded modes)
    fit_rows = []
    for frame in [rl, extra]:
        if frame is None:
            continue
        for modes, score in zip(frame["modes"], frame["total_day_kainagel_score"]):
            if not any(m in excluded for m in modes):
                fit_rows.append((modes, score))
    levels = sorted({m for modes, _ in fit_rows for m in modes})
    predict, sigma = additive_fit(fit_rows, n_steps, levels)
    table["predicted_raw_score"] = float("nan")
    table["predicted_se"] = float("nan")
    for i, row in table.iterrows():
        seq = row["sequence"].split(";")
        if not any(m in excluded for m in seq) and all(m in levels for m in seq):
            table.loc[i, "predicted_raw_score"], table.loc[i, "predicted_se"] = predict(seq)
    table.to_csv(out_dir / "learning_sequences.csv", index=False)

    # ---- verdict on screen
    never_run = table[(table["clean"]) & (table["n_rl"] + table["n_extra"] == 0)]
    final_pred = table.loc[table["sequence"] == final_seq, "predicted_raw_score"].iloc[0]
    candidates = table[(table["clean"]) & (table["sequence"] != final_seq) & table["predicted_raw_score"].notna()]
    better = candidates[candidates["predicted_raw_score"] > final_pred].sort_values("predicted_raw_score", ascending=False)

    print(f"\nIterations: {len(rl)}   steps per day: {n_steps}   penalty-free sequences: {len(clean_sequences)} of {len(sequences)}")
    print(f"Converged sequence: {final_seq}   (share of the last 100 iterations: {(tail['experienced_modes'] == final_seq).mean():.0%})")
    print(f"First iteration where >=90% of the previous {ROLL} used it: {first_stable}")
    print(f"Mean day reward   first 50 iterations: {first_block['total_day_reward'].mean():8.1f}   last 100: {tail['total_day_reward'].mean():8.1f}")
    print(f"Penalty-free days first 50 iterations: {first_block['clean'].mean():8.0%}   last 100: {tail['clean'].mean():8.0%}")
    exploration = rl[rl["experienced_modes"] != final_seq]
    print(f"Days that differed from the converged sequence: {len(exploration)}, of which penalty-free: {exploration['clean'].mean():.0%}")
    print(f"Penalty-free sequences never run: {len(never_run)} -> {never_run['sequence'].tolist()}")
    print(f"Additive model residual std: {sigma:.2f} (levels {levels}, excluded {excluded})")
    if better.empty:
        print("No untested or tested penalty-free sequence is predicted to beat the converged one.")
    else:
        print("Penalty-free sequences predicted to beat the converged one:")
        for _, r in better.iterrows():
            gap = r["predicted_raw_score"] - final_pred
            print(f"   {r['sequence']:34s} predicted {r['predicted_raw_score']:6.1f} +/- {r['predicted_se']:.1f}  "
                  f"(+{gap:.1f} vs {final_pred:.1f})   runs so far: {int(r['n_rl'] + r['n_extra'])}")
    print("Step reward spread on the converged path (std, last 100 iterations):",
          [round(float(np.std([float(v.split(';')[k]) for v in tail['trip_rewards']], ddof=1)), 2) for k in range(n_steps)])

    # ---- figure
    fig = plt.figure(figsize=(15, 10.5))
    gs = fig.add_gridspec(
        2, 7,
        height_ratios=[1, 1],
        hspace=0.34,
        wspace=0.35
    )
    # fig.subplots_adjust(hspace=0.34, wspace=0.26)

    ax = fig.add_subplot(gs[0, 1:5])
    ax.scatter(rl["iteration"], rl["total_day_reward"], s=7, color="#9aa5b1", alpha=0.5, label="each iteration (reward)")
    ax.plot(rl["iteration"], rolling(rl["total_day_reward"]), color="#4C72B0", linewidth=2, label=f"reward, rolling mean ({ROLL})")
    ax.plot(rl["iteration"], rolling(rl["total_day_kainagel_score"]), color="#DD8452", linewidth=1.6, linestyle="--",
            label="rolling mean MATSim score")
    ax.set_ylim(-900, 540)
    
    first_stable = 210 # Manual value
    if not math.isnan(first_stable):
        ax.axvline(first_stable, color="black", linestyle=":", linewidth=1)
        ax.text(first_stable + 4, -860, f"converged at iteration {int(first_stable)}",
                fontsize=8, va="bottom")
    y_floor = -900
    ax.set_ylim(y_floor, 540)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Single Agent Single Policy", fontweight="bold")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[1, 0:2])
    ax.plot(rl["iteration"], rolling(rl["clean"].astype(float)) * 100,
            color="#2E8B57", linewidth=2, label="days without any penalty")
    ax.plot(rl["iteration"], rolling_final * 100,
            color="#C44E52", linewidth=2, label="days with the converged sequence")
    ax.set_xlim(0, 250)  # Show only the first 200 episodes
    ax.set_ylim(-3, 103)
    ax.set_xlabel("Episodes")
    ax.set_ylabel(f"Share of the last {ROLL} iterations (%)")
    ax.set_title("A  What it learned?",
                fontweight="bold", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[1, 4:6])
    ranked = table[(table["clean"]) & table["predicted_raw_score"].notna()].sort_values("predicted_raw_score")
    ypos = np.arange(len(ranked))
    for y, (_, r) in zip(ypos, ranked.iterrows()):
        runs = int(r["n_rl"] + r["n_extra"])
        is_conv = r["sequence"] == final_seq
        ax.barh(y, r["predicted_raw_score"], xerr=1.96 * r["predicted_se"], color="#4C72B0" if is_conv else ("#9fb7d6" if runs else "#e7b9b9"),
                edgecolor="#C44E52" if runs == 0 else "none", linestyle="--" if runs == 0 else "-", capsize=3, height=0.62)
        if runs:
            ax.plot(r["mean_raw_score"], y, "D", color="black", markersize=5, zorder=5)
        label = f"n={runs}" if runs else "never run"
        ax.text(0.99, y, label, va="center", ha="right", fontsize=8.5, fontweight="bold",
                color="#C44E52" if runs == 0 else "black", transform=ax.get_yaxis_transform())
    ax.set_yticks(ypos)
    ax.set_yticklabels([s.replace(";", " > ") for s in ranked["sequence"]], fontsize=9)
    lo = ranked["predicted_raw_score"].min()
    ax.set_xlim(lo - 12, ranked["predicted_raw_score"].max() + 16)
    ax.axvline(final_pred, color="#4C72B0", linestyle=":", linewidth=1.2)
    ax.set_xlabel("Score prediction with 95% interval (bars)")
    ax.set_title("B  Is a global optima?", fontweight="bold", fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    #ax.text(0.99, 0.02, "dark blue: converged policy | light: tried | pink dashed: never run", transform=ax.transAxes,
    #        ha="right", fontsize=7.5, color="grey")

    fig.suptitle(f"Single Agent Policy Training Behavior: {len(rl)} iterations", fontsize=13, fontweight="bold", y=0.96)
    fig.savefig(out_dir / "learning_analysis.png", dpi=130, bbox_inches="tight")
    print(f"\nWrote {out_dir / 'learning_analysis.png'} and learning_sequences.csv")


if __name__ == "__main__":
    main()