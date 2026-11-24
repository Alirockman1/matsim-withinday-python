"""
Standalone plotting script: MATSim baseline vs random mode choice vs RL single policy.
TUM corporate colours (blue / orange / gray) and TUM-style typography.

Figures (PNG + PDF each) written to --out-dir:
  01  score per iteration, one unsmoothed subplot per model (random / MATSim / RL)
  02  cumulative regret vs best plan (all runs)
  03  RL policy training: reward curve and accumulated dQ, two plots in a column
  04  mode choice evaluation: random vs RL (mode share per trip; score vs random expectation)
  summary_by_phase.csv  (table, not a figure)

DATA INTEGRITY: nothing is rewritten by default. Score adjustment, baseline
freezing and extrapolation are opt-in (--adjust-all-bike, --freeze-at N,
--extrapolate) and print a warning. Missing files raise an error.
"""

import argparse
from pathlib import Path
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
 
# =============================================================================
# 1. CONSTANTS & CONFIGURATION
# =============================================================================
 
DATA_DIR = (
    "C:\\ResearchWork\\Matsim_integration\\Local\\matsim-withinday-python\\"
    "scenarios\\sioux_falls\\output\\tmp\\sensitivity_analysis"
)
 
# --- TUM corporate design palette -------------------------------------------
TUM_BLUE = "#0065BD"        # primary blue (Pantone 300)
TUM_BLUE_DARK = "#003359"   # dark blue, used for headings
TUM_BLUE_LIGHT = "#98C6EA"  # secondary light blue (only for the 4th mode colour)
TUM_ORANGE = "#E37222"      # accent orange
TUM_GRAY = "#808080"        # mid gray
TUM_GRAY_DARK = "#333333"   # text / axes
TUM_GRAY_LIGHT = "#CCCCC6"  # frames / grid
 
# Fonts: TUM Neue Helvetica if installed, otherwise the closest Helvetica-like fallbacks.
TUM_FONTS = ["TUM Neue Helvetica", "Helvetica Neue", "Helvetica", "Arial",
             "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]
 
# key: (file name, label, colour). Comment out any entry to drop it.
# Model comparison (keys "random", "baseline", "rl" enable figures 03 / 04):
# RUNS = {
#     "random": ("single_agent_random_mode_choice.csv", "Random mode choice", TUM_GRAY),
#     "baseline": ("single_agent_discrete_choice.csv", "MATSim discrete choice (baseline)", TUM_ORANGE),
#     "rl": ("single_agent_single_policy_training_without_previous_mode.csv",
#            "RL agent (single policy)", TUM_BLUE),
# }
# Seed runs (used by --seed; also work for figures 01 / 02):
RUNS = {
    "42": ("with_prev_mode\\alpah_0.3\\agent_tracking_42.csv", "Seed 42", TUM_GRAY),
    "101": ("with_prev_mode\\alpah_0.3\\agent_tracking_101.csv", "Seed 101", TUM_ORANGE),
    "555": ("with_prev_mode\\alpah_0.3\\agent_tracking_555.csv", "Seed 555", TUM_BLUE),
    "999": ("with_prev_mode\\alpah_0.3\\agent_tracking_999.csv", "Seed 999", TUM_BLUE_DARK),
    "2026": ("with_prev_mode\\alpah_0.3\\agent_tracking_2026.csv", "Seed 2026", TUM_GRAY_DARK),
}
 
MODE_COLORS = {"bike": TUM_BLUE, "car": TUM_ORANGE, "pt": TUM_GRAY, "pedestrian": TUM_BLUE_LIGHT}
MODE_NAMES = {"car": "Car", "pt": "Public transport", "bike": "Bike", "pedestrian": "Walk"}
 
SCORE = "total_day_kainagel_score"
BEST_PLAN = "bike;bike;bike;bike"
DEFAULT_MAX_ITER = 500
CONVERGED_AT = 160      # RL policy convergence (read from the plots)
MODE_OFF_AT = 400       # subtour mode choice switched off for ALL models
LOW_SCORE = 490         # threshold for "poor iteration" share in the summary
 
 
# =============================================================================
# 2. DATA HELPERS
# =============================================================================
 
def load_run(file_path, max_iter, extrapolate=False, seed=42):
    """Load a run CSV. Raises if the file is missing (no synthetic fallback)."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing run file: {path}")
 
    df = pd.read_csv(path).sort_values("iteration").reset_index(drop=True)
    df = df[df["iteration"] <= max_iter].copy()
    df["extrapolated"] = False
 
    current_max = int(df["iteration"].max())
    if extrapolate and current_max < max_iter:
        print(f"[WARNING] {path.name}: iterations {current_max + 1}-{max_iter} are SYNTHETIC "
              f"(bootstrapped). Disclose this in the paper.")
        rng = np.random.default_rng(seed)
        n = max_iter - current_max
        if "random" in path.name.lower():
            scores = rng.choice(df[SCORE].to_numpy(), size=n, replace=True)
            modes = rng.choice(df["experienced_modes"].dropna().to_numpy(), size=n, replace=True)
        else:
            recent = df[SCORE].tail(min(50, len(df)))
            scores = np.clip(rng.normal(recent.mean(), max(recent.std(), 1.0), size=n),
                             df[SCORE].min(), df[SCORE].max())
            modes = rng.choice(df["experienced_modes"].dropna().tail(50).to_numpy(), size=n)
        ext = pd.DataFrame({"iteration": np.arange(current_max + 1, max_iter + 1),
                            SCORE: scores, "experienced_modes": modes, "extrapolated": True})
        df = pd.concat([df, ext], ignore_index=True)
    return df
 
 
def adjust_all_bike(df, lo, hi, seed=0):
    """OPT-IN. Overwrites the score of every all-bike iteration with U(lo, hi)."""
    df = df.copy()
    df["score_raw"] = df[SCORE]
    mask = (df["experienced_modes"] == BEST_PLAN).to_numpy()
    if mask.sum():
        df.loc[mask, SCORE] = np.random.default_rng(seed).uniform(lo, hi, size=int(mask.sum()))
    return df
 
 
def freeze_baseline(df, freeze_at, seed=0):
    """OPT-IN. Replaces baseline scores from `freeze_at` on with resampled all-bike scores."""
    df = df.copy()
    pool = df.loc[df["experienced_modes"] == BEST_PLAN, SCORE].to_numpy()
    if len(pool) == 0:
        pool = df[SCORE].tail(20).to_numpy()
    mask = (df["iteration"] >= freeze_at).to_numpy()
    if mask.sum():
        df.loc[mask, SCORE] = np.random.default_rng(seed).choice(pool, size=int(mask.sum()))
        df.loc[mask, "experienced_modes"] = BEST_PLAN
    return df
 
 
def best_plan_reference(data):
    """Reference score of the best plan: pooled median of all-bike iterations (else max score)."""
    pooled = np.concatenate([
        d.loc[d["experienced_modes"] == BEST_PLAN, SCORE].to_numpy() for d in data.values()
    ])
    if len(pooled):
        return float(np.median(pooled))
    return float(max(d[SCORE].max() for d in data.values()))
 
 
def summarize(data, max_iter):
    """Mean / median / p5 / std / share below LOW_SCORE per run and phase."""
    phases = [(f"0-{CONVERGED_AT}", 0, CONVERGED_AT),
              (f"{CONVERGED_AT}-{MODE_OFF_AT}", CONVERGED_AT, MODE_OFF_AT),
              (f"{MODE_OFF_AT}-{max_iter}", MODE_OFF_AT, max_iter)]
    phases = [ph for ph in phases if ph[1] < ph[2] and ph[1] < max_iter]  # drop empty phases
    rows = []
    for k, d in data.items():
        for name, a, b in phases:
            s = d.loc[(d["iteration"] >= a) & (d["iteration"] <= b), SCORE]
            rows.append({"run": RUNS[k][1], "phase": name, "n": len(s),
                         "mean": s.mean(), "median": s.median(), "p5": s.quantile(0.05),
                         "std": s.std(), f"share_below_{LOW_SCORE}": (s < LOW_SCORE).mean()})
    return pd.DataFrame(rows).round(2)
 
 
# =============================================================================
# 3. STYLE & PLOT HELPERS
# =============================================================================
 
def apply_tum_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": TUM_FONTS,
        "font.size": 13,
        "axes.titlesize": 15, "axes.titleweight": "bold",
        "axes.labelsize": 14, "axes.labelcolor": TUM_GRAY_DARK,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "xtick.color": TUM_GRAY_DARK, "ytick.color": TUM_GRAY_DARK,
        "text.color": TUM_GRAY_DARK,
        "axes.edgecolor": TUM_GRAY_DARK,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": TUM_GRAY_LIGHT,
        "legend.fontsize": 10, "legend.frameon": True, "legend.framealpha": 0.95,
        "legend.edgecolor": TUM_GRAY_LIGHT,
        "figure.titleweight": "bold", "figure.titlesize": 16,
        "pdf.fonttype": 42,
    })
 
 
def title(ax, text, **kw):
    ax.set_title(text, color=TUM_BLUE_DARK, **kw)
 
 
def mark_milestones(ax, max_iter, show_rl=True, label=True):
    """Neutral gray line + shading for mode-choice switch-off; blue line for RL convergence."""
    if max_iter > MODE_OFF_AT:  # cutoff only shown if the plotted range extends past it
        ax.axvspan(MODE_OFF_AT, max_iter, color=TUM_GRAY, alpha=0.08, linewidth=0, zorder=0)
        ax.axvline(MODE_OFF_AT, color=TUM_GRAY_DARK, linestyle="--", linewidth=1.3, zorder=5,
                   label=f"Subtour mode choice off (iter {MODE_OFF_AT})" if label else None)
    if show_rl:
        ax.axvline(CONVERGED_AT, color=TUM_BLUE, linestyle=":", linewidth=2.0, zorder=5,
                   label=f"RL policy convergence (iter {CONVERGED_AT})" if label else None)
 
 
def save(fig, out, stem):
    fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
 
 
# =============================================================================
# 4. FIGURE FUNCTIONS
# =============================================================================
 
def figure_mode_choice_evaluation(data, out, max_iter, w):
    """Fig 04: what a random policy would have given vs what the RL policy chose."""
    if "random" not in data or "rl" not in data:
        return
    rnd, rl = data["random"], data["rl"]
    order = ["bike", "pt", "pedestrian", "car"]
 
    def split(d):
        return d["experienced_modes"].str.split(";", expand=True)
 
    fig, (ax_a, ax_c) = plt.subplots(2, 1, figsize=(11, 11.5),
                                     gridspec_kw={"height_ratios": [1.3, 1], "hspace": 0.38})
 
    # ---- A: mode share per trip, random vs RL (learning) vs RL (converged) ----
    sources = [("Random", rnd),
               (f"RL, iter 0-{CONVERGED_AT}", rl[rl["iteration"] < CONVERGED_AT]),
               (f"RL, iter {CONVERGED_AT}-{max_iter}", rl[rl["iteration"] >= CONVERGED_AT])]
    for t in range(4):
        y0 = t * (len(sources) + 1)
        ax_a.text(0, y0, f"Trip {t + 1}", fontweight="bold", color=TUM_BLUE_DARK, va="center", fontsize=11)
        for j, (name, df) in enumerate(sources):
            sh = split(df)[t].value_counts(normalize=True)
            left, y = 0.0, y0 + 1 + j
            for m in order:
                v = 100 * float(sh.get(m, 0.0))
                if v <= 0:
                    continue
                ax_a.barh(y, v, left=left, color=MODE_COLORS[m], height=0.8, linewidth=0)
                if v >= 9:
                    ax_a.text(left + v / 2, y, f"{v:.0f}", ha="center", va="center", fontsize=9,
                              color="white" if m != "pedestrian" else TUM_GRAY_DARK)
                left += v
    ax_a.set_yticks([t * 4 + 1 + j for t in range(4) for j in range(3)])
    ax_a.set_yticklabels([n for _ in range(4) for n, _ in sources], fontsize=10)
    ax_a.invert_yaxis()
    ax_a.set_xlim(0, 100)
    ax_a.set_xlabel("Mode share (%)")
    ax_a.tick_params(axis="y", length=0)
    ax_a.legend(handles=[Patch(color=MODE_COLORS[m], label=MODE_NAMES[m]) for m in order],
                loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=10,
                labelcolor=TUM_GRAY_DARK)
    title(ax_a, "A   Mode choice per trip", loc="left", fontsize=13, pad=32)
 
    # ---- B: RL score vs what a random policy would have scored ----
    r_mean, r_lo, r_hi = rnd[SCORE].mean(), rnd[SCORE].quantile(0.05), rnd[SCORE].quantile(0.95)
    d = rl.sort_values("iteration")
    ax_c.axhspan(r_lo, r_hi, color=TUM_GRAY, alpha=0.18, linewidth=0, zorder=1,
                 label=f"Random policy, 5th-95th pct ({r_lo:.0f}-{r_hi:.0f})")
    ax_c.axhline(r_mean, color=TUM_GRAY, linestyle="--", linewidth=1.8, zorder=2,
                 label=f"Random policy, mean ({r_mean:.0f})")
    ax_c.plot(d["iteration"], d[SCORE], color=TUM_BLUE, linewidth=0.9, alpha=0.5, zorder=3,
              label="RL agent (raw)")
    ax_c.plot(d["iteration"], d[SCORE].rolling(w, center=True, min_periods=1).median(),
              color=TUM_BLUE, linewidth=2.6, zorder=4, label=f"RL agent (rolling median, W={w})")
    mark_milestones(ax_c, max_iter, True, label=True)
    conv = d.loc[d["iteration"] >= CONVERGED_AT, SCORE].median()
    xa = min(max_iter - 18, 455)
    ax_c.annotate("", xy=(xa, conv), xytext=(xa, r_mean),
                  arrowprops=dict(arrowstyle="<->", color=TUM_GRAY_DARK, lw=1.4), zorder=6)
    ax_c.text(xa - 6, (conv + r_mean) / 2, f"+{conv - r_mean:.0f} pts", ha="right", va="center",
              fontsize=11, fontweight="bold", color=TUM_BLUE_DARK, zorder=6)
    ax_c.set_xlim(0, max_iter)
    ax_c.set_ylim(300, 515)
    ax_c.set_xlabel("MATSim iteration")
    ax_c.set_ylabel("Daily plan score")
    ax_c.grid(alpha=0.5, linewidth=0.6)
    ax_c.legend(loc="lower right", fontsize=8.5, labelcolor=TUM_GRAY_DARK, ncol=2)
    title(ax_c, "B   RL score vs random expectation", loc="left", fontsize=13)
 
    fig.suptitle("Mode choice evaluation: random policy vs learned RL policy",
                 color=TUM_BLUE_DARK, y=0.97)
    save(fig, out, "04_mode_choice_evaluation")
 
 
def generate_robustness_analysis(data, out, max_iter, w):
    """Fig 05: seed robustness. A = all seeds overlaid (rolling median),
    B = mean over seeds with std and 5th-95th percentile envelope."""
    # iterations x seeds matrix; seeds with shorter runs yield NaN (handled below)
    mat = pd.DataFrame({k: d.set_index("iteration")[SCORE] for k, d in data.items()})
    mat = mat.loc[mat.index <= max_iter].sort_index()
    its = mat.index.to_numpy()
    vals = mat.to_numpy()
 
    mean = np.nanmean(vals, axis=1)
    std = np.nanstd(vals, axis=1)
    p5 = np.nanquantile(vals, 0.05, axis=1)
    p95 = np.nanquantile(vals, 0.95, axis=1)
    ref = best_plan_reference(data)
 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10), sharex=True, sharey=True)
 
    # ---- A: all seeds overlaid ----
    for k in mat.columns:
        ax1.plot(its, mat[k].rolling(w, center=True, min_periods=1).mean(),
                 color=RUNS[k][2], linewidth=1.2, alpha=0.9, label=RUNS[k][1])
    ax1.axhline(ref, color=TUM_BLUE_DARK, linestyle="--", linewidth=1.0, alpha=0.5, zorder=1)
    mark_milestones(ax1, max_iter, show_rl=False, label=True)
    title(ax1, "A   Score paths per random seed (smoothing)", loc="left", fontsize=13)
    ax1.set_ylabel("Daily plan score")
    ax1.set_ylim(425, 510)
    ax1.grid(alpha=0.5, linewidth=0.6)
    ax1.legend(title="Random seed", loc="lower right", ncol=3, labelcolor=TUM_GRAY_DARK)
 
    # ---- B: aggregate over seeds ----
    ax2.fill_between(its, p5, p95, color=TUM_BLUE, alpha=0.15, linewidth=0,
                     label="5th-95th percentile over seeds")
    ax2.fill_between(its, mean - std, mean + std, color=TUM_BLUE, alpha=0.25, linewidth=0,
                     label="Mean ± 1 std")
    ax2.plot(its, mean, color=TUM_BLUE_DARK, linewidth=2.5, label="Mean over seeds (raw)")
    ax2.axhline(ref, color=TUM_BLUE_DARK, linestyle="--", linewidth=1.0, alpha=0.5, zorder=1)
    mark_milestones(ax2, max_iter, show_rl=False, label=False)
    title(ax2, "B   Aggregate over seeds: mean and variability envelope", loc="left", fontsize=13)
    ax2.set_ylabel("Daily plan score")
    ax2.set_xlabel("MATSim iteration")
    ax2.set_xlim(0, max_iter)
    ax1.set_ylim(300, 510)
    ax2.grid(alpha=0.5, linewidth=0.6)
    ax2.legend(loc="lower right", labelcolor=TUM_GRAY_DARK)
 
    fig.suptitle(f"Seed robustness analysis ({len(mat.columns)} seeds)", color=TUM_BLUE_DARK, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, out, "05_seed_robustness")
 
 
# =============================================================================
# 5. MAIN
# =============================================================================
 
def main():
    global MODE_OFF_AT  # must come before any use of MODE_OFF_AT in this function
    p = argparse.ArgumentParser(description="Plot MATSim vs RL training comparison.")
    p.add_argument("--dir", default=DATA_DIR, help="Directory containing the run CSVs")
    p.add_argument("--out-dir", default="./out")
    p.add_argument("--max-iter", "--max-iteration", dest="max_iter", type=int,
                   default=DEFAULT_MAX_ITER, help="Maximum iteration to plot")
    p.add_argument("--cutoff-iter", "--cutoff-iteration", dest="cutoff_iter", type=int,
                   default=MODE_OFF_AT,
                   help="Iteration at which subtour mode choice is switched off; the marker is "
                        "only drawn if --max-iter is greater than this")
    p.add_argument("--window", type=int, default=15, help="Rolling window (median/share plots)")
    p.add_argument("--seed", action="store_true",
                   help="Only generate the seed robustness analysis (skips all other figures)")
    p.add_argument("--extrapolate", action="store_true", help="OPT-IN synthetic extension to max-iter")
    p.add_argument("--adjust-all-bike", action="store_true", help="OPT-IN overwrite of all-bike scores")
    p.add_argument("--allbike-lo", type=float, default=482.0)
    p.add_argument("--allbike-hi", type=float, default=485.0)
    p.add_argument("--freeze-at", type=int, default=0, help="OPT-IN baseline freeze iteration (0 = off)")
    p.add_argument("--reward-ymin", type=float, default=-1000.0, help="Lower y-limit of the reward plot")
    p.add_argument("--no-show", action="store_true", help="Do not open plot windows")
    args = p.parse_args()
 
    MODE_OFF_AT = args.cutoff_iter
    w, max_iter = args.window, args.max_iter
    apply_tum_style()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
 
    # ---- load -------------------------------------------------------------
    data = {k: load_run(Path(args.dir) / f, max_iter, args.extrapolate) for k, (f, _, _) in RUNS.items()}
 
    if args.adjust_all_bike:
        print(f"[WARNING] all-bike scores overwritten with U({args.allbike_lo}, {args.allbike_hi}) "
              f"for baseline/random. Raw scores are kept in 'score_raw'. Disclose and justify this.")
        for k, seed in (("baseline", 11), ("random", 12)):
            if k in data:
                data[k] = adjust_all_bike(data[k], args.allbike_lo, args.allbike_hi, seed)
    if args.freeze_at > 0 and "baseline" in data:
        print(f"[WARNING] baseline scores from iteration {args.freeze_at} replaced by resampled values.")
        data["baseline"] = freeze_baseline(data["baseline"], args.freeze_at)
 
    # ---- seed mode: ONLY the seed robustness analysis ----------------------
    if args.seed:
        generate_robustness_analysis(data, out, max_iter, w)
        if not args.no_show:
            plt.show()
        return
 
    show_rl = "rl" in data
 
    table = summarize(data, max_iter)
    table.to_csv(out / "summary_by_phase.csv", index=False)
    print(table.to_string(index=False))
 
    # =========================================================================
    # FIGURE 1: one unsmoothed subplot per run
    # =========================================================================
    fig, axes = plt.subplots(len(data), 1, figsize=(11, 3.3 * len(data) + 0.8), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (k, d) in zip(axes, data.items()):
        _, name, color = RUNS[k]
        d = d.sort_values("iteration")
        ax.plot(d["iteration"], d[SCORE], color=color, linewidth=1.3, zorder=4)
        mark_milestones(ax, max_iter, show_rl=(k == "rl"), label=False)
        title(ax, name, loc="left", fontsize=13)
        ax.set_xlim(0, max_iter)
        ax.set_ylim(300, 515)
        ax.grid(alpha=0.5, linewidth=0.6)
    axes[-1].set_xlabel("MATSim iteration")
    fig.supylabel("Daily plan score", color=TUM_GRAY_DARK, fontsize=14)
    handles = []
    if max_iter > MODE_OFF_AT:
        handles.append(Line2D([0], [0], color=TUM_GRAY_DARK, linestyle="--", linewidth=1.3,
                              label=f"Subtour mode choice off (iter {MODE_OFF_AT})"))
    if show_rl:
        handles.append(Line2D([0], [0], color=TUM_BLUE, linestyle=":", linewidth=2.0,
                              label=f"RL policy convergence (iter {CONVERGED_AT})"))
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Agent daily score per iteration (unsmoothed)", color=TUM_BLUE_DARK, y=0.995)
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.98))
    save(fig, out, "01_agent_score_per_model")
 
    # =========================================================================
    # FIGURE 2: cumulative regret
    # =========================================================================
    ref = best_plan_reference(data)
    fig, ax = plt.subplots(figsize=(11, 6))
    for k, (_, name, color) in RUNS.items():
        d = data[k].sort_values("iteration")
        ax.plot(d["iteration"], (ref - d[SCORE]).clip(lower=0).cumsum(), color=color,
                linewidth=2.6, label=name)
    mark_milestones(ax, max_iter, show_rl)
    ax.set_xlim(0, max_iter)
    ax.set_xlabel("MATSim iteration")
    ax.set_ylabel("Cumulative regret (score points)")
    title(ax, f"Cumulative regret vs best plan (reference score {ref:.1f})")
    ax.grid(alpha=0.5, linewidth=0.6)
    ax.legend(loc="upper left", labelcolor=TUM_GRAY_DARK)
    save(fig, out, "02_cumulative_regret")
 
    # =========================================================================
    # FIGURE 3: RL policy training - reward curve and accumulated dQ
    # =========================================================================
    need = {"total_day_reward", "accumulated_delta_q", "step_delta_qs"}
    if show_rl and need <= set(data["rl"].columns):
        d = data["rl"].sort_values("iteration").copy()
        its = d["iteration"].to_numpy()
        reward = d["total_day_reward"]
        penalty = (d[SCORE] - d["total_day_reward"]).clip(lower=0)
        # dQ is logged as exactly 0 with empty step_delta_qs every 40th iteration and
        # after training stops. These are logging gaps, not learning -> treat as missing.
        dq = d["accumulated_delta_q"].where(d["step_delta_qs"].notna())
        events = d.loc[(penalty > 1) & (d["iteration"] >= CONVERGED_AT), "iteration"].to_numpy()
        n_clip = int((reward < args.reward_ymin).sum())
 
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                     gridspec_kw={"height_ratios": [1.1, 1]})
        # (a) reward
        a1.plot(its, reward, color=TUM_BLUE, linewidth=0.9, alpha=0.45, label="Reward (raw)")
        a1.plot(its, reward.rolling(w, center=True, min_periods=1).median(),
                color=TUM_BLUE, linewidth=2.8, label=f"Rolling median (W={w})")
        a1.set_ylim(args.reward_ymin, 520)
        a1.set_ylabel("Daily reward")
        title(a1, "RL policy training", loc="left")
        if n_clip:
            a1.text(0.995, 0.03, f"{n_clip} values below {args.reward_ymin:.0f} clipped",
                    transform=a1.transAxes, ha="right", va="bottom", fontsize=9, color=TUM_GRAY)
        # (b) accumulated dQ
        a2.plot(its, dq, color=TUM_BLUE, linewidth=0.9, alpha=0.45, label="Accumulated ΔQ (raw)")
        a2.plot(its, dq.rolling(w, center=True, min_periods=1).median(),
                color=TUM_BLUE, linewidth=2.8, label=f"Rolling median (W={w})")
        a2.set_yscale("symlog", linthresh=0.1)
        a2.set_ylim(0, 1000)
        a2.set_ylabel("Accumulated ΔQ per iteration")
        a2.set_xlabel("MATSim iteration")
        for a in (a1, a2):
            mark_milestones(a, max_iter, True, label=(a is a1))
            for e in events:
                a.axvline(e, color=TUM_ORANGE, linewidth=1.0, alpha=0.55, zorder=1)
            a.set_xlim(0, max_iter)
            a.grid(alpha=0.5, linewidth=0.6)
        if len(events):
            a1.plot([], [], color=TUM_ORANGE, linewidth=1.5, alpha=0.8,
                    label="Off-policy action after convergence (penalty > 0)")
        a1.legend(loc="center right", labelcolor=TUM_GRAY_DARK, fontsize=9)
        a2.legend(loc="upper right", labelcolor=TUM_GRAY_DARK, fontsize=9)
        save(fig, out, "03_rl_policy_training")
 
    # =========================================================================
    # FIGURE 4: mode choice evaluation (random policy vs RL policy)
    # =========================================================================
    figure_mode_choice_evaluation(data, out, max_iter, w)
 
    if not args.no_show:
        plt.show()
 
 
if __name__ == "__main__":
    main()