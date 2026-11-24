"""
Which state features explain the step score, and which only add state sparsity?
Feature Ablation and Variance Attribution pipeline for RL mode choice in MATSim.
 
Two commands
------------
  variance  Phase 2: multi-way ANOVA / variance partitioning of the step score over the discrete state features,
            plus a sparsity test (cross-validated R2 with every feature left out) and a prune / keep verdict.
              python state_ablation_analysis.py variance state_log.csv --out-dir out \
                     [--features origin_taz,dest_taz,time_bin,dist_bin,asset,cong_bin,prev_mode] \
                     [--groups "location=origin_taz+dest_taz"] [--bit-widths "origin_taz=5,dest_taz=5,..."] \
                     [--step-labels "home>work,work>lunch,lunch>work,work>home"] [--agent ID]
 
  ablation  Phase 3: compare the RL training runs (full state vs leave-one-feature-group-out) on peak score,
            convergence speed, stability and residual pure noise, and label every feature keep / prune.
              python state_ablation_analysis.py ablation ablation_runs.csv --out-dir out --baseline full
 
Input of `variance` (long format, ONE ROW PER agent x iteration x step; see example_state_log.csv)
  iteration, agent_id, step, mode, score, <one column per decoded state feature>
  The state columns are the state the agent SAW when it chose `mode` at that step (decoded from the bits).
Input of `ablation`: the same columns plus `experiment` (e.g. full, no_prev_mode) and optionally `seed`.
  State columns are optional there; if present they are used to make the noise estimate strict.
 
Definitions
  unique effect       ANOVA sum of squares lost when the feature is dropped from the full additive model
                      (mode, and step when pooled, are always in the model).  eta2_unique = SS_unique / SS_total.
  partial omega2      bias-corrected partial effect size.  Raw eta2 is inflated for features with many levels
                      (zone IDs!); omega2 subtracts the amount that pure chance would produce.
  x mode interaction  does the feature change WHICH MODE IS BEST?  A feature can shift all scores (main effect)
                      and still be useless for the policy; only the feature x mode interaction helps the agent choose.
  pure noise          spread between rows with identical agent, step, mode history and identical state:
                      nothing else differs, so this is the randomisation of the simulation.  Sets the ceiling:
                      no state definition can explain more than (1 - share_noise) of the variance.
  cv_gain             CV R2 of a table lookup (state cell x mode) with the feature minus without it.  This is what a
                      tabular agent actually experiences: negative = the feature fragments the data more than it informs.
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
 
try:
    from scipy import stats as sps
except ImportError:                                   # p-values become NaN, everything else works
    sps = None
 
RESERVED = {"iteration", "agent_id", "step", "mode", "score", "experiment", "seed", "prefix", "state_bits"}
MODE_COLORS = {"car": "#4C72B0", "pt": "#55A868", "bike": "#C44E52", "pedestrian": "#8172B2"}
NOISE_COLOR, SHARED_COLOR, UNEXPL_COLOR, ACTION_COLOR = "#C44E52", "#B0B0B0", "#E8E8E8", "#333333"
_TAB = plt.get_cmap("tab10")
FEATURE_PALETTE = [_TAB(i) for i in (0, 1, 2, 4, 5, 6, 8, 9)]
VERDICT_STYLE = {"keep": ("#2E8B57", "o"), "borderline": ("#E0A800", "s"), "prune": ("#C0392B", "X")}
 
 
# ------------------------------------------------------------------------------------ loading
def load_log(path, agent=None, experiment=None, features=None):
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = {"iteration", "agent_id", "step", "mode", "score"} - set(header)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")
    if features is None:
        features = [c for c in header if c not in RESERVED]
    absent = [f for f in features if f not in header]
    if absent:
        raise SystemExit(f"Feature columns not in the CSV: {absent}")
    df = pd.read_csv(path, dtype={c: str for c in features + ["agent_id", "mode"]})
    if experiment is not None:
        df = df[df["experiment"].astype(str) == str(experiment)]
    if agent is not None:
        df = df[df["agent_id"].astype(str) == str(agent)]
    if df.empty:
        raise SystemExit("No rows left after filtering (check --agent / --experiment).")
    df = df.dropna(subset=["score", "mode"]).copy()
    for col in features:
        df[col] = df[col].astype(str)
 
    keys = [c for c in ("experiment", "seed") if c in df.columns] + ["agent_id", "iteration"]
    df = df.sort_values(keys + ["step"], kind="stable").reset_index(drop=True)
    if not (df.groupby(keys).cumcount().to_numpy() == df["step"].to_numpy()).all():
        raise SystemExit("Steps must run 0,1,2,... without gaps inside every (agent, iteration).")
 
    prefix, current, last = [], "", None                   # mode history up to and including this step
    for key, mode in zip(zip(*[df[k] for k in keys]), df["mode"]):
        current = mode if key != last else current + ">" + mode
        last = key
        prefix.append(current)
    df["prefix"] = prefix
    return df, features
 
 
# ------------------------------------------------------------------------------------ building blocks
def dummies(series):
    codes, uniques = pd.factorize(series, sort=True)
    out = np.zeros((len(series), max(len(uniques) - 1, 0)))
    mask = codes > 0
    if out.shape[1]:
        out[np.nonzero(mask)[0], codes[mask] - 1] = 1.0
    return out
 
 
def fit_rss(mats, y):
    x = np.column_stack([np.ones(len(y))] + mats) if mats else np.ones((len(y), 1))
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    return float(resid @ resid), int(rank)
 
 
def cell_codes(frame, cols):
    if not cols:
        return np.zeros(len(frame), dtype=int)
    return frame.groupby(cols, sort=False).ngroup().to_numpy()
 
 
def effect_stats(ss, df, mse, n, dof_res):
    if df <= 0 or not np.isfinite(mse) or mse <= 0 or dof_res <= 0:
        return {"F": np.nan, "p": np.nan, "omega2_partial": np.nan, "partial_eta2": np.nan}
    f_value = (ss / df) / mse
    omega = max(0.0, df * (f_value - 1) / (df * (f_value - 1) + n))
    p_value = float(sps.f.sf(f_value, df, dof_res)) if sps is not None else np.nan
    return {"F": f_value, "p": p_value, "omega2_partial": omega, "partial_eta2": ss / (ss + mse * dof_res)}
 
 
def estimate_noise(frame, feature_cols, mode="auto", min_df=30):
    """Pooled within-group std of the score over rows that are replicates of each other. -> (std, df, key used)."""
    def pooled(cols):
        stats = frame.groupby(cols, sort=False)["score"].agg(["size", "var"])
        use = stats["size"] >= 2
        degrees = int((stats.loc[use, "size"] - 1).sum())
        if degrees == 0:
            return float("nan"), 0
        ss = float(((stats.loc[use, "size"] - 1) * stats.loc[use, "var"]).sum())
        return math.sqrt(ss / degrees), degrees
 
    strict = ["agent_id", "step", "prefix"] + feature_cols     # same agent, same mode history, same state
    relaxed = ["agent_id", "step", "mode"] + feature_cols      # earlier modes beyond the state count as noise
    if mode in ("prefix", "auto"):
        std, degrees = pooled(strict)
        if mode == "prefix" or degrees >= min_df:
            return std, degrees, "prefix"
    std, degrees = pooled(relaxed)
    return std, degrees, "state"
 
 
def cv_r2(codes_cell, codes_bo, y, k=5, shrink=2.0, repeats=2, seed=0):
    """K-fold R2 of a smoothed table lookup: cell mean shrunk towards the action(-and-step) mean."""
    n = len(y)
    sst = float(((y - y.mean()) ** 2).sum())
    if n < 2 * k or sst == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    n_cell, n_bo = codes_cell.max() + 1, codes_bo.max() + 1
    scores = []
    for _ in range(repeats):
        fold = rng.permutation(n) % k
        sse = 0.0
        for f in range(k):
            test, train = fold == f, fold != f
            s = np.bincount(codes_cell[train], weights=y[train], minlength=n_cell)
            c = np.bincount(codes_cell[train], minlength=n_cell)
            bs = np.bincount(codes_bo[train], weights=y[train], minlength=n_bo)
            bc = np.bincount(codes_bo[train], minlength=n_bo)
            backoff = np.where(bc > 0, bs / np.maximum(bc, 1), y[train].mean())
            pred = (s[codes_cell[test]] + shrink * backoff[codes_bo[test]]) / (c[codes_cell[test]] + shrink)
            sse += float(((y[test] - pred) ** 2).sum())
        scores.append(1 - sse / sst)
    return float(np.mean(scores))
 
 
def sparsity_stats(frame, cols, ctrl, min_obs):
    state_cols = cols + [c for c in ctrl if c != "mode"]
    state = cell_codes(frame, state_cols)
    action = cell_codes(frame, state_cols + ["mode"])
    state_sizes, action_sizes = np.bincount(state), np.bincount(action)
    return {"n_states_seen": len(state_sizes), "n_state_action_cells": len(action_sizes),
            "median_rows_per_state": float(np.median(state_sizes)),
            "median_rows_per_state_action": float(np.median(action_sizes)),
            "pct_cells_singleton": 100.0 * float((action_sizes == 1).mean()),
            "pct_rows_in_thin_cells": 100.0 * float(action_sizes[action_sizes < min_obs].sum() / len(frame))}
 
 
# ------------------------------------------------------------------------------------ one scope
def analyze_scope(frame, scope, groups, ctrl, bits, args):
    """ANOVA + variance split + sparsity for one set of rows (one step, or all steps pooled)."""
    y = frame["score"].to_numpy(float)
    n = len(y)
    sst = float(((y - y.mean()) ** 2).sum())
    total_var = sst / (n - 1) if n > 1 else float("nan")
    blocks = {c: dummies(frame[c]) for c in ctrl}
    for name, cols in groups.items():
        blocks[name] = np.column_stack([dummies(frame[c]) for c in cols])
    names = list(groups)
    all_names = ctrl + names
 
    def fit(model, extra=()):
        mats = [blocks[m] for m in model if blocks[m].shape[1] > 0] + list(extra)
        return fit_rss(mats, y)
 
    rss_full, rank_full = fit(all_names)
    dof_res = n - rank_full
    mse = rss_full / dof_res if dof_res > 0 else float("nan")
    rss_ctrl, _ = fit(ctrl)
    saturated = dof_res <= 0
 
    # ---- unique effect of every control and feature group (drop-one from the full additive model)
    unique = {}
    for name in all_names:
        if blocks[name].shape[1] == 0:
            unique[name] = (0.0, 0)
            continue
        rss_drop, rank_drop = fit([m for m in all_names if m != name])
        unique[name] = (max(rss_drop - rss_full, 0.0), rank_full - rank_drop)
 
    rows = []
    for name in names:
        cols = groups[name]
        levels = int(cell_codes(frame, cols).max() + 1)
        ss_u, df_u = unique[name]
        stat = effect_stats(ss_u, df_u, mse, n, dof_res)
        rss_alone, _ = fit(ctrl + [name]) if blocks[name].shape[1] else (rss_ctrl, 0)
        eta_alone = max(rss_ctrl - rss_alone, 0.0) / sst if sst > 0 else float("nan")
 
        inter = {"F": np.nan, "p": np.nan, "omega2_partial": np.nan}
        df_i = 0
        if "mode" in ctrl and blocks[name].shape[1] and blocks["mode"].shape[1]:
            width = blocks[name].shape[1] * blocks["mode"].shape[1]
            if width <= args.max_inter_cols:
                prod = (blocks[name][:, :, None] * blocks["mode"][:, None, :]).reshape(n, -1)
                rss_i, rank_i = fit(all_names, extra=[prod])
                df_i = rank_i - rank_full
                dof_i = n - rank_i
                mse_i = rss_i / dof_i if dof_i > 0 else float("nan")
                inter = effect_stats(max(rss_full - rss_i, 0.0), df_i, mse_i, n, dof_i)
        rows.append({"scope": scope, "feature": name, "bits": bits[name], "levels": levels, "df_unique": df_u,
                     "eta2_alone": eta_alone, "eta2_unique": ss_u / sst if sst > 0 else float("nan"),
                     "partial_eta2": stat["partial_eta2"], "partial_omega2": stat["omega2_partial"],
                     "F": stat["F"], "p_value": stat["p"],
                     "df_x_mode": df_i, "x_mode_partial_omega2": inter["omega2_partial"], "x_mode_p_value": inter["p"]})
 
    # ---- variance split
    feature_cols = [c for cols in groups.values() for c in cols]
    noise, noise_df, noise_key = estimate_noise(frame, feature_cols, args.noise_key, args.min_noise_df)
    share_noise = min(1.0, noise ** 2 / total_var) if np.isfinite(noise) and total_var > 0 else float("nan")
    adj_r2 = 1 - mse / total_var if not saturated and total_var > 0 else float("nan")
    shares = {}
    for name in all_names:
        ss_u, df_u = unique[name]
        shares[name] = max(0.0, (ss_u - df_u * mse) / sst) if not saturated and sst > 0 else 0.0
    shared = max(0.0, (adj_r2 if np.isfinite(adj_r2) else 0.0) - sum(shares.values()))
    noise_part = share_noise if np.isfinite(share_noise) else 0.0
    unexplained = max(0.0, 1.0 - sum(shares.values()) - shared - noise_part)
    scale = max(1.0, sum(shares.values()) + shared + noise_part + unexplained)
    split = {"scope": scope, "n_rows": n, "total_std": math.sqrt(total_var) if total_var > 0 else 0.0,
             "pure_noise_std": noise, "noise_df": noise_df, "noise_key": noise_key, "share_noise": share_noise,
             "noise_ceiling_r2": 1 - share_noise if np.isfinite(share_noise) else float("nan"),
             "r2_full": 1 - rss_full / sst if sst > 0 else float("nan"), "adj_r2_full": adj_r2,
             "saturated_model": saturated}
    for name in all_names:
        split[f"share_{name}"] = shares[name] / scale
    split["share_shared"] = shared / scale
    split["share_unexplained_structure"] = unexplained / scale
    split["share_noise_plot"] = noise_part / scale
 
    # ---- sparsity: leave-one-group-out, out-of-sample
    subsets = [("full", names)] + [(f"-{g}", [m for m in names if m != g]) for g in names] + [("action only", [])]
    bo_codes = cell_codes(frame, ctrl)
    sparse_rows = []
    for label, members in subsets:
        cols = [c for m in members for c in groups[m]]
        stats = sparsity_stats(frame, cols, ctrl, args.min_obs)
        cv = cv_r2(cell_codes(frame, cols + ctrl), bo_codes, y, args.cv_folds, args.cv_shrink, args.cv_repeats)
        sparse_rows.append({"scope": scope, "state_definition": label, "bits": sum(bits[m] for m in members),
                            "cv_r2": cv, **stats})
    return rows, split, sparse_rows
 
 
# ------------------------------------------------------------------------------------ verdicts
def prune_verdicts(anova, sparse, scope, args):
    cv = sparse[sparse["scope"] == scope].set_index("state_definition")["cv_r2"]
    out = []
    for row in anova[anova["scope"] == scope].itertuples():
        gain = cv["full"] - cv[f"-{row.feature}"]
        omega_main = 0.0 if pd.isna(row.partial_omega2) else row.partial_omega2
        omega_inter = 0.0 if pd.isna(row.x_mode_partial_omega2) else row.x_mode_partial_omega2
        signal = max(omega_main, omega_inter)
        if row.levels <= 1:
            label, reason = "prune", "constant (never varies)"
        elif row.df_unique == 0:
            label, reason = "prune", "fully determined by the other features"
        elif gain < -args.cv_tol:
            label, reason = "prune", "harmful sparsity: out-of-sample fit improves without it"
        elif signal < args.min_effect and gain < args.cv_tol:
            label, reason = "prune", "no explanatory signal and no out-of-sample benefit"
        elif signal >= args.min_effect and gain >= args.cv_tol:
            label, reason = "keep", "explains score variance and improves out-of-sample fit"
        else:
            label, reason = "borderline", "mixed evidence: confirm with an ablation run"
        out.append({"feature": row.feature, "bits": row.bits, "levels": row.levels,
                    "omega2_main": omega_main, "omega2_x_mode": omega_inter, "cv_r2_gain": gain,
                    "verdict": label, "reason": reason})
    return pd.DataFrame(out)
 
 
# ------------------------------------------------------------------------------------ figure: variance
def make_variance_figure(anova, split, sparse, verdicts, labels, group_names, out_path, title, args):
    scopes = [s for s in split["scope"] if s != "ALL"]
    n_steps = len(scopes)
    order = list(verdicts.sort_values(["omega2_x_mode", "omega2_main"], ascending=True)
                 .assign(k=lambda d: d[["omega2_main", "omega2_x_mode"]].max(axis=1)).sort_values("k")["feature"])
    ypos = {f: i for i, f in enumerate(order)}
    palette = {g: FEATURE_PALETTE[i % len(FEATURE_PALETTE)] for i, g in enumerate(group_names)}
 
    fig = plt.figure(figsize=(max(4.2 * n_steps, 14), 12.5))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.42)
    top = outer[0].subgridspec(1, n_steps, wspace=0.12)
    xmax = np.nanmax(np.r_[anova["partial_omega2"].to_numpy(float), anova["x_mode_partial_omega2"].to_numpy(float),
                          anova["eta2_alone"].to_numpy(float)]) * 1.12 * 100
    xmax = max(xmax, 5.0)
 
    for i, scope in enumerate(scopes):
        axis = fig.add_subplot(top[0, i])
        frame = anova[anova["scope"] == scope]
        for row in frame.itertuples():
            y = ypos[row.feature]
            axis.barh(y + 0.19, 100 * (0 if pd.isna(row.partial_omega2) else row.partial_omega2), height=0.36,
                      color="#4C72B0", label="unique effect (partial omega2)")
            axis.barh(y - 0.19, 100 * (0 if pd.isna(row.x_mode_partial_omega2) else row.x_mode_partial_omega2),
                      height=0.36, color="#DD8452", label="feature x mode interaction (decision value)")
            axis.scatter(100 * row.eta2_alone, y, marker="D", s=26, facecolor="white", edgecolor="black", zorder=5,
                         linewidths=1, label="raw eta2 alone (inflated by many levels)")
        axis.set_yticks(range(len(order)))
        axis.set_yticklabels(order if i == 0 else [], fontsize=9)
        axis.set_xlim(0, xmax)
        axis.set_title(labels[i], fontsize=11, fontweight="bold")
        axis.grid(axis="x", alpha=0.25)
        axis.set_xlabel("% of step-score variance", fontsize=8)
        sp = split[split["scope"] == scope].iloc[0]
        cv = sparse[(sparse["scope"] == scope) & (sparse["state_definition"] == "full")]["cv_r2"].iloc[0]
        noise = "n/a" if pd.isna(sp["share_noise"]) else f"{100 * sp['share_noise']:.0f}%"
        axis.text(0.98, 0.02, f"pure noise {noise}\nadj R2 {100 * sp['adj_r2_full']:.0f}% | CV R2 {100 * cv:.0f}%",
                  transform=axis.transAxes, ha="right", va="bottom", fontsize=8,
                  bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
        if i == 0:
            handles, names = axis.get_legend_handles_labels()
            seen = dict(zip(names, handles))
            axis.legend(seen.values(), seen.keys(), fontsize=7, loc="upper right", frameon=True)
 
    bottom = outer[1].subgridspec(1, 3, wspace=0.55, width_ratios=[1.15, 1.0, 1.05])
 
    # ---- bottom left: where does the variance of each step come from?
    axis = fig.add_subplot(bottom[0, 0])
    sp = split.copy()
    sp["label"] = [("All steps pooled" if s == "ALL" else labels[scopes.index(s)]) for s in sp["scope"]]
    y_idx = np.arange(len(sp))[::-1]
    parts = [(f"share_{g}", g, palette[g]) for g in group_names]
    parts += [("share_mode", "action (mode)", ACTION_COLOR)]
    if "share_step" in sp.columns:
        parts += [("share_step", "step", "#7F7F7F")]
    parts += [("share_shared", "shared / overlapping", SHARED_COLOR),
              ("share_unexplained_structure", "unexplained structure (interactions, missing features)", UNEXPL_COLOR),
              ("share_noise_plot", "pure noise (randomisation)", NOISE_COLOR)]
    left = np.zeros(len(sp))
    for column, name, color in parts:
        if column not in sp.columns:
            continue
        widths = np.nan_to_num(sp[column].to_numpy(float)) * 100
        axis.barh(y_idx, widths, left=left, color=color, label=name, height=0.62,
                  hatch="//" if color == UNEXPL_COLOR else None, edgecolor="white" if color != UNEXPL_COLOR else "#BBBBBB",
                  linewidth=0.4)
        for yy, x0, w in zip(y_idx, left, widths):
            if w >= 6:
                axis.text(x0 + w / 2, yy, f"{w:.0f}%", ha="center", va="center", fontsize=8, fontweight="bold",
                          color="white" if color not in (UNEXPL_COLOR, SHARED_COLOR) else "black")
        left += widths
    axis.set_yticks(y_idx)
    axis.set_yticklabels(sp["label"], fontsize=9)
    axis.set_xlim(0, 100)
    axis.set_xlabel("share of score variance (%)")
    axis.set_title("What makes the score vary?\n(feature shares = unique, bias-corrected)", fontsize=11, fontweight="bold")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=7, frameon=False)
 
    # ---- bottom middle: prune map
    axis = fig.add_subplot(bottom[0, 1])
    gain_pp = 100 * verdicts["cv_r2_gain"].to_numpy(float)
    signal = 100 * verdicts[["omega2_main", "omega2_x_mode"]].max(axis=1).to_numpy(float)
    axis.axvspan(min(gain_pp.min() - 1, -1) - 1, 100 * args.cv_tol, ymin=0, ymax=1, color="#C0392B", alpha=0.06)
    for row, gx, sy in zip(verdicts.itertuples(), gain_pp, signal):
        color, marker = VERDICT_STYLE[row.verdict]
        axis.scatter(gx, sy, s=80, color=color, marker=marker, edgecolors="white", linewidths=0.6, zorder=4)
        axis.annotate(row.feature, (gx, sy), textcoords="offset points", xytext=(6, 4), fontsize=8)
    axis.axhline(100 * args.min_effect, color="#C0392B", linestyle="--", linewidth=1)
    axis.axvline(100 * args.cv_tol, color="#C0392B", linestyle="--", linewidth=1)
    axis.axvline(0, color="grey", linewidth=0.8)
    axis.set_xlabel("out-of-sample gain from the feature (CV R2 points, pooled)")
    axis.set_ylabel("explanatory signal: max(unique, x mode) omega2 (%)")
    axis.set_title("Keep or prune?\n(red region / left of line = no benefit)", fontsize=11, fontweight="bold")
    axis.grid(alpha=0.25)
    handles = [plt.Line2D([], [], marker=m, color=c, linestyle="", label=k) for k, (c, m) in VERDICT_STYLE.items()]
    axis.legend(handles=handles, fontsize=8, loc="best")
 
    # ---- bottom right: leave-one-out CV R2 (sparsity test)
    axis = fig.add_subplot(bottom[0, 2])
    pooled = sparse[sparse["scope"] == "ALL"].reset_index(drop=True)
    pooled = pd.concat([pooled[pooled["state_definition"] == "full"],
                        pooled[~pooled["state_definition"].isin(["full", "action only"])].sort_values("cv_r2"),
                        pooled[pooled["state_definition"] == "action only"]]).reset_index(drop=True)
    yy = np.arange(len(pooled))[::-1]
    full_cv = pooled.loc[0, "cv_r2"]
    colors = ["#333333" if d == "full" else "#999999" if d == "action only" else
              ("#C0392B" if cv > full_cv + args.cv_tol else "#2E8B57" if cv < full_cv - args.cv_tol else "#E0A800")
              for d, cv in zip(pooled["state_definition"], pooled["cv_r2"])]
    axis.barh(yy, 100 * pooled["cv_r2"], color=colors, height=0.6)
    for y_i, row in zip(yy, pooled.itertuples()):
        axis.text(100 * max(row.cv_r2, 0) + 0.4, y_i, f"{row.bits} bits | {row.median_rows_per_state_action:.0f} rows/cell",
                  va="center", fontsize=7)
    axis.axvline(100 * full_cv, color="black", linestyle="--", linewidth=1)
    ceiling = split[split["scope"] == "ALL"]["noise_ceiling_r2"].iloc[0]
    if np.isfinite(ceiling):
        axis.axvline(100 * ceiling, color=NOISE_COLOR, linestyle=":", linewidth=1.4)
        axis.text(100 * ceiling, 1.01, "noise ceiling", transform=axis.get_xaxis_transform(), color=NOISE_COLOR,
                  ha="right", fontsize=7)
    axis.set_yticks(yy)
    axis.set_yticklabels(pooled["state_definition"], fontsize=9)
    axis.set_xlim(0, 100 * max(ceiling if np.isfinite(ceiling) else 0, pooled["cv_r2"].max()) * 1.45)
    axis.set_xlabel("cross-validated R2 of a table lookup (%)")
    axis.set_title("Sparsity test: state without one feature\n(red bar longer than full = feature was harmful)",
                   fontsize=11, fontweight="bold")
    axis.grid(axis="x", alpha=0.25)
 
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
 
 
def run_variance(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features = args.features.split(",") if args.features else None
    df, features = load_log(args.csv, agent=args.agent, experiment=args.experiment, features=features)
    if not features:
        raise SystemExit("No state-feature columns found; pass --features.")
 
    groups = {f: [f] for f in features}
    for spec in (args.groups.split(",") if args.groups else []):
        name, members = spec.split("=")
        members = members.split("+")
        for m in members:
            groups.pop(m, None)
        groups[name] = members
    explicit = dict(item.split("=") for item in args.bit_widths.split(",")) if args.bit_widths else {}
    bits = {}
    for name, cols in groups.items():
        bits[name] = sum(int(explicit[c]) if c in explicit else
                         max(1, math.ceil(math.log2(max(df[c].nunique(), 2)))) for c in cols)
    print(f"Rows: {len(df)} | agents: {df['agent_id'].nunique()} | iterations: {df['iteration'].nunique()} | "
          f"feature groups: {', '.join(f'{g} ({b} bits)' for g, b in bits.items())} | total {sum(bits.values())} bits")
    if df["agent_id"].nunique() > 1 and not args.agent:
        print("Note: several agents pooled. Agent-specific differences that the state cannot see end up in 'unexplained'.")
 
    n_steps = df["step"].nunique()
    if args.step_labels:
        names = [s.strip() for s in args.step_labels.split(",")]
        if len(names) != n_steps:
            raise SystemExit(f"--step-labels needs {n_steps} entries, got {len(names)}.")
        labels = [f"Step {i}: {n}" for i, n in enumerate(names)]
    else:
        labels = [f"Step {i}" for i in range(n_steps)]
 
    anova_rows, split_rows, sparse_rows = [], [], []
    for step in sorted(df["step"].unique()):
        frame = df[df["step"] == step].reset_index(drop=True)
        r, s, sp = analyze_scope(frame, int(step), groups, ["mode"], bits, args)
        anova_rows += r
        split_rows.append(s)
        sparse_rows += sp
    r, s, sp = analyze_scope(df, "ALL", groups, ["mode", "step"], bits, args)
    anova_rows += r
    split_rows.append(s)
    sparse_rows += sp
 
    anova, split, sparse = pd.DataFrame(anova_rows), pd.DataFrame(split_rows), pd.DataFrame(sparse_rows)
    verdicts = prune_verdicts(anova, sparse, "ALL", args)
    verdicts["bits_if_pruned"] = verdicts["bits"].where(verdicts["verdict"] == "prune", 0)
 
    # per-step cv gain, to see whether a verdict holds at every step
    per_step = []
    for step in sorted(df["step"].unique()):
        cv = sparse[sparse["scope"] == step].set_index("state_definition")["cv_r2"]
        for g in groups:
            per_step.append({"step": step, "feature": g, "cv_r2_gain": cv["full"] - cv[f"-{g}"]})
    per_step = pd.DataFrame(per_step).pivot(index="feature", columns="step", values="cv_r2_gain")
    per_step.columns = [f"cv_gain_step{c}" for c in per_step.columns]
    verdicts = verdicts.merge(per_step.reset_index(), on="feature")
 
    anova.to_csv(out_dir / "state_feature_anova.csv", index=False)
    split.to_csv(out_dir / "state_variance_split.csv", index=False)
    sparse.to_csv(out_dir / "state_sparsity.csv", index=False)
    verdicts.to_csv(out_dir / "state_pruning_verdict.csv", index=False)
    title = (f"State-feature variance attribution  |  {df['agent_id'].nunique()} agents, {df['iteration'].nunique()} "
             f"iterations, {sum(bits.values())}-bit state")
    make_variance_figure(anova, split, sparse, verdicts, labels, list(groups), out_dir / "state_variance_analysis.png", title, args)
 
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("\n=== Variance split per step (shares in %; noise_ceiling_r2 = best any state can do) ===")
    shown = split.copy()
    for c in [c for c in shown.columns if c.startswith("share_") or c in ("r2_full", "adj_r2_full", "noise_ceiling_r2")]:
        shown[c] = (100 * shown[c]).round(1)
    print(shown.drop(columns=["share_noise_plot", "saturated_model"]).round(3).to_string(index=False))
    print("\n=== Feature ANOVA, all steps pooled (step and mode in the model) ===")
    cols = ["feature", "bits", "levels", "df_unique", "eta2_alone", "eta2_unique", "partial_omega2", "p_value",
            "x_mode_partial_omega2", "x_mode_p_value"]
    print(anova[anova["scope"] == "ALL"][cols].round(4).to_string(index=False))
    print("\n=== Sparsity test, all steps pooled ===")
    print(sparse[sparse["scope"] == "ALL"].drop(columns="scope").round(3).to_string(index=False))
    print("\n=== Verdict ===")
    print(verdicts[["feature", "bits", "omega2_main", "omega2_x_mode", "cv_r2_gain", "verdict", "reason"]].round(4).to_string(index=False))
    freed = int(verdicts["bits_if_pruned"].sum())
    print(f"\nPruning every 'prune' feature frees {freed} of {sum(bits.values())} bits. Confirm 'borderline' and the final set "
          f"with the ablation runs (command: ablation).\nWrote {out_dir / 'state_variance_analysis.png'} and 4 CSV files.")
 
 
# ------------------------------------------------------------------------------------ ablation
def run_metrics(frame, features, args):
    """Metrics of ONE training run (one experiment, one seed)."""
    daily = frame.groupby(["iteration", "agent_id"])["score"].sum().groupby("iteration").mean().sort_index()
    its = daily.index.to_numpy()
    values = daily.to_numpy()
    n_it = len(values)
    w = max(2, min(args.smooth, n_it // 4))
    smooth = pd.Series(values).rolling(w, min_periods=w).mean().to_numpy()
    tail = max(3, int(round(args.final_frac * n_it)))
    start, final = float(values[:w].mean()), float(values[-tail:].mean())
    peak = float(np.nanmax(smooth))
    target = start + 0.9 * (final - start)
    reached = np.flatnonzero(np.nan_to_num(smooth, nan=-np.inf) >= target) if final > start else []
    conv = float(its[reached[0]]) if len(reached) else float("nan")
 
    window = frame[frame["iteration"] >= its[-tail]]                       # converged phase only
    noise_rows = []
    for step, sub in window.groupby("step"):
        std, degrees, key = estimate_noise(sub, features, args.noise_key, args.min_noise_df)
        var = float(sub["score"].var(ddof=1)) if len(sub) > 1 else float("nan")
        noise_rows.append({"step": step, "pure_noise_std": std, "noise_df": degrees, "noise_key": key,
                           "total_std": math.sqrt(var) if var > 0 else 0.0,
                           "share_noise": min(1.0, std ** 2 / var) if np.isfinite(std) and var > 0 else float("nan")})
    noise = pd.DataFrame(noise_rows)
    ok = noise[noise["noise_df"] > 0]
    pooled_noise = math.sqrt(float((ok["noise_df"] * ok["pure_noise_std"] ** 2).sum() / ok["noise_df"].sum())) if len(ok) else float("nan")
    return ({"start_score": start, "peak_score": peak, "final_score": final, "convergence_iter": conv,
             "final_window_std": float(values[-tail:].std(ddof=1)), "final_window_n": tail,
             "pure_noise_std": pooled_noise, "share_noise": float(noise["share_noise"].mean())},
            pd.DataFrame({"iteration": its, "score": values, "smoothed": smooth}), noise)
 
 
def run_ablation(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features = args.features.split(",") if args.features else None
    df, features = load_log(args.csv, agent=args.agent, features=features)
    if "experiment" not in df.columns:
        raise SystemExit("The ablation CSV needs an `experiment` column.")
    if "seed" not in df.columns:
        df["seed"] = 0
    if args.baseline not in set(df["experiment"]):
        raise SystemExit(f"Baseline experiment '{args.baseline}' not found; available: {sorted(df['experiment'].unique())}")
 
    per_seed, curves, noise_all = [], [], []
    for (exp, seed), frame in df.groupby(["experiment", "seed"], sort=False):
        metrics, curve, noise = run_metrics(frame, features, args)
        per_seed.append({"experiment": exp, "seed": seed, **metrics})
        curve = curve.assign(experiment=exp, seed=seed)
        curves.append(curve)
        noise_all.append(noise.assign(experiment=exp, seed=seed))
    per_seed, curves, noise_all = pd.DataFrame(per_seed), pd.concat(curves), pd.concat(noise_all)
 
    metric_cols = ["start_score", "peak_score", "final_score", "convergence_iter", "final_window_std",
                   "pure_noise_std", "share_noise"]
    agg = per_seed.groupby("experiment", sort=False)[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["n_seeds"] = per_seed.groupby("experiment", sort=False).size()
    agg = agg.reset_index()
 
    base = per_seed[per_seed["experiment"] == args.baseline]
    b_peak, b_peak_sd = base["peak_score"].mean(), base["peak_score"].std(ddof=1) if len(base) > 1 else float("nan")
    b_gain = abs(b_peak - base["start_score"].mean())
    verdicts, d_peak_all, tol_all = [], [], []
    for exp in agg["experiment"]:
        sub = per_seed[per_seed["experiment"] == exp]
        if exp == args.baseline:
            verdicts.append("BASELINE")
            d_peak_all.append(0.0)
            tol_all.append(float("nan"))
            continue
        d_peak = sub["peak_score"].mean() - b_peak
        if len(sub) > 1 and len(base) > 1:
            se = math.sqrt(sub["peak_score"].var(ddof=1) / len(sub) + base["peak_score"].var(ddof=1) / len(base))
        else:                                                              # one seed: fall back on iteration scatter
            se = math.sqrt(sub["final_window_std"].iloc[0] ** 2 / sub["final_window_n"].iloc[0]
                           + base["final_window_std"].iloc[0] ** 2 / base["final_window_n"].iloc[0])
        tol = max(args.peak_tol * b_gain, 2 * se)
        e = agg[agg["experiment"] == exp].iloc[0]
        b = agg[agg["experiment"] == args.baseline].iloc[0]
        faster = np.isfinite(e["convergence_iter_mean"]) and np.isfinite(b["convergence_iter_mean"]) and \
            (b["convergence_iter_mean"] - e["convergence_iter_mean"]) / b["convergence_iter_mean"] >= 0.10
        steadier = (b["final_window_std_mean"] - e["final_window_std_mean"]) / b["final_window_std_mean"] >= 0.10
        quieter = np.isfinite(b["share_noise_mean"]) and (b["share_noise_mean"] - e["share_noise_mean"]) >= 0.02
        if d_peak < -tol:
            verdicts.append("KEEP: removing it costs peak score")
        elif d_peak > tol:
            verdicts.append("PRUNE: better without it")
        elif faster or steadier or quieter:
            why = [n for n, f in (("faster convergence", faster), ("more stable", steadier), ("less noise share", quieter)) if f]
            verdicts.append("PRUNE: harmful sparsity (" + ", ".join(why) + ")")
        else:
            verdicts.append("NEUTRAL: no measurable effect (prune for parsimony)")
        d_peak_all.append(d_peak)
        tol_all.append(tol)
    agg["delta_peak_vs_baseline"] = d_peak_all
    agg["peak_tolerance"] = tol_all
    agg["verdict"] = verdicts
 
    per_seed.to_csv(out_dir / "ablation_per_seed.csv", index=False)
    agg.to_csv(out_dir / "ablation_summary.csv", index=False)
    noise_all.to_csv(out_dir / "ablation_noise_variance_split.csv", index=False)
    make_ablation_figure(per_seed, curves, agg, args, out_dir / "ablation_analysis.png")
 
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("=== Ablation summary (mean over seeds) ===")
    show = agg[["experiment", "n_seeds", "peak_score_mean", "peak_score_std", "final_score_mean", "convergence_iter_mean",
                "final_window_std_mean", "pure_noise_std_mean", "share_noise_mean", "delta_peak_vs_baseline",
                "peak_tolerance", "verdict"]]
    print(show.round(3).to_string(index=False))
    if (agg["n_seeds"] < 3).any():
        print("\nWarning: fewer than 3 seeds for some experiments; the tolerance then relies on iteration scatter.")
    print("\nNote: share_noise = noise^2 / total variance, so it can rise when a better policy REMOVES explainable variance; "
          "read it together with the absolute pure_noise_std.")
    print(f"Wrote {out_dir / 'ablation_analysis.png'} and 3 CSV files.")
 
 
def make_ablation_figure(per_seed, curves, agg, args, out_path):
    exps = list(agg["experiment"])
    colors = {e: ("#333333" if e == args.baseline else plt.get_cmap("tab10")(i % 10)) for i, e in enumerate(exps)}
    fig = plt.figure(figsize=(17, 10.5))
    grid = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.28)
 
    axis = fig.add_subplot(grid[0, :2])
    for e in exps:
        c = curves[curves["experiment"] == e]
        m = c.groupby("iteration")["smoothed"].agg(["mean", "std"])
        axis.plot(m.index, m["mean"], color=colors[e], linewidth=2.6 if e == args.baseline else 1.6, label=e)
        if c["seed"].nunique() > 1:
            axis.fill_between(m.index, m["mean"] - m["std"], m["mean"] + m["std"], color=colors[e], alpha=0.12)
    axis.set_xlabel("iteration")
    axis.set_ylabel(f"mean agent-day score (rolling mean, window {args.smooth})")
    axis.set_title("Training curves (band: std across seeds)", fontsize=11, fontweight="bold")
    axis.legend(fontsize=8, ncol=3, loc="lower right")
    axis.grid(alpha=0.25)
 
    axis = fig.add_subplot(grid[0, 2])
    y = np.arange(len(exps))[::-1]
    base_row = agg[agg["experiment"] == args.baseline].iloc[0]
    for yy, e in zip(y, exps):
        r = agg[agg["experiment"] == e].iloc[0]
        label = r["verdict"]
        col = "#333333" if label == "BASELINE" else "#2E8B57" if label.startswith("KEEP") else \
            "#C0392B" if label.startswith("PRUNE") else "#E0A800"
        err = 0 if pd.isna(r["peak_score_std"]) else r["peak_score_std"]
        axis.errorbar(r["peak_score_mean"], yy, xerr=err, fmt="o", color=col, markersize=8, capsize=4)
        axis.text(1.02, yy, label.split(":")[0], transform=axis.get_yaxis_transform(), va="center", fontsize=8,
                  color=col, fontweight="bold")
        if not pd.isna(r["peak_tolerance"]):
            axis.plot([base_row["peak_score_mean"] - r["peak_tolerance"]] * 2, [yy - 0.25, yy + 0.25], color="grey", linewidth=1)
    axis.axvline(base_row["peak_score_mean"], color="#333333", linestyle="--", linewidth=1)
    axis.set_yticks(y)
    axis.set_yticklabels(exps, fontsize=9)
    axis.set_xlabel("peak score (mean +/- std over seeds)")
    axis.set_title("Peak score vs baseline\n(grey tick: loss that counts as real)", fontsize=11, fontweight="bold")
    axis.grid(axis="x", alpha=0.25)
 
    def bar_panel(position, column, title, ylabel, scale=1.0):
        ax = fig.add_subplot(grid[1, position])
        vals = agg[f"{column}_mean"].to_numpy(float) * scale
        errs = np.nan_to_num(agg[f"{column}_std"].to_numpy(float)) * scale
        ax.bar(range(len(exps)), vals, yerr=errs, color=[colors[e] for e in exps], capsize=4, width=0.65)
        ax.axhline(vals[exps.index(args.baseline)], color="#333333", linestyle="--", linewidth=1)
        ax.set_xticks(range(len(exps)))
        ax.set_xticklabels(exps, rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
 
    bar_panel(0, "convergence_iter", "Convergence speed\n(iteration reaching 90% of the final gain)", "iteration")
    bar_panel(1, "final_window_std", "Stability\n(std of the score in the final window)", "score std")
    bar_panel(2, "share_noise", "Residual pure noise share\n(converged phase)", "% of score variance", scale=100)
    fig.suptitle("State-feature ablation: full state vs leave-one-group-out training runs", fontsize=13,
                 fontweight="bold", y=0.995)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
 
 
# ------------------------------------------------------------------------------------------ main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
 
    def common(p):
        p.add_argument("csv")
        p.add_argument("--out-dir", default=".")
        p.add_argument("--agent", default=None)
        p.add_argument("--features", default=None, help="comma-separated state columns (default: every non-reserved column)")
        p.add_argument("--noise-key", choices=["auto", "prefix", "state"], default="auto",
                       help="replicate definition: prefix = identical mode history + state; state = identical this-mode + state")
        p.add_argument("--min-noise-df", type=int, default=30)
 
    v = sub.add_parser("variance", help="Phase 2: ANOVA, sparsity test and prune verdict")
    common(v)
    v.add_argument("--experiment", default=None, help="if the CSV has an experiment column, analyse only this one")
    v.add_argument("--groups", default=None, help='analyse columns as one unit, e.g. "location=origin_taz+dest_taz,load=cong_bin"')
    v.add_argument("--bit-widths", default=None, help='e.g. "origin_taz=5,dest_taz=5,time_bin=3" (default: ceil(log2(levels)))')
    v.add_argument("--step-labels", default=None)
    v.add_argument("--min-effect", type=float, default=0.01, help="partial omega2 below this counts as no signal")
    v.add_argument("--cv-tol", type=float, default=0.002, help="CV R2 change that counts as real (0.002 = 0.2 points)")
    v.add_argument("--cv-folds", type=int, default=5)
    v.add_argument("--cv-repeats", type=int, default=2)
    v.add_argument("--cv-shrink", type=float, default=2.0, help="pseudo-observations pulling a cell towards the action mean")
    v.add_argument("--min-obs", type=int, default=5, help="a state-action cell with fewer rows is 'thin'")
    v.add_argument("--max-inter-cols", type=int, default=800)
 
    a = sub.add_parser("ablation", help="Phase 3: compare training runs with different state definitions")
    common(a)
    a.add_argument("--baseline", default="full")
    a.add_argument("--smooth", type=int, default=10, help="rolling-mean window for curves, peak and convergence")
    a.add_argument("--final-frac", type=float, default=0.2, help="share of last iterations used as 'converged' window")
    a.add_argument("--peak-tol", type=float, default=0.02, help="peak loss below this share of the baseline learning gain is ignored")
 
    args = parser.parse_args()
    run_variance(args) if args.command == "variance" else run_ablation(args)
 
 
if __name__ == "__main__":
    main()