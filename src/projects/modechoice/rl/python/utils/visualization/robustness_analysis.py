#!/usr/bin/env python3
"""
Seed-robustness report for single-agent RL runs: one agent_tracking_<seed>.csv per seed and condition,
optionally one q_table.pkl (final checkpoint) per seed.

    python seed_robustness_report.py \
        --group "no prev. mode=data/noprev/*.csv"  --qtables "no prev. mode=runs/noprev/seed_*/q_table.pkl" \
        --group "with prev. mode=data/prev/*.csv"  --qtables "with prev. mode=runs/prev/seed_*/q_table.pkl" \
        --out report
(or use run_report.sh)

Any number of groups can be given. Seeds are read from the 'seed' column of the CSVs and from
config_arg['randomSeed'] of the pickles, so file and folder names do not matter. Byte-identical CSVs inside a
group are reported and dropped. --qtables is optional (without it, convergence is judged on the updates alone).

Figures written to --out
  fig1_seed_effect.png          per group: reward curves per seed, test-phase plan per seed, final reward per seed
  fig3_robustness_metrics.png   all groups side by side (4 plots): final reward per seed, between-seed spread,
                                optimal vs suboptimal convergence, convergence iteration
  metrics.csv                   one row per run, including the Q-table entropy used for the convergence check

Definitions
  test phase     the last --test-iters iterations (greedy, no exploration, no Q updates); training = the rest
  best plan      plan (experienced_modes) with the highest mean day reward among plans seen >= --min-plan-count
                 times, pooled over all runs. Its mean reward is the "converged policy average" (reference)
  test_reward    mean day reward over the test phase;  gap = test_reward - reference
  tolerance      gap counts as "on par" if gap >= -tol, with tol = --opt-z * SD(best plan day reward) / sqrt(test_iters)
                 (the noise of a test average of a policy that earns the reference), or --opt-tol in reward units
  update signal  per training iteration: |delta_q| if the CSV has a delta_q column, otherwise 1 if the plan differs
                 from the previous iteration, else 0 ("plan switch")
  still_updating mean update signal over the last --update-window training iterations is above --dq-tol (delta_q)
                 or --switch-tol (plan switches)
  H_tau<t>       Shannon entropy of softmax(Q(s,.)/t), normalised to [0,1], averaged over the states weighted by
                 visits (state_count), from the final q_table.pkl. Low = decisive policy.
  converged      NOT still_updating  AND  H_tau<first --tau> <= --entropy-max (the entropy test is skipped for seeds
                 without a q table)
  conv_it        convergence iteration = the last training iteration whose update signal is above its threshold
                 (the policy no longer changes afterwards); NaN if the seed has not converged
  status         optimal         gap >= -tol: the test average equals the converged-policy average
                 suboptimal      gap < -tol and converged: settled on a suboptimal policy (local optimum)
                 not_converged   gap < -tol and not converged: may still improve
  final          mean day reward over the last 100 iterations;  regret = mean over iterations of max(0, best - reward)
  test_is_best   the most frequent plan of the test phase is the best plan (test_best_share = share of test days)
  q_margin / q_tied_share / q_untried_share  further Q-table diagnostics written to metrics.csv (visit-weighted gap
                 between best and second-best Q, share of near-tied states, share of states with a never-tried action)
"""
import argparse, glob, hashlib, os, pickle, re, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

rng = np.random.default_rng(0)
COL = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
GREEN, GREY = "#2ca02c", "#bbbbbb"
STATUS_STYLE = {"optimal": dict(color=GREEN, marker="o"), "not_converged": dict(color="#ff7f0e", marker="^"), "suboptimal": dict(color="#d62728", marker="X")}
STATUS_LABEL = {"optimal": "optimal (test avg = converged-policy avg)", "not_converged": "suboptimal, not converged yet",
                "suboptimal": "suboptimal convergence (stuck)"}


# ------------------------------------------------------------------------------------------------ loading
def read(f):
    return pd.read_csv(f, dtype={"experienced_modes": str}).sort_values("iteration").reset_index(drop=True)


def dq_column(d):
    for c in d.columns:
        if c.lower().replace("_", "") in ("deltaq", "maxdeltaq", "meandeltaq", "absdeltaq"):
            return c
    return None


def load_groups(specs):
    G = {}
    for sp in specs:
        name, pat = sp.split("=", 1)
        files = sorted(glob.glob(pat)); seen, runs = {}, {}
        for f in files:
            h = hashlib.md5(open(f, "rb").read()).hexdigest()
            if h in seen:
                print(f"WARNING [{name}]: {os.path.basename(f)} is byte-identical to {seen[h]} -> dropped (duplicate run)"); continue
            seen[h] = os.path.basename(f); d = read(f)
            if d.agent_id.nunique() > 1:
                print(f"WARNING [{name}]: {os.path.basename(f)} has {d.agent_id.nunique()} agents (not a single-agent run) -> dropped"); continue
            s = int(d.seed.iloc[0])
            if s in runs:
                print(f"WARNING [{name}]: seed {s} appears twice ({os.path.basename(f)}) -> dropped"); continue
            runs[s] = d
        G[name] = runs; print(f"{name}: {len(runs)} runs, seeds {sorted(runs)}")
    return G


def softmax_entropy(Q, tau):
    A = Q.shape[1]
    if A < 2:
        return np.zeros(len(Q))
    z = Q / tau; z = z - z.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    return -(p * np.log(p + 1e-12)).sum(1) / np.log(A)


def q_stats(path, taus, tie_tol):
    """Statistics of one final q_table.pkl: {'q_tables': {agent: {state: {mode: Q}}}, 'memories': {...}, 'config_arg': {...}}."""
    d = pickle.load(open(path, "rb")); cfg = d.get("config_arg", {}) or {}
    seed = cfg.get("randomSeed")
    if seed is None:
        m = re.search(r"seed_(\d+)", path); seed = int(m.group(1)) if m else None
    if seed is None:
        raise ValueError("no randomSeed in config_arg and no seed_<n> in the path")
    tol = tie_tol if tie_tol is not None else float(cfg.get("tieTolerance", 1.0))
    Q, N, W = [], [], []
    for agent, qs in d["q_tables"].items():
        mem = (d.get("memories") or {}).get(agent, {})
        for s, qa in qs.items():
            acts = sorted(qa); counts = [mem.get(s, {}).get("action_counts", {}).get(a, 0) for a in acts]
            Q.append([qa[a] for a in acts]); N.append(counts)
            W.append(mem.get(s, {}).get("state_count", sum(counts)))
    Q, N, W = np.array(Q, float), np.array(N, float), np.array(W, float)
    keep = W > 0; Q, N, W = Q[keep], N[keep], W[keep]
    w = W / W.sum(); srt = np.sort(Q, 1); gap = srt[:, -1] - srt[:, -2]
    out = dict(seed=int(seed), q_states=len(Q), q_iteration=d.get("iteration"), q_margin=float((w * gap).sum()),
               q_tied_share=float((w * (gap < tol)).sum()), q_untried_share=float((w * (N == 0).any(1)).sum()))
    for t in taus:
        out[f"H_tau{t:g}"] = float((w * softmax_entropy(Q, t)).sum())
    return out


def load_q(specs, a):
    rows = []
    for sp in specs:
        name, pat = sp.split("=", 1); seen = set()
        for f in sorted(glob.glob(pat)):
            try:
                r = q_stats(f, a.tau, a.tie_tol)
            except Exception as e:
                print(f"WARNING [{name}]: could not read {f}: {e}"); continue
            if r["seed"] in seen:
                print(f"WARNING [{name}]: seed {r['seed']} appears twice among the q tables ({f}) -> dropped"); continue
            seen.add(r["seed"]); r["group"] = name; rows.append(r)
        print(f"{name}: {len(seen)} q tables")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ metrics
def update_signal(train, use_dq):
    if use_dq:
        return train[dq_column(train)].abs().fillna(0).values
    sw = np.array((train.experienced_modes != train.experienced_modes.shift()).astype(float).values)
    sw[0] = 0.0
    return sw


def metrics(G, a, use_dq):
    """Per-run metrics from the CSVs (rewards, test phase, updates remaining)."""
    allr = pd.concat([d for r in G.values() for d in r.values()])
    pm = allr.groupby("experienced_modes").total_day_reward.agg(["mean", "size"]); pm = pm[pm["size"] >= a.min_plan_count]
    best_plan, best = pm["mean"].idxmax(), pm["mean"].max()
    rank = pm["mean"].rank(ascending=False, method="min")
    sd_best = allr.loc[allr.experienced_modes == best_plan, "total_day_reward"].std()
    tol = a.opt_tol if a.opt_tol is not None else a.opt_z * sd_best / np.sqrt(a.test_iters)
    sig_thr = a.dq_tol if use_dq else 0.0; still_thr = a.dq_tol if use_dq else a.switch_tol
    rows = []
    for g, runs in G.items():
        for s, d in runs.items():
            n = len(d); cut = max(n - a.test_iters, 1); train, test = d.iloc[:cut], d.iloc[cut:]
            vc = test.experienced_modes.value_counts(); tp = vc.index[0]
            sig = update_signal(train, use_dq); act = np.where(sig > sig_thr)[0]
            last_active = train.iteration.iloc[act[-1]] if len(act) else train.iteration.iloc[0]
            late = sig[-a.update_window:].mean(); last = d.iloc[-100:]; test_reward = test.total_day_reward.mean()
            rows.append(dict(group=g, seed=s, final=last.total_day_reward.mean(),
                             regret=(best - d.total_day_reward).clip(lower=0).mean(),
                             test_reward=test_reward, gap=test_reward - best, test_plan=tp, test_plan_share=vc.iloc[0] / len(test),
                             test_is_best=bool(tp == best_plan), test_best_share=(test.experienced_modes == best_plan).mean(),
                             test_plan_rank=rank.get(tp, np.nan), late_update=late, still_updating=bool(late > still_thr),
                             last_active_it=last_active, cutoff_it=train.iteration.iloc[-1],
                             n_plans=d.experienced_modes.nunique(), n_iter=n))
    return pd.DataFrame(rows), best, best_plan, pm, tol


def classify(M, tol, hcol, a):
    """Convergence = no updates remaining AND (if a q table exists) Q-table entropy <= --entropy-max."""
    M["entropy_ok"] = ~(M[hcol] > a.entropy_max)          # NaN (no q table) -> test skipped -> True
    M["converged"] = (~M.still_updating) & M.entropy_ok
    M["conv_it"] = np.where(M.converged, M.last_active_it, np.nan)
    M["status"] = np.where(M.gap >= -tol, "optimal", np.where(M.converged, "suboptimal", "not_converged"))
    return M


def boot_ci(x, f=np.mean, n=3000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2: return (np.nan, np.nan)
    return tuple(np.percentile([f(rng.choice(x, len(x))) for _ in range(n)], [2.5, 97.5]))


def err(med, lo, hi):
    return [[max(med - lo, 0)], [max(hi - med, 0)]]


# ------------------------------------------------------------------------------------------------ figure 1
def fig1(G, M, best, best_plan, a, out):
    names = list(G); w = a.window; t = a.test_iters
    fig, ax = plt.subplots(len(names), 3, figsize=(19, 4.6 * len(names)), squeeze=False)
    others = [p for p in M.test_plan.value_counts().index if p != best_plan][:2]
    pc = {best_plan: GREEN}; pc.update(dict(zip(others, ["#1f77b4", "#ff7f0e"]))); pc["other"] = GREY
    for r, g in enumerate(names):
        runs = G[g]; c = COL[r % 6]; seeds = sorted(runs)
        curves = pd.DataFrame({s: runs[s].set_index("iteration").total_day_reward.rolling(w).mean() for s in seeds})
        for s in seeds: ax[r, 0].plot(curves.index, curves[s], color=c, lw=.9, alpha=.55)
        ax[r, 0].plot(curves.index, curves.mean(axis=1), color="k", lw=2.2, label="mean over seeds")
        ax[r, 0].fill_between(curves.index, curves.min(axis=1), curves.max(axis=1), color=c, alpha=.12, label="min-max over seeds")
        ax[r, 0].axvspan(runs[seeds[0]].iteration.iloc[-t], curves.index.max(), color="k", alpha=.08, label=f"test phase ({t} its)")
        ax[r, 0].axhline(best, color="k", ls=":", lw=1)
        ax[r, 0].set(ylim=(100, 510), xlabel="iteration", ylabel=f"day reward (rolling {w})", title=f"{g}: reward per seed")
        ax[r, 0].legend(fontsize=8, loc="lower right"); ax[r, 0].grid(alpha=.3)
        # plan composition in the test phase
        comp = np.array([[(runs[s].experienced_modes.iloc[-t:] == p).mean() if p != "other" else 0 for p in pc] for s in seeds])
        comp[:, -1] = 1 - comp[:, :-1].sum(1); left = np.zeros(len(seeds))
        for j, p in enumerate(pc):
            ax[r, 1].barh(range(len(seeds)), comp[:, j], left=left, color=pc[p], label=p + (" (best)" if p == best_plan else "")); left += comp[:, j]
        ax[r, 1].set_yticks(range(len(seeds))); ax[r, 1].set_yticklabels([f"seed {s}" for s in seeds]); ax[r, 1].invert_yaxis()
        k = int(M[M.group == g].test_is_best.sum())
        ax[r, 1].set(xlim=(0, 1), xlabel=f"share of the last {t} (test) days", title=f"{g}: test-phase plan per seed\n{k}/{len(seeds)} seeds run the best plan")
        if r == 0: ax[r, 1].legend(fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.14), ncol=2)
        # final reward per seed, green = test phase on the best plan
        x = M[M.group == g].sort_values("seed")
        ax[r, 2].bar(range(len(x)), x.final, color=[GREEN if b else GREY for b in x.test_is_best]); ax[r, 2].axhline(best, color="k", ls=":", lw=1)
        ax[r, 2].set_xticks(range(len(x))); ax[r, 2].set_xticklabels(x.seed.astype(str), rotation=45); ax[r, 2].set_ylim(min(350, x.final.min() - 10), 510)
        ax[r, 2].set(title=f"{g}: final reward per seed (green = test phase on best plan)\nmedian {x.final.median():.1f}, IQR {x.final.quantile(.75)-x.final.quantile(.25):.1f}",
                     ylabel="mean reward, last 100 its")
        ax[r, 2].grid(alpha=.3, axis="y")
    fig.suptitle(f"Effect of the random seed (thick line = mean over seeds, band = min-max; dotted = mean reward of the best plan {best_plan})", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .97)); fig.savefig(os.path.join(out, "fig1_seed_effect.png"), dpi=140); plt.close(fig)


# ------------------------------------------------------------------------------------------------ figure 3
def fig3(M, best, tol, hcol, use_dq, a, out):
    names = list(dict.fromkeys(M.group)); fig, ax = plt.subplots(2, 2, figsize=(15, 10.5)); xs = np.arange(len(names))
    tops, labels = [], []; iqr = lambda v: np.percentile(v, 75) - np.percentile(v, 25); have_h = M[hcol].notna().any()
    uname = "delta_q" if use_dq else "plan switches"
    for i, g in enumerate(names):
        x = M[M.group == g].sort_values("seed"); c = COL[i % 6]; j = rng.normal(0, .06, len(x)); xi = np.full(len(x), i) + j
        st = x.status.values; gp = x.gap.values
        # (0,0) final reward per seed
        ax[0, 0].scatter(xi, x.final, color=c, s=40, alpha=.8, zorder=3)
        med = x.final.median(); lo, hi = boot_ci(x.final, np.median); ax[0, 0].errorbar(i + .22, med, yerr=err(med, lo, hi), color="k", marker="_", ms=14, capsize=3)
        # (0,1) between-seed spread
        q = iqr(x.final); lq, hq = boot_ci(x.final, iqr); ax[0, 1].bar(i, q, color=c, alpha=.8)
        ax[0, 1].errorbar(i, q, yerr=err(q, lq, hq), color="k", capsize=4); top = hq if hq == hq else q; tops.append(top)
        ax[0, 1].text(i, top * 1.02, f"SD {x.final.std():.1f}", ha="center", fontsize=8)
        # (1,0) optimal vs suboptimal convergence: test average vs converged-policy average
        for s_, sty in STATUS_STYLE.items():
            m = st == s_
            if m.any(): ax[1, 0].scatter(xi[m], gp[m], s=80, zorder=3, **sty)
        for xx, yy, sd in zip(xi[st != "optimal"], gp[st != "optimal"], x.seed.values[st != "optimal"]):
            ax[1, 0].annotate(f"seed {sd}", (xx, yy), textcoords="offset points", xytext=(7, 3), fontsize=8)
        labels.append(f"{g}\n{int((st == 'suboptimal').sum())} suboptimal, {int((st == 'not_converged').sum())} not converged of {len(x)}")
        # (1,1) convergence iteration; x = not converged at the training cutoff
        conv = x.conv_it.values; nc = np.isnan(conv)
        ax[1, 1].scatter(xi[~nc], conv[~nc], color=c, s=40, alpha=.8, zorder=3)
        ax[1, 1].scatter(xi[nc], x.cutoff_it.values[nc], color="k", marker="x", s=70, zorder=4)
    for a_ in (ax[0, 0], ax[0, 1], ax[1, 1]): a_.set_xticks(xs); a_.set_xticklabels(names, rotation=20)
    for a_ in (ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]): a_.grid(alpha=.3, axis="y")
    ax[0, 0].axhline(best, color="k", ls=":"); ax[0, 0].set(ylabel="mean reward, last 100 its", title="Final reward per seed (bar = median, 95% bootstrap CI)")
    ax[0, 1].set_ylim(0, max(tops) * 1.18); ax[0, 1].set(ylabel="reward points", title="Between-seed IQR of final reward (robust spread; 95% bootstrap CI)")
    gaps = M.gap.values; lo_, hi_ = min(-3 * tol, np.nanmin(gaps) - tol), max(3 * tol, np.nanmax(gaps) + tol)
    ax[1, 0].axhspan(-tol, tol, color=GREEN, alpha=.12); ax[1, 0].axhline(0, color="k", ls=":", lw=1)
    ax[1, 0].set_ylim(lo_, hi_ + .4 * (hi_ - lo_)); ax[1, 0].set_xlim(-.5, len(names) - .5)  # headroom for the legend
    ax[1, 0].set_xticks(xs); ax[1, 0].set_xticklabels(labels, fontsize=9)
    ax[1, 0].legend(handles=[Line2D([0], [0], linestyle="", markersize=9, label=STATUS_LABEL[k], **v) for k, v in STATUS_STYLE.items()], fontsize=8, loc="upper left")
    ax[1, 0].set(ylabel=f"test average - converged-policy average ({best:.1f})",
                 title=f"Optimal vs suboptimal convergence (test phase, last {a.test_iters} its)\ngreen band = on par (+/-{tol:.1f} = noise of the converged policy)")
    ax[1, 1].axhline(M.cutoff_it.max(), color="k", ls=":", lw=1)
    rule = f"no updates left ({uname})" + (f" and entropy {hcol} <= {a.entropy_max:g}" if have_h else "")
    ax[1, 1].set(ylabel="iteration", title=f"Convergence iteration = last policy update\nconverged = {rule}; x = not converged at the cutoff (dotted)")
    fig.suptitle("Robustness metrics per condition", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(os.path.join(out, "fig3_robustness_metrics.png"), dpi=140); plt.close(fig)


# ------------------------------------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", action="append", required=True, help='"name=glob of the CSVs" (repeatable)')
    p.add_argument("--qtables", action="append", default=[], help='"name=glob of the q_table.pkl files" (repeatable, optional)')
    p.add_argument("--compare", nargs=2, default=None, help="ignored (the paired-comparison figure was removed)")
    p.add_argument("--out", default="report")
    p.add_argument("--window", type=int, default=30, help="rolling window of the curves")
    p.add_argument("--test-iters", type=int, default=10, help="length of the test phase = last N iterations")
    p.add_argument("--update-window", type=int, default=50, help="training iterations used for 'updates remaining'")
    p.add_argument("--switch-tol", type=float, default=.05, help="still updating if plan-switch rate in the window exceeds this")
    p.add_argument("--dq-tol", type=float, default=.5, help="delta_q threshold (reward units) when the CSV has a delta_q column")
    p.add_argument("--entropy-max", type=float, default=.5, help="converged only if the normalised Q-table entropy (first --tau) is <= this")
    p.add_argument("--opt-z", type=float, default=2.0, help="on-par tolerance = z * SD(best plan day reward) / sqrt(test iters)")
    p.add_argument("--opt-tol", type=float, default=None, help="on-par tolerance in reward units (overrides --opt-z)")
    p.add_argument("--tau", type=float, nargs="+", default=[10.0, 30.0], help="softmax temperature(s); the first one is used for convergence")
    p.add_argument("--tie-tol", type=float, default=None, help="tie tolerance for q_tied_share (default: tieTolerance of the pickle, else 1.0)")
    p.add_argument("--min-plan-count", type=int, default=50, help="a plan needs this many days to qualify as the best plan")
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True)

    G = load_groups(a.group)
    use_dq = all(dq_column(d) for r in G.values() for d in r.values())
    print("update signal:", "delta_q column of the CSVs" if use_dq else "plan switches (no delta_q column in the CSVs)")
    M, best, best_plan, pm, tol = metrics(G, a, use_dq); hcol = f"H_tau{a.tau[0]:g}"
    print(f"best plan (pooled): {best_plan}  mean reward {best:.2f} (= converged-policy average); on-par tolerance +/-{tol:.2f}")
    print(pm.sort_values("mean", ascending=False).head(4).round(2).to_string())

    if a.qtables:
        Q = load_q(a.qtables, a)
        if len(Q):
            for g in G:
                qs = set(Q[Q.group == g].seed); cs = set(G[g])
                if qs - cs: print(f"WARNING [{g}]: q tables for seeds {sorted(qs - cs)} have no CSV run")
                if cs - qs: print(f"WARNING [{g}]: no q table for seeds {sorted(cs - qs)} (entropy test skipped for them)")
            M = M.merge(Q, on=["group", "seed"], how="left")
    if hcol not in M:
        M[hcol] = np.nan; print("no q tables: convergence is judged on the updates remaining only")
    M = classify(M, tol, hcol, a); M.to_csv(os.path.join(a.out, "metrics.csv"), index=False)

    cols = [c for c in ["final", "regret", "test_reward", "gap", "late_update", "conv_it", hcol] if c in M and M[c].notna().any()]
    print(M.groupby("group")[cols].median().round(3).to_string())
    print(M.groupby("group").agg(seeds=("seed", "size"), still_updating=("still_updating", "sum"), entropy_too_high=("entropy_ok", lambda s: int((~s).sum())),
                                 converged=("converged", "sum")).to_string())
    print(M.groupby(["group", "status"]).seed.apply(lambda s: ", ".join(map(str, sorted(s)))).to_string())

    fig1(G, M, best, best_plan, a, a.out); fig3(M, best, tol, hcol, use_dq, a, a.out); print("saved figures and metrics.csv in", a.out)


if __name__ == "__main__":
    main()