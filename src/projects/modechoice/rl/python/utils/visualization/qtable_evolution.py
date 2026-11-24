#!/usr/bin/env python3
"""
Visualise how a tabular Q-table evolves across checkpoint pickles.

Usage
-----
    python qtable_evolution.py --folder /path/to/pickles --out out_dir
    python qtable_evolution.py --folder . --pattern "q_table*.pkl" --init 600 --interp 8

Inputs : pickle files written by DecentralizedQLearningAgent.dump()
         {"iteration", "config_arg", "q_tables": {agent_tag: {state: {mode: q}}},
          "memories": {agent_tag: {state: {"state_count", "action_counts"}}}}
Outputs: <out>/qtable_dashboard.png   static summary of the learning dynamics
         <out>/qtable_breathing.gif   3D animation (use --mp4 if ffmpeg is installed)

3D animation layout
-------------------
 * Each state is a point; its (x, y) position is a fixed 2D PCA projection of the state's
   bit-vector, so similar states sit close together and nothing jumps between frames.
 * Height (z)   = max_a Q(s, a)                      -> the table "breathes" as values move
 * Colour       = greedy action argmax_a Q(s, a)     -> policy flips are visible
 * Size         = number of visits to the state
 * Faded grey   = state seen but never updated (all Q still at the initial value)
 * States appear when they are first visited; frames between checkpoints are interpolated.

Requirements: numpy, matplotlib (+ pillow for GIF). No other dependencies.
"""
import argparse, glob, os, pickle, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.lines import Line2D

ACTION_COLORS = {"bike": "#2ca02c", "car": "#d62728", "pedestrian": "#ff7f0e", "pt": "#1f77b4"}
FALLBACK = ["#9467bd", "#8c564b", "#e377c2", "#17becf"]


# ----------------------------------------------------------------------------- loading
def load_checkpoints(folder, pattern, agent):
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files:
        sys.exit(f"No files match {os.path.join(folder, pattern)}")
    cps = []
    for f in files:
        try:
            with open(f, "rb") as fh:
                d = pickle.load(fh)
            cps.append((int(d.get("iteration", 0)), f, d))
        except Exception as e:
            print(f"skip {f}: {e}")
    cps.sort(key=lambda t: t[0])
    tag = agent
    if tag is None:
        tag = next(iter(cps[0][2]["q_tables"]))
    print(f"{len(cps)} checkpoints, agent tag = {tag}")
    out = []
    for it, f, d in cps:
        qt = d["q_tables"].get(tag) or d["q_tables"].get(str(tag))
        mem = d.get("memories", {}).get(tag) or d.get("memories", {}).get(str(tag)) or {}
        if qt is None:
            print(f"skip {f}: agent {tag} not in file")
            continue
        out.append(dict(iteration=it - 1, file=os.path.basename(f), q=qt, mem=mem))  # saved counter = iteration + 1
    return out


def build_tensors(cps, init):
    """Union state index in first-seen order -> Q[T, S, A], visits[T, S], seen[T, S]."""
    modes = []
    for c in cps:
        for qa in c["q"].values():
            for m in qa:
                if m not in modes:
                    modes.append(m)
    modes = sorted(modes)
    states, index = [], {}
    for c in cps:
        for s in c["q"]:
            if s not in index:
                index[s] = len(states)
                states.append(s)
    T, S, A = len(cps), len(states), len(modes)
    Q = np.full((T, S, A), np.nan)
    V = np.zeros((T, S))
    C = np.zeros((T, S, A))
    for t, c in enumerate(cps):
        for s, qa in c["q"].items():
            i = index[s]
            for m, v in qa.items():
                Q[t, i, modes.index(m)] = v
            V[t, i] = c["mem"].get(s, {}).get("state_count", 0)
            for m, n in c["mem"].get(s, {}).get("action_counts", {}).items():
                if m in modes:
                    C[t, i, modes.index(m)] = n
    seen = ~np.isnan(Q).all(axis=2)
    return states, modes, Q, V, C, seen


def pca_layout(states):
    """2D PCA of bit-vector states; non-numeric states (e.g. 'TERMINAL') go to the origin."""
    rows, ok = [], []
    L = max((len(s) for s in states if isinstance(s, (tuple, list))), default=0)
    for s in states:
        try:
            r = np.array([float(b) for b in s], float)
            r = np.pad(r, (0, L - len(r)))
            rows.append(r); ok.append(True)
        except Exception:
            rows.append(np.zeros(L)); ok.append(False)
    X = np.array(rows)
    Xc = X - X[ok].mean(axis=0) if any(ok) else X
    if L < 2 or not any(ok):
        xy = np.c_[np.arange(len(states)) % 20, np.arange(len(states)) // 20].astype(float)
    else:
        _, _, vt = np.linalg.svd(Xc[ok], full_matrices=False)
        xy = Xc @ vt[:2].T
        # tiny deterministic jitter so states with identical projections don't overlap exactly
        rng = np.random.default_rng(0)
        xy = xy + rng.normal(0, 0.02 * (np.ptp(xy, axis=0) + 1e-9), xy.shape)
    return xy


TIE_ORDER = {"bike": 0, "pt": 1, "pedestrian": 2, "car": 3}   # same as DecentralizedQLearningAgent.TIE_ORDER


def greedy_actions(Qf, C, seen, modes):
    """Greedy action per (checkpoint, state) using the agent's own rule:
    1) highest Q  2) most-tried action  3) fixed priority (TIE_ORDER)."""
    q = np.where(seen[:, :, None], Qf, -np.inf)
    cand = q == q.max(2, keepdims=True)
    c = np.where(cand, C, -1)
    cand &= c == c.max(2, keepdims=True)
    pri = np.array([TIE_ORDER.get(m, len(TIE_ORDER)) for m in modes], float)
    p = np.where(cand, pri, np.inf)
    return p.argmin(2)


# ----------------------------------------------------------------------------- dashboard
def dashboard(cps, modes, Q, V, C, seen, init, path):
    its = np.array([c["iteration"] for c in cps])
    T, S, A = Q.shape
    Qf = np.where(np.isnan(Q), init, Q)
    n_states = seen.sum(1)
    updated = (~np.isclose(Qf, init)) & seen[:, :, None]
    n_upd = updated.sum((1, 2))
    tot_entries = n_states * A
    frac_init = 1 - n_upd / np.maximum(tot_entries, 1)
    revisited = (V > 1).sum(1)

    # per-gap change on states present in both snapshots
    dq, new_per_iter, upd_existing = [np.nan], [np.nan], [np.nan]
    for t in range(1, T):
        both = seen[t] & seen[t - 1]
        gap = max(its[t] - its[t - 1], 1)
        d = np.abs(Qf[t][both] - Qf[t - 1][both])
        dq.append(d.mean() / gap if d.size else 0)
        new_per_iter.append((n_states[t] - n_states[t - 1]) / gap)
        # entries that were already updated before and changed again (true refinement)
        prev_upd = updated[t - 1][both]
        changed = ~np.isclose(Qf[t][both], Qf[t - 1][both])
        upd_existing.append((prev_upd & changed).sum())

    greedy = greedy_actions(Qf, C, seen, modes)
    tied = np.isclose(Qf.max(2), Qf.min(2))  # all actions equal -> greedy is just a tie-break
    share = np.zeros((T, A))
    for t in range(T):
        m = seen[t] & ~tied[t]
        for a in range(A):
            share[t, a] = (greedy[t][m] == a).mean() if m.any() else 0
    meanq = np.array([[np.nanmean(Q[t, seen[t], a]) for a in range(A)] for t in range(T)])

    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    x = np.arange(T)  # evenly spaced so the 6 -> 101 gap does not squash early checkpoints
    lab = [str(i) for i in its]

    def style(a, title, ylabel):
        a.set_title(title); a.set_ylabel(ylabel); a.set_xticks(x); a.set_xticklabels(lab, rotation=60, fontsize=8)
        a.set_xlabel("checkpoint (MATSim iteration)"); a.grid(alpha=.3)

    ax[0, 0].plot(x, n_states, "o-", label="distinct states")
    ax[0, 0].plot(x, revisited, "s-", label="states visited > 1x")
    style(ax[0, 0], "State-space coverage", "count"); ax[0, 0].legend()

    ax[0, 1].plot(x, frac_init * 100, "o-", color="#7f7f7f")
    style(ax[0, 1], "Q-entries still at initial value", "% of seen (state, action)")

    ax[0, 2].bar(x - .2, new_per_iter, .4, label="new states / iteration", color="#1f77b4")
    ax[0, 2].bar(x + .2, dq, .4, label="mean |dQ| / iteration (existing states)", color="#d62728")
    style(ax[0, 2], "Novelty vs. refinement (per gap)", "value"); ax[0, 2].legend(fontsize=8)

    for a, m in enumerate(modes):
        c = ACTION_COLORS.get(m, FALLBACK[a % 4])
        ax[1, 0].plot(x, share[:, a] * 100, "o-", color=c, label=m)
        ax[1, 1].plot(x, meanq[:, a], "o-", color=c, label=m)
    style(ax[1, 0], "Greedy action share (excl. all-tied states)", "% of states"); ax[1, 0].legend()
    style(ax[1, 1], "Mean Q per action (seen states)", "Q"); ax[1, 1].axhline(init, color="k", ls=":", lw=1)
    ax[1, 1].legend()

    # heat map of the final table, states in first-seen order
    last = Qf[-1][seen[-1]]
    im = ax[1, 2].imshow(last, aspect="auto", cmap="RdYlGn", vmin=np.nanpercentile(last, 2), vmax=max(init, last.max()),
                         interpolation="nearest")
    ax[1, 2].set_xticks(range(A)); ax[1, 2].set_xticklabels(modes)
    ax[1, 2].set_ylabel("state (first-seen order)"); ax[1, 2].set_title(f"Final Q-table (it {its[-1]})")
    fig.colorbar(im, ax=ax[1, 2], fraction=.046)
    fig.suptitle("Q-table evolution across checkpoints", fontsize=14)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print("saved", path)
    print(f"{'it':>6} {'states':>7} {'revisited':>9} {'%init':>7} {'dQ/it':>8}")
    for t in range(T):
        print(f"{its[t]:>6} {n_states[t]:>7} {revisited[t]:>9} {frac_init[t]*100:>7.1f} {dq[t]:>8.3f}")


# ----------------------------------------------------------------------------- 3D animation
def animate(cps, modes, Q, V, C, seen, xy, init, path, interp, fps, mp4):
    its = np.array([c["iteration"] for c in cps])
    T, S, A = Q.shape
    Qf = np.where(np.isnan(Q), init, Q)
    H = np.where(seen, Qf.max(2), np.nan)                       # height per (checkpoint, state)
    G = greedy_actions(Qf, C, seen, modes)                      # greedy action (agent's tie-break rule)
    UPD = (~np.isclose(Qf, init)).any(2) & seen                 # has at least one learned value

    # frames: interpolate heights/visits between checkpoints; states appear at their first checkpoint
    frames = []
    for t in range(T - 1):
        for k in range(interp):
            w = k / interp
            frames.append((t, t + 1, w))
    frames.append((T - 1, T - 1, 0.0))

    zmin = np.nanmin(H) - 30; zmax = max(init, np.nanmax(H)) + 30
    fig = plt.figure(figsize=(11, 8)); ax = fig.add_subplot(111, projection="3d")
    cols = np.array([ACTION_COLORS.get(m, FALLBACK[i % 4]) for i, m in enumerate(modes)])
    ax.set_xlabel("PCA-1 of state bits"); ax.set_ylabel("PCA-2"); ax.set_zlabel("max Q")
    ax.set_zlim(zmin, zmax)
    ax.set_xlim(xy[:, 0].min(), xy[:, 0].max()); ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
    handles = [Line2D([], [], marker="o", ls="", color=cols[i], label=f"greedy = {m}") for i, m in enumerate(modes)]
    handles.append(Line2D([], [], marker="o", ls="", color="#bbbbbb", label="never updated (all Q at init)"))
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    title = ax.set_title("")
    sc = [None]

    def draw(i):
        a, b, w = frames[i]
        if sc[0] is not None:
            sc[0].remove()
        # a state exists if seen at checkpoint a (or b once w crosses half-way for brand-new states)
        exists = seen[a] | (seen[b] & (w >= 0.5))
        ha = np.where(seen[a], H[a], H[b]); hb = np.where(seen[b], H[b], H[a])
        h = (1 - w) * ha + w * hb
        v = (1 - w) * V[a] + w * V[b]
        learned = UPD[b] if w >= 0.5 else UPD[a]
        g = G[b] if w >= 0.5 else G[a]
        colors = np.where(learned[:, None], matplotlib.colors.to_rgba_array(cols[g]),
                          matplotlib.colors.to_rgba_array(["#bbbbbb"] * S))
        m = exists & ~np.isnan(h)
        sc[0] = ax.scatter(xy[m, 0], xy[m, 1], h[m], c=colors[m], s=18 + 18 * np.sqrt(np.maximum(v[m], 0)),
                           depthshade=True, edgecolors="k", linewidths=.3)
        # stems to the floor make the height changes easier to read
        it = (1 - w) * its[a] + w * its[b]
        title.set_text(f"iteration ~{it:.0f}   |   states: {m.sum()}   |   learned states: {(learned & exists).sum()}")
        ax.view_init(elev=26, azim=35 + 0.6 * i)
        return sc[0],

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=1000 / fps, blit=False)
    if mp4:
        path = path.replace(".gif", ".mp4"); anim.save(path, writer=FFMpegWriter(fps=fps))
    else:
        anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig); print("saved", path)


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder", default=".", help="folder containing the pickle checkpoints")
    p.add_argument("--pattern", default="q_table*.pkl")
    p.add_argument("--out", default="qtable_viz")
    p.add_argument("--agent", default=None, help="agent tag inside the pickle (default: first one found)")
    p.add_argument("--init", type=float, default=600.0, help="initial Q value used in RLAgent.init_state")
    p.add_argument("--interp", type=int, default=6, help="interpolated frames between checkpoints")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--mp4", action="store_true", help="write mp4 instead of gif (needs ffmpeg)")
    p.add_argument("--no-anim", action="store_true", help="skip the 3D animation")
    a = p.parse_args()
    agent = int(a.agent) if a.agent and a.agent.isdigit() else a.agent

    os.makedirs(a.out, exist_ok=True)
    cps = load_checkpoints(a.folder, a.pattern, agent)
    states, modes, Q, V, C, seen = build_tensors(cps, a.init)
    print(f"{len(states)} unique states, modes = {modes}")
    dashboard(cps, modes, Q, V, C, seen, a.init, os.path.join(a.out, "qtable_dashboard.png"))
    if not a.no_anim:
        xy = pca_layout(states)
        animate(cps, modes, Q, V, C, seen, xy, a.init, os.path.join(a.out, "qtable_breathing.gif"), a.interp, a.fps, a.mp4)


if __name__ == "__main__":
    main()
