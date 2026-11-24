#!/usr/bin/env python3
"""
3D bar GIF of a Q-table evolving over checkpoints.

  x axis = state (in the order each state was first visited)
  y axis = mode  (one colour per mode)
  bars   = Q(s, a), drawn hanging from the initial value (default 600):
           a bar that drops below the plane has been learned to be worse than the optimistic start

    python qtable_bars3d.py --folder /path/to/pickles --out viz
    options: --pattern "q_table*.pkl" --init 600 --interp 6 --fps 8 --agent 1001 --mp4
Needs qtable_evolution.py in the same folder (loading helpers), numpy, matplotlib, pillow.
"""
import argparse, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.lines import Line2D
import qtable_evolution as qe

COL = {"bike": "#2ca02c", "car": "#d62728", "pedestrian": "#ff7f0e", "pt": "#1f77b4"}
EXTRA = ["#9467bd", "#8c564b", "#e377c2", "#17becf"]
GAP = 1.7                                            # spacing between mode rows
ORDER = ["bike", "pt", "pedestrian", "car"]          # y-axis order (same as the agent's tie priority)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder", default="."); p.add_argument("--pattern", default="q_table*.pkl")
    p.add_argument("--out", default="qtable_viz"); p.add_argument("--agent", default=None)
    p.add_argument("--init", type=float, default=600.0); p.add_argument("--interp", type=int, default=6)
    p.add_argument("--fps", type=int, default=8); p.add_argument("--mp4", action="store_true")
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True)

    agent = int(a.agent) if a.agent and a.agent.isdigit() else a.agent
    cps = qe.load_checkpoints(a.folder, a.pattern, agent)
    states, modes, Q, V, C, seen = qe.build_tensors(cps, a.init)
    raw_modes = list(modes)
    modes = [m for m in ORDER if m in raw_modes] + [m for m in raw_modes if m not in ORDER]
    Q = Q[:, :, [raw_modes.index(m) for m in modes]]            # reorder mode axis to display order
    its = np.array([c["iteration"] for c in cps]); T, S, A = Q.shape
    Qf = np.where(np.isnan(Q), a.init, Q)
    cols = [COL.get(m, EXTRA[i % 4]) for i, m in enumerate(modes)]

    pairs = [(t, t + 1, k / a.interp) for t in range(T - 1) for k in range(a.interp)] + [(T - 1, T - 1, 0.0)]
    lo = min(a.init - 5, np.nanmin(Qf[seen]) - 10); hi = max(a.init + 5, np.nanmax(Qf[seen]) + 10)

    fig = plt.figure(figsize=(14, 7.5)); ax = fig.add_subplot(111, projection="3d")
    xs = np.repeat(np.arange(S), A).astype(float); ys = np.tile(np.arange(A), S).astype(float) * GAP
    mode_of = np.tile(np.arange(A), S); state_of = np.repeat(np.arange(S), A)
    barcols = np.array([cols[m] for m in mode_of])
    handles = [Line2D([], [], marker="s", ls="", color=c, ms=10, label=m) for m, c in zip(modes, cols)]
    title = fig.suptitle("", fontsize=14)
    art = [None]

    def draw(i):
        t0, t1, w = pairs[i]
        if art[0] is not None:
            art[0].remove()
        ex = seen[t0] | (seen[t1] & (w >= .5))
        q0 = np.where(seen[t0][:, None], Qf[t0], Qf[t1])
        q1 = np.where(seen[t1][:, None], Qf[t1], Qf[t0])
        q = (1 - w) * q0 + w * q1                                      # [S, A] interpolated Q
        flat = q.reshape(-1); keep = np.repeat(ex, A)
        dz = (flat - a.init)[keep]
        art[0] = ax.bar3d(xs[keep] - .4, ys[keep] - .4, np.full(keep.sum(), a.init), .8, .8, dz,
                          color=barcols[keep], alpha=.92, shade=True, linewidth=0)
        it = (1 - w) * its[t0] + w * its[t1]
        n_learned = int((np.abs(flat.reshape(S, A) - a.init) > 1e-9)[ex].sum())
        title.set_text(f"Q-table at iteration ~{it:.0f}    states: {int(ex.sum())}    learned (state, mode) cells: {n_learned}")
        ax.view_init(elev=36, azim=-68 + 14 * np.sin(2 * np.pi * i / len(pairs)))
        return art[0],

    # static decoration
    ax.set_xlim(0, S); ax.set_ylim(-.9, (A - 1) * GAP + .9); ax.set_zlim(lo, hi)
    ax.set_xlabel("state (order first visited)", labelpad=10); ax.set_zlabel("Q-value")
    ax.set_yticks([i * GAP for i in range(A)]); ax.set_yticklabels(modes)
    gx, gy = np.meshgrid([0, S], [-.9, (A - 1) * GAP + .9])
    ax.plot_surface(gx, gy, np.full_like(gx, a.init, float), color="k", alpha=.07, linewidth=0)
    ax.text(0, (A - 1) * GAP + 1.0, a.init + 6, f"initial value = {a.init:.0f}", fontsize=9)
    ax.legend(handles=handles, loc="upper left", title="mode", fontsize=9)
    ax.set_box_aspect((4.2, 1.5, 1.3)); fig.subplots_adjust(left=0, right=1, bottom=0, top=.94)

    anim = FuncAnimation(fig, draw, frames=len(pairs), interval=1000 / a.fps, blit=False)
    path = os.path.join(a.out, "qtable_bars3d." + ("mp4" if a.mp4 else "gif"))
    anim.save(path, writer=FFMpegWriter(fps=a.fps) if a.mp4 else PillowWriter(fps=a.fps), dpi=90)
    print("saved", path, f"({len(pairs)} frames, {S} states x {A} modes)")


if __name__ == "__main__":
    main()