"""validation_report.py -- optional figure + text summary written to <out_dir>/validation_report.png
Called by generate_synthetic_siouxfalls.py (make_report). Needs matplotlib."""
import collections
import math
import os

import numpy as np


def make_report(out_dir, persons, S, fac_raw, orig_pop, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    # departure time from home
    dep = [p["plan"]["t0"] / 3600 for p in persons]
    ax[0, 0].hist(dep, bins=np.arange(4, 22, 0.5), color="#4C78A8")
    ax[0, 0].set(title="First departure from home", xlabel="hour of day")
    # age
    ax[0, 1].hist([p["age"] for p in persons], bins=range(15, 100, 5), color="#59A14F")
    ax[0, 1].set(title="Age (16+)", xlabel="years")
    # activities per plan, synthetic vs original
    n_syn = collections.Counter(len(p["plan"]["stops"]) - 2 for p in persons)
    n_org = collections.Counter(sum(1 for e in p["seq"] if e["kind"] == "A") - 2 for p in orig_pop) if orig_pop else {}
    ks = sorted(set(n_syn) | set(n_org))
    w = 0.4
    ax[0, 2].bar(np.array(ks) - w / 2, [n_syn.get(k, 0) / len(persons) for k in ks], w, label="synthetic")
    if n_org:
        ax[0, 2].bar(np.array(ks) + w / 2, [n_org.get(k, 0) / len(orig_pop) for k in ks], w, label="original")
    ax[0, 2].set(title="Intermediate activities per plan", xlabel="number of non-home activities (1 = H-X-H)")
    ax[0, 2].legend()
    # mode shares
    modes = collections.Counter(m for p in persons for m in p["plan"]["modes"])
    ax[1, 0].bar(list(modes), [v / sum(modes.values()) for v in modes.values()], color="#F28E2B")
    ax[1, 0].set(title="Initial mode share (legs)")
    # commute beeline
    d = []
    for p in persons:
        k = [s["kind"] for s in p["plan"]["stops"]]
        if "work" in k:
            h = S["home"]["xy"][p["plan"]["stops"][0]["site"]]
            wk = S["work"]["xy"][p["plan"]["stops"][k.index("work")]["site"]]
            d.append(math.dist(h, wk) / 1000)
    ax[1, 1].hist(d, bins=40, color="#E15759")
    ax[1, 1].set(title=f"Home-work beeline (mean {np.mean(d):.1f} km)", xlabel="km")
    # homes / workplaces map
    ax[1, 2].scatter(*S["home"]["xy"].T, s=1, c="#BBBBBB", label="homes")
    ax[1, 2].scatter(*S["work"]["xy"].T, s=4, c="#E15759", label="work sites")
    ax[1, 2].set(title="Facility geography", aspect="equal")
    ax[1, 2].legend(markerscale=4)
    fig.tight_layout()
    path = os.path.join(out_dir, "validation_report.png")
    fig.savefig(path, dpi=110)
    print("Wrote", path)
