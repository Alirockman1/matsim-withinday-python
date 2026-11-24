#!/usr/bin/env python3
"""
generate_synthetic_siouxfalls.py
================================

Builds a synthetic MATSim population and a cleaned/enriched facilities file for the
Sioux Falls, SD scenario (UTM 14N, 24 TAZs).

What is KEPT from the uploaded data
    * The geography: every facility id, coordinate and TAZ suffix (``<id>_<taz>``).
    * Residential capacities (used as the relative residential density).
    * The activity types ``home`` / ``work`` / ``secondary`` (your scoring config only
      knows these, so no new activity types are introduced).
    * Link ids: each facility is snapped to the nearest car link of ``network.xml``
      (v2). Without --network the old fallback (link recovery from the original
      population) is used.

What is REGENERATED
    * Persons: age, sex, employment, licence, car_avail, bike_avail, household structure.
    * Home location: sampled proportionally to residential capacity.
    * Work / secondary locations: gravity model (facility capacity x distance decay).
    * Daily plans: realistic trip chains, departure times, durations, initial modes.
    * Facilities: ``work`` / ``secondary`` get capacities (originally missing), empty
      facilities are dropped, opening times are made consistent with the plans.

v2 additions
    * --network            snap facilities to links of the real MATSim network
    * --chain-share        share of workers whose plan contains >= 1 non-work stop
                           (sex-differentiated, see CFG["chain_*"]); everyone else keeps H-W-H
    * --taz-geojson        re-assign every facility to a REAL TAZ polygon (Census TIGER / MPO)
    * --lodes-wac/-rac/-xwalk/-od   scale job and residential capacities to Census LEHD LODES
                           block counts and compare commute length with LODES OD (see real_data.py)

Every demographic target lives in ``CFG`` below. Entries tagged [ACS] come from the
US Census Bureau (ACS 2019-2023 / QuickFacts) for Sioux Falls city; entries tagged
[ASSUMPTION] are modelling choices that you should adjust if you have better data.

Usage
    python generate_synthetic_siouxfalls.py \
        --facilities facilities_xml.gz \
        --population population_project_specific.xml \
        --out-dir out --n-persons 8385 --seed 42
"""
import argparse
import collections
import gzip
import json
import math
import os
import xml.etree.ElementTree as ET
import re

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

try:
    import real_data as RD
except ImportError:          # optional government-data layer
    RD = None

# =============================================================================
# CONFIGURATION
# =============================================================================
CFG = {
    # ---- population size -----------------------------------------------------
    "n_persons": 8385,                 # same size as the uploaded population
    "seed": 42,

    # ---- age / sex (persons 16+; the model has no school activity) ----------
    # [ACS] Age shares of the TOTAL city population (QuickFacts: 24.9% under 18,
    # 13.7% 65+, median age ~35). 16-17 is 2/13 of the 5-17 group.
    "age_bands": [(16, 18), (18, 25), (25, 35), (35, 45), (45, 55), (55, 65), (65, 75), (75, 85), (85, 96)],
    "age_band_share": [2.75, 9.5, 15.5, 13.5, 11.0, 11.9, 8.0, 4.3, 1.4],
    # [ACS] Share female per band (calibrated so the whole city is ~49.5% female).
    "female_share": [0.485, 0.485, 0.480, 0.485, 0.500, 0.505, 0.525, 0.575, 0.655],

    # ---- households ----------------------------------------------------------
    # [ACS] 2.32 persons/household incl. children -> ~1.9 adults (16+) per household.
    "hh_adults_share": {1: 0.36, 2: 0.47, 3: 0.11, 4: 0.045, 5: 0.015},   # [ASSUMPTION]
    "persons_per_dwelling_slot": 2.3,  # [ACS] a building of capacity c hosts ~c/2.3 households

    # ---- employment -----------------------------------------------------------
    # [ACS] labour-force participation ~72.6%, unemployment 2.3% -> ~71% employed (16+).
    "employed_by_band": [0.45, 0.76, 0.88, 0.89, 0.88, 0.76, 0.27, 0.07, 0.02],   # [ASSUMPTION shape]
    "employed_sex_factor": {"m": 1.06, "f": 0.93},     # women ~47% of workforce [ACS]
    "part_time_by_band": [0.75, 0.40, 0.14, 0.12, 0.12, 0.20, 0.55, 0.75, 0.90],  # [ASSUMPTION]
    "part_time_sex_factor": {"m": 0.7, "f": 1.4},
    "wfh_by_band": [0.02, 0.03, 0.07, 0.08, 0.07, 0.07, 0.08, 0.08, 0.08],        # [ACS] ~6-7% work from home

    # ---- mobility tools ------------------------------------------------------
    "licence_by_band": [0.72, 0.90, 0.95, 0.96, 0.96, 0.95, 0.93, 0.88, 0.75],    # [ASSUMPTION]
    # household vehicles by number of adults: P(0), P(1), P(2), P(3+)  [ACS: ~5% zero-vehicle hh, ~2 veh/hh]
    "hh_vehicles": {1: [0.07, 0.50, 0.33, 0.10], 2: [0.02, 0.17, 0.50, 0.31], 3: [0.01, 0.06, 0.30, 0.63]},
    "bike_by_band": [0.85, 0.65, 0.58, 0.55, 0.50, 0.42, 0.30, 0.15, 0.05],       # [ASSUMPTION]
    "bike_sex_logit": {"m": 0.3, "f": -0.3},
    "bike_hh_sigma": 0.8,              # household random effect (bikes cluster in households)

    # ---- trip chains ---------------------------------------------------------
    # W=work, S=secondary. Workers: P(chain) is a single calibration dial (--chain-share);
    # the table below only gives the *conditional* split among chain types (normalised at run time).
    # [NPTS 1995, FHWA] share of workers stopping on the way HOME->WORK: men 19.8%, women 33%;
    #                   WORK->HOME: men 46.4%, women 61.2%  => P(>=1 stop, indep.) ~ 57% (m) / 74% (f), ratio 1.30.
    # [Brookings 2023, NHTS-based] share of household trips made on the way to/from work fell 50% -> 40%
    #                   (2019 -> 2022), so the default sits clearly below the 1995 bound.   [ASSUMPTION: 0.45]
    "chain_share_workers": 0.45,
    "chain_sex_ratio_f_over_m": 1.30,
    "worker_female_share": 0.47,       # [ACS] women ~47% of the workforce
    "worker_patterns": [(["W"], 0.62), (["W", "S"], 0.17), (["S", "W"], 0.06),
                        (["W", "S", "W"], 0.09), (["S", "W", "S"], 0.03), (["W", "S", "S"], 0.03)],
    "nonworker_patterns": [(["S"], 0.62), (["S", "S"], 0.28), (["S", "S", "S"], 0.10)],
    "nonworker_patterns_75plus": [(["S"], 0.80), (["S", "S"], 0.17), (["S", "S", "S"], 0.03)],

    # ---- destination choice --------------------------------------------------
    # P(dest) ~ capacity * exp(-beeline/tau). In an unbounded area the mean beeline
    # is 2*tau; here the study area truncates it. [ACS] mean commute = 17 min
    # => tau = 8.5 min * 40 km/h / 1.3 detour = 4.4 km.
    "tau_work_km": 4.4,
    "tau_secondary_km": 3.0,           # [ASSUMPTION] non-work trips are shorter
    "tau_lunch_km": 0.8,
    "tau_detour_km": 1.5,              # chained stops stay close to the path
    "tau_next_stop_km": 1.5,

    # ---- facility capacities -------------------------------------------------
    "jobs_per_resident_worker": 1.2,   # [ASSUMPTION] Sioux Falls is a regional job centre
    "job_size_sigma": 1.1,             # lognormal; heavy tail = few large employers
    "secondary_size_sigma": 1.2,
    "secondary_visits_per_adult": 1.0,
    "core_boost_work": 1.5,            # [ASSUMPTION] larger establishments near the commercial core
    "core_boost_secondary": 0.6,
    "core_scale_m": 2500.0,
    "adult_share_of_capacity": 0.78,   # [ACS] 16+ share of residents (used to scale capacities)

    # ---- links ---------------------------------------------------------------
    "link_max_dist_m": 500.0,          # facilities farther than this from any car link are not used (covers ~99% here; network is sparse)
    # ---- LeBlanc zone structure (--leblanc-trips) ---------------------------------
    "leblanc_intrazonal_share": 0.15,  # LeBlanc's table has an empty diagonal (=100 % interzonal). 0.15 keeps ~15 % of work trips inside the home zone, matching the 85 % interzonal share of the gravity version [ASSUMPTION; 0 = pure paper]
    # ---- mode-choice dials (multiply the weights of MODE_W) ------------------------
    "pt_scale": 1.0, "bike_scale": 1.0,
    "link_exclude_freespeed": 25.0,    # m/s: freeways (>= ~90 km/h) are never activity access links
    "link_min_length_m": 10.0,         # tiny connector links are skipped as access links

    # ---- travel-time model (only used to keep plan times feasible) ----------
    "detour": 1.3,
    "speed_kmh": {"car": 38.0, "bike": 14.0, "pt": 17.0, "pedestrian": 4.8},
    "pt_wait_s": 480.0,

    # ---- opening times written to the facility file --------------------------
    "open_work": (5 * 3600, 23 * 3600),
    "open_secondary": (6 * 3600, 23.5 * 3600),
}

# Initial plan mode choice (the RL agent overrides this for the agents it controls;
# for all other agents it is the starting point of MATSim's own replanning).
MODE_W = {
    # car available:   w_car, (w_ped, scale_km), (w_bike, scale_km), w_pt
    "car_yes": dict(car=1.0, ped=(0.9, 0.7), bike=(0.12, 3.5), pt=0.012),
    # no car access
    "car_no": dict(car=0.0, ped=(2.0, 1.5), bike=(1.0, 5.0), pt=0.8),
}

MODES = ["car", "bike", "pt", "pedestrian"]


# =============================================================================
# I/O helpers
# =============================================================================
def hms(sec):
    sec = int(round(sec))
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def parse_facilities(path):
    fac = {}
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for _, el in ET.iterparse(f, events=("end",)):
            if el.tag != "facility":
                continue
            acts = {}
            for a in el.findall("activity"):
                cap = a.find("capacity")
                acts[a.get("type")] = float(cap.get("value")) if cap is not None else None
            fid = el.get("id")
            fac[fid] = dict(id=fid, x=float(el.get("x")), y=float(el.get("y")),
                            acts=acts, taz=int(fid.split("_")[1]))
            el.clear()
    return fac


def parse_population(path):
    persons = []
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag != "person":
            continue
        attr = {a.get("name"): a.text for a in el.find("attributes")}
        seq = []
        for e in el.find("plan"):
            if e.tag == "activity":
                seq.append(dict(kind="A", type=e.get("type"), fac=e.get("facility"), link=e.get("link"),
                                x=float(e.get("x")), y=float(e.get("y")), et=e.get("end_time")))
            else:
                seq.append(dict(kind="L", mode=e.get("mode")))
        persons.append(dict(id=el.get("id"), attr=attr, seq=seq))
        el.clear()
    return persons


# =============================================================================
# Links (the network is not available, so links are recovered from the original plans)
# =============================================================================
def resolve_links(fac, original_pop, max_dist):
    """facility id -> link id.

    1. Exact: the facility appears in the original population -> use its link.
    2. Nearest known facility of the same kind within ``max_dist`` metres.
    3. Nearest known facility of any kind within ``max_dist`` metres.
    Facilities with no link source are left out (they are not used as destinations).
    """
    known = {}
    for p in original_pop:
        for a in p["seq"]:
            if a["kind"] == "A" and a["fac"] and a["fac"] not in known:
                known[a["fac"]] = a["link"]

    ids = list(fac)
    xy = np.array([[fac[k]["x"], fac[k]["y"]] for k in ids])
    known_idx = np.array([i for i, k in enumerate(ids) if k in known])
    tree_all = cKDTree(xy[known_idx])

    kind_trees = {}
    for kind in ("home", "work", "secondary"):
        idx = np.array([i for i in known_idx if kind in fac[ids[i]]["acts"]])
        if len(idx):
            kind_trees[kind] = (idx, cKDTree(xy[idx]))

    link, source = {}, collections.Counter()
    for i, k in enumerate(ids):
        if k in known:
            link[k] = known[k]
            source["exact"] += 1
            continue
        got = False
        for kind in ("home", "work", "secondary"):
            if kind in fac[k]["acts"] and kind in kind_trees:
                idx, tr = kind_trees[kind]
                d, j = tr.query(xy[i])
                if d <= max_dist:
                    link[k] = known[ids[idx[j]]]
                    source["nn_same_kind"] += 1
                    got = True
                    break
        if got:
            continue
        d, j = tree_all.query(xy[i])
        if d <= max_dist:
            link[k] = known[ids[known_idx[j]]]
            source["nn_any_kind"] += 1
        else:
            source["unresolved"] += 1
    return link, source


def load_network_links(path, cfg):
    """Parse a MATSim network_v1 XML; return sampled points along every usable car link."""
    nodes, links = {}, []
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag == "node":
            nodes[el.get("id")] = (float(el.get("x")), float(el.get("y")))
        elif el.tag == "link":
            links.append((el.get("id"), el.get("from"), el.get("to"), float(el.get("length")),
                          float(el.get("freespeed")), el.get("modes")))
        el.clear()
    pts, ids, skipped = [], [], collections.Counter()
    for lid, a, b, ln, fs, modes in links:
        if "car" not in modes.split(","):
            skipped["not_car"] += 1; continue
        if fs >= cfg["link_exclude_freespeed"]:
            skipped["freeway"] += 1; continue
        if ln < cfg["link_min_length_m"]:
            skipped["too_short"] += 1; continue
        (x0, y0), (x1, y1) = nodes[a], nodes[b]
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) // 15) + 1)
        for t in np.linspace(0.0, 1.0, n):
            pts.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0))); ids.append(lid)
    return np.array(pts), np.array(ids), dict(n_links=len(links), skipped=dict(skipped),
                                              bbox=[min(v[0] for v in nodes.values()), min(v[1] for v in nodes.values()),
                                                    max(v[0] for v in nodes.values()), max(v[1] for v in nodes.values())])


def snap_to_network(fac, net_pts, net_ids, max_dist):
    """facility id -> nearest car-link id (<= max_dist metres). Returns (link, dist, stats)."""
    ids = list(fac)
    xy = np.array([[fac[k]["x"], fac[k]["y"]] for k in ids])
    d, j = cKDTree(net_pts).query(xy)
    link = {k: str(net_ids[jj]) for k, dd, jj in zip(ids, d, j) if dd <= max_dist}
    dist = dict(zip(ids, d))
    return link, dist, dict(snapped=len(link), unresolved=len(ids) - len(link),
                            median_dist_m=float(np.median(d)), p95_dist_m=float(np.percentile(d, 95)))


# =============================================================================
# Facilities
# =============================================================================
def build_facilities(fac_raw, link, cfg, rng, real=None):
    """Drop empty facilities, add capacities to work/secondary, return sites arrays."""
    fac = {k: v for k, v in fac_raw.items() if v["acts"]}
    dropped_empty = len(fac_raw) - len(fac)

    # commercial core = densest cluster of work + secondary facilities (data-driven)
    pts = np.array([[v["x"], v["y"]] for v in fac.values() if "work" in v["acts"] or "secondary" in v["acts"]])
    x0, y0 = pts.min(0)
    cell = 250.0
    nx, ny = int((pts[:, 0].max() - x0) / cell) + 2, int((pts[:, 1].max() - y0) / cell) + 2
    grid = np.zeros((nx, ny))
    for x, y in pts:
        grid[int((x - x0) / cell), int((y - y0) / cell)] += 1
    grid = gaussian_filter(grid, sigma=3)
    gi, gj = np.unravel_index(np.argmax(grid), grid.shape)
    core = np.array([x0 + (gi + 0.5) * cell, y0 + (gj + 0.5) * cell])

    # residential totals -> scale job / visit capacities
    home_cap = sum(v["acts"]["home"] or 0 for v in fac.values() if "home" in v["acts"])
    adults = home_cap * cfg["adult_share_of_capacity"]
    emp_rate = 0.71 * (1 - 0.065)
    total_jobs = adults * emp_rate * cfg["jobs_per_resident_worker"]
    total_visits = adults * cfg["secondary_visits_per_adult"]

    real = real or {}
    real_notes = {}

    def sized(kind, sigma, boost, total, block=None):
        ids = [k for k, v in fac.items() if kind in v["acts"]]
        raw = rng.lognormal(0.0, sigma, size=len(ids))
        d = np.array([math.hypot(fac[k]["x"] - core[0], fac[k]["y"] - core[1]) for k in ids])
        w = raw * (1.0 + boost * np.exp(-d / cfg["core_scale_m"]))
        for n, k in enumerate(ids):  # home-based businesses are tiny; schools (edu) are large employers
            if "home" in fac[k]["acts"]:
                w[n] *= 0.3
            elif "edu" in fac[k]["acts"] and kind == "work":
                w[n] *= 3.0
        if block is not None:        # LEHD LODES: block totals are split over the block's facilities
            fxy = np.array([[fac[k]["x"], fac[k]["y"]] for k in ids])
            alloc, lost = RD.allocate_to_blocks(fxy, w, block[0], block[1])
            real_notes[kind] = dict(source="LODES", total=float(block[1].sum()), fallback_share=lost, n_blocks=len(block[1]))
            return dict(zip(ids, np.maximum(1, np.round(alloc)).astype(int)))
        cap = np.maximum(1, np.round(w / w.sum() * total)).astype(int)
        return dict(zip(ids, cap))

    if "wac" in real:
        total_jobs = float(real["wac"][1].sum())
    work_cap = sized("work", cfg["job_size_sigma"], cfg["core_boost_work"], total_jobs, real.get("wac"))
    sec_cap = sized("secondary", cfg["secondary_size_sigma"], cfg["core_boost_secondary"], total_visits)
    for k, c in work_cap.items():
        fac[k]["acts"]["work"] = float(c)
    for k, c in sec_cap.items():
        fac[k]["acts"]["secondary"] = float(c)
    if "rac" in real:      # residential density from LODES workers-by-home-block, total kept = original capacity
        hid = [k for k, v in fac.items() if "home" in v["acts"]]
        hxy = np.array([[fac[k]["x"], fac[k]["y"]] for k in hid])
        hw = np.array([fac[k]["acts"]["home"] or 1.0 for k in hid])
        alloc, lost = RD.allocate_to_blocks(hxy, hw, real["rac"][0], real["rac"][1])
        alloc = alloc / alloc.sum() * hw.sum()
        for k, c in zip(hid, alloc):
            fac[k]["acts"]["home"] = float(max(c, 0.25))
        real_notes["home"] = dict(source="LODES RAC", fallback_share=lost, n_blocks=len(real["rac"][1]))

    def sites(kind):
        ids = [k for k, v in fac.items() if kind in v["acts"] and k in link]
        return dict(ids=ids,
                    xy=np.array([[fac[k]["x"], fac[k]["y"]] for k in ids]),
                    cap=np.array([fac[k]["acts"][kind] for k in ids], dtype=float),
                    link=[link[k] for k in ids],
                    taz=np.array([fac[k]["taz"] for k in ids]))

    S = {"home": sites("home"), "work": sites("work"), "secondary": sites("secondary")}
    meta = dict(real_notes=real_notes, dropped_empty=dropped_empty, core=core.tolist(), total_jobs=float(total_jobs),
                total_visits=float(total_visits), adults_in_area=float(adults))
    return fac, S, meta


def write_facilities(path, fac, cfg, n_persons, meta):
    ow, osec = cfg["open_work"], cfg["open_secondary"]
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<!DOCTYPE facilities SYSTEM "http://www.matsim.org/files/dtd/facilities_v1.dtd">\n')
        f.write('<facilities name="SiouxFalls Facilities (synthetic capacities)">\n')
        f.write(f'<!-- Geography/ids from the original file. work+secondary capacities are generated and only meaningful '
                f'as RELATIVE weights (they sum to ~{meta["total_jobs"]:.0f} jobs, from the file\'s own residential capacity of '
                f'~{meta["adults_in_area"]:.0f} adults, not from the city total). Population = {n_persons} agents = '
                f'{meta.get("sample_rate", 0) * 100:.1f}% of the city residents aged 16+. -->\n\n')
        for fid, v in fac.items():
            f.write(f'\t<facility id="{fid}" x="{float(v["x"])!r}" y="{float(v["y"])!r}">\n')
            for kind in ("home", "work", "secondary", "edu"):
                if kind not in v["acts"]:
                    continue
                cap = v["acts"][kind]
                f.write(f'\t\t<activity type="{kind}">\n')
                if cap is not None:
                    f.write(f'\t\t\t<capacity value="{cap}" />\n')
                if kind == "home":
                    o = (0, 30 * 3600)
                elif kind == "work":
                    o = ow
                elif kind == "secondary":
                    o = osec
                else:
                    o = (7.5 * 3600, 16 * 3600)
                f.write(f'\t\t\t<opentime day="wkday" start_time="{hms(o[0])}" end_time="{hms(o[1])}" />\n')
                f.write('\t\t</activity>\n')
            f.write('\t</facility>\n\n')
        f.write('</facilities>\n')


# =============================================================================
# Households and persons
# =============================================================================
def band_index(age, bands):
    for i, (lo, hi) in enumerate(bands):
        if lo <= age < hi:
            return i
    return len(bands) - 1


def draw_age(b, bands, rng):
    lo, hi = bands[b]
    if b == len(bands) - 1:                      # 85+: geometric tail
        return int(min(hi - 1, lo + rng.geometric(0.30) - 1))
    return int(rng.integers(lo, hi))


def draw_household_members(n, cfg, rng, pmf_by_n1):
    """Return a list of (age, sex) for one household with n adults."""
    bands, fem = cfg["age_bands"], cfg["female_share"]
    share = np.array(cfg["age_band_share"], dtype=float)

    def sex_for(age):
        return "f" if rng.random() < fem[band_index(age, bands)] else "m"

    def marginal_age(prop=None):
        p = share if prop is None else share * prop
        return draw_age(int(rng.choice(len(bands), p=p / p.sum())), bands, rng)

    if n == 1:
        a = marginal_age(pmf_by_n1)
        return [(a, sex_for(a))]

    # couple: opposite sex 96%, male older by ~2.5 y
    couple_prop = np.array([0, 0.2, 1.0, 1.1, 1.1, 1.1, 1.2, 1.1, 0.7])
    if n == 2:
        u = rng.random()
        if u < 0.80:                              # couple
            a1 = marginal_age(couple_prop)
            a2 = int(np.clip(round(a1 + rng.normal(0, 3.5)), 16, 95))
            s1 = sex_for(a1)
            s2 = ("m" if s1 == "f" else "f") if rng.random() < 0.96 else s1
            return [(a1, s1), (a2, s2)]
        if u < 0.92:                              # parent + adult child
            c = int(np.clip(round(rng.normal(24, 5)), 16, 40))
            p = int(np.clip(round(c + rng.normal(27, 4.5)), 38, 95))
            return [(p, sex_for(p)), (c, sex_for(c))]
        a1 = int(np.clip(round(rng.normal(25, 4.5)), 18, 45))            # roommates
        a2 = int(np.clip(round(a1 + rng.normal(0, 3)), 18, 55))
        return [(a1, sex_for(a1)), (a2, sex_for(a2))]

    # 3+ adults: parents + (n-2) adult children (or roommates 15%)
    if rng.random() < 0.15:
        out = []
        base = int(np.clip(round(rng.normal(26, 5)), 18, 45))
        for _ in range(n):
            a = int(np.clip(round(base + rng.normal(0, 3.5)), 18, 60))
            out.append((a, sex_for(a)))
        return out
    a1 = marginal_age(np.array([0, 0, 0.3, 1.2, 1.6, 1.3, 0.5, 0.1, 0.0]))
    a2 = int(np.clip(round(a1 + rng.normal(0, 3.0)), 30, 95))
    s1 = sex_for(a1)
    out = [(a1, s1), (a2, "m" if s1 == "f" else "f")]
    for _ in range(n - 2):
        c = int(np.clip(round(min(a1, a2) - rng.normal(27, 5.5)), 16, 40))
        out.append((c, sex_for(c)))
    return out


def ipf_household_weights(hh_list, cfg, iters=60):
    """Rake household weights so (age band x sex) person margins and household-size
    margins match the targets."""
    bands = cfg["age_bands"]
    nb = len(bands)
    share = np.array(cfg["age_band_share"], dtype=float)
    fem = np.array(cfg["female_share"], dtype=float)
    tgt_cell = np.concatenate([share * (1 - fem), share * fem])          # m bands, then f bands
    tgt_cell /= tgt_cell.sum()
    sizes = sorted(cfg["hh_adults_share"])
    tgt_size = np.array([cfg["hh_adults_share"][s] for s in sizes])
    tgt_size /= tgt_size.sum()

    M = np.zeros((len(hh_list), 2 * nb))
    size_idx = np.zeros(len(hh_list), dtype=int)
    for h, members in enumerate(hh_list):
        for age, sex in members:
            M[h, band_index(age, bands) + (nb if sex == "f" else 0)] += 1
        size_idx[h] = min(len(members), sizes[-1]) - 1
    n_h = M.sum(1)
    w = np.ones(len(hh_list))
    for _ in range(iters):
        cur = (w[:, None] * M).sum(0)
        cur = np.maximum(cur / cur.sum(), 1e-12)
        ratio = tgt_cell / cur
        with np.errstate(divide="ignore"):
            w *= np.exp((M * np.log(ratio)[None, :]).sum(1) / n_h)
        w /= w.mean()
        cur_s = np.bincount(size_idx, weights=w, minlength=len(sizes))
        cur_s = np.maximum(cur_s / cur_s.sum(), 1e-12)
        w *= (tgt_size / cur_s)[size_idx]
        w /= w.mean()
    return w


def build_persons(n_target, cfg, rng):
    """Households -> persons with all attributes except locations and plans."""
    bands = cfg["age_bands"]
    sizes = sorted(cfg["hh_adults_share"])
    size_p = np.array([cfg["hh_adults_share"][s] for s in sizes], dtype=float)
    size_p /= size_p.sum()
    pmf1 = np.array([0.0, 0.8, 1.2, 0.9, 0.9, 1.0, 1.3, 1.8, 2.2])   # singles skew young / old

    pool, total = [], 0
    while total < 3 * n_target:
        n = int(rng.choice(sizes, p=size_p))
        pool.append(draw_household_members(n, cfg, rng, pmf1))
        total += n
    w = ipf_household_weights(pool, cfg)

    # weighted sampling without replacement (Efraimidis-Spirakis), fill to exactly n_target persons
    keys = rng.random(len(pool)) ** (1.0 / np.maximum(w, 1e-9))
    chosen, count = [], 0
    for h in np.argsort(-keys):
        if count + len(pool[h]) <= n_target:
            chosen.append(int(h))
            count += len(pool[h])
        if count == n_target:
            break

    persons, hh_rows = [], []
    for hh_no, h in enumerate(chosen, start=1):
        members = pool[h]
        n_ad = len(members)
        veh_key = min(n_ad, 3)
        veh = int(rng.choice(4, p=cfg["hh_vehicles"][veh_key]))
        bike_u = rng.normal(0, cfg["bike_hh_sigma"])
        hh_persons = []
        for k, (age, sex) in enumerate(members, start=1):
            b = band_index(age, bands)
            p_emp = min(0.97, cfg["employed_by_band"][b] * cfg["employed_sex_factor"][sex])
            employed = rng.random() < p_emp
            part_time = wfh = False
            if employed:
                p_pt = min(0.95, cfg["part_time_by_band"][b] * cfg["part_time_sex_factor"][sex])
                part_time = rng.random() < p_pt
                wfh = rng.random() < cfg["wfh_by_band"][b]
            licensed = rng.random() < cfg["licence_by_band"][b]
            pb = min(0.99, max(0.01, cfg["bike_by_band"][b]))
            logit = math.log(pb / (1 - pb)) + cfg["bike_sex_logit"][sex] + bike_u
            bike = rng.random() < 1.0 / (1.0 + math.exp(-logit))
            hh_persons.append(dict(id=f"{hh_no:05d}_{k}", hh=f"{hh_no:05d}", age=age, sex=sex, band=b,
                                   employed=employed, part_time=part_time, wfh=wfh,
                                   licensed=licensed, bike=bike, hh_vehicles=veh))
        # car access: licensed members get the household's vehicles; commuters first
        order = sorted(hh_persons, key=lambda p: (not p["licensed"],
                                                  not (p["employed"] and not p["wfh"]),
                                                  rng.random()))
        cars_left = veh
        for p in order:
            if p["licensed"] and cars_left > 0:
                p["car"] = True
                cars_left -= 1
            else:
                p["car"] = False
        persons.extend(hh_persons)
        hh_rows.append(hh_persons)
    return persons


# =============================================================================
# Destination choice
# =============================================================================
def pick(rng, w):
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        return int(rng.integers(len(w)))
    return int(np.searchsorted(np.cumsum(w), rng.random() * s))


def d_to(xy, p):
    return np.hypot(xy[:, 0] - p[0], xy[:, 1] - p[1])


def choose_gravity(rng, S, origin, tau_m, exclude=None):
    w = S["cap"] * np.exp(-d_to(S["xy"], origin) / tau_m)
    if exclude is not None:
        w[exclude] = 0.0
    return pick(rng, w)


def choose_detour(rng, S, a, b, tau_m, exclude=None):
    ab = math.hypot(a[0] - b[0], a[1] - b[1])
    d = d_to(S["xy"], a) + d_to(S["xy"], b) - ab
    w = S["cap"] * np.exp(-d / tau_m)
    if exclude is not None:
        w[exclude] = 0.0
    return pick(rng, w)


# =============================================================================
# Plans
# =============================================================================
def lognorm_min(rng, median_min, sigma, lo, hi):
    return float(np.clip(rng.lognormal(math.log(median_min), sigma), lo, hi)) * 60.0


def travel_time(cfg, dist_m, mode):
    t = dist_m * cfg["detour"] / (cfg["speed_kmh"][mode] / 3.6)
    return t + (cfg["pt_wait_s"] if mode == "pt" else 0.0)


MODE_SCALE = {"pt": 1.0, "bike": 1.0}


def choose_mode(rng, p, dist_m):
    d_km = dist_m / 1000.0
    m = MODE_W["car_yes" if p["car"] else "car_no"]
    w = np.zeros(4)
    w[0] = m["car"]
    w[3] = m["ped"][0] * math.exp(-d_km / m["ped"][1])
    w[1] = MODE_SCALE["bike"] * (m["bike"][0] * math.exp(-d_km / m["bike"][1]) if (p["bike"] and d_km > 0.4) else 0.0)
    w[2] = MODE_SCALE["pt"] * m["pt"] if d_km > 0.8 else 0.0
    if w.sum() <= 0:
        w[3] = 1.0
    return MODES[pick(rng, w)]


def choose_work(rng, cfg, S, hi, hxy, tau_m):
    """Work site. With a LeBlanc OD table: draw the destination ZONE from T[home zone][.] (as in Chakirov & Fourie 2014),
    then the site inside that zone by capacity x distance decay. Otherwise: plain gravity model."""
    W, probs = S["work"], cfg.get("od_probs")
    if not probs:
        return choose_gravity(rng, W, hxy, tau_m)
    hz = int(S["home"]["taz"][hi])
    pz = probs.get(hz)
    if pz is None:
        return choose_gravity(rng, W, hxy, tau_m)
    zones = [z for z in pz if z in W["zone_idx"]]
    z = zones[pick(rng, np.array([pz[z] for z in zones]))]
    idx = W["zone_idx"][z]
    sub = dict(cap=W["cap"][idx], xy=W["xy"][idx])
    return int(idx[choose_gravity(rng, sub, hxy, tau_m)])


def worker_pattern_table(cfg, sex):
    """Patterns for a worker: P(chain) = chain_share, split by sex so that the workforce average stays chain_share."""
    r, fw = cfg["chain_sex_ratio_f_over_m"], cfg["worker_female_share"]
    m_male = cfg["chain_share_workers"] / ((1 - fw) + fw * r)
    p_chain = min(0.95, m_male * (r if sex == "f" else 1.0))
    chains = [(pt, w) for pt, w in cfg["worker_patterns"] if pt != ["W"]]
    tot = sum(w for _, w in chains)
    return [(["W"], 1.0 - p_chain)] + [(pt, p_chain * w / tot) for pt, w in chains]


def draw_pattern(rng, patterns):
    return list(patterns[pick(rng, np.array([p for _, p in patterns]))][0])


def home_departure(rng, p, has_work):
    if not has_work:
        u = rng.random()
        t = rng.normal(10.0, 1.3) if u < 0.50 else (rng.normal(14.0, 1.8) if u < 0.88 else rng.normal(18.0, 1.2))
        return float(np.clip(t, 7.0, 20.5)) * 3600
    if p["part_time"]:
        u = rng.random()
        t = rng.normal(8.0, 1.2) if u < 0.40 else (rng.normal(12.5, 1.5) if u < 0.75 else rng.normal(16.0, 1.2))
        return float(np.clip(t, 5.5, 18.0)) * 3600
    u = rng.random()
    t = rng.normal(7.2, 0.7) if u < 0.78 else (rng.normal(8.6, 0.8) if u < 0.92 else rng.normal(5.9, 0.5))
    return float(np.clip(t, 5.0, 10.0)) * 3600


def work_duration(rng, p):
    if p["part_time"]:
        return float(np.clip(rng.lognormal(math.log(4.8), 0.3), 2.5, 7.0)) * 3600
    return float(np.clip(rng.normal(8.5, 0.9), 6.5, 11.0)) * 3600


def build_plan(rng, cfg, p, S):
    """Returns (list of stops incl. home at both ends, list of leg modes) or None."""
    H = S["home"]
    hi = p["home_idx"]
    hxy = H["xy"][hi]
    is_worker = p["employed"] and not p["wfh"]

    if is_worker:
        pattern = draw_pattern(rng, worker_pattern_table(cfg, p["sex"]))
    else:
        pats = cfg["nonworker_patterns_75plus"] if p["age"] >= 75 else cfg["nonworker_patterns"]
        pattern = draw_pattern(rng, pats)

    tauW, tauS = cfg["tau_work_km"] * 1000, cfg["tau_secondary_km"] * 1000
    tauD, tauN, tauL = cfg["tau_detour_km"] * 1000, cfg["tau_next_stop_km"] * 1000, cfg["tau_lunch_km"] * 1000
    W, SEC = S["work"], S["secondary"]

    # ---- locations ----
    work_i = choose_work(rng, cfg, S, hi, hxy, tauW) if "W" in pattern else None
    wxy = W["xy"][work_i] if work_i is not None else None
    stops = [dict(kind="home", site=hi)]
    used_sec = set()

    def add_sec(i):
        used_sec.add(i)
        stops.append(dict(kind="secondary", site=i))

    for pos, c in enumerate(pattern):
        if c == "W":
            stops.append(dict(kind="work", site=work_i))
            continue
        # secondary stop: choose by context
        prev = stops[-1]
        prev_xy = (W if prev["kind"] == "work" else SEC if prev["kind"] == "secondary" else H)["xy"][prev["site"]]
        excl = None
        if pattern == ["W", "S", "W"]:
            i = choose_gravity(rng, SEC, wxy, tauL)
        elif pattern == ["S"]:
            i = choose_gravity(rng, SEC, hxy, tauS)
        elif pos == 0 and "W" in pattern:                                    # S before W: between H and W
            i = choose_detour(rng, SEC, hxy, wxy, tauD)
        elif "W" in pattern and prev["kind"] == "work":                       # S after W: between W and H
            i = choose_detour(rng, SEC, wxy, hxy, tauD)
        elif pos == 0:                                                        # first stop of a non-work tour
            i = choose_gravity(rng, SEC, hxy, tauS)
        else:                                                                 # further stops: near previous, on way home
            e = np.zeros(len(SEC["cap"]), dtype=bool)
            e[list(used_sec)] = True
            i = choose_detour(rng, SEC, prev_xy, hxy, tauN, exclude=e)
        add_sec(i)
    stops.append(dict(kind="home", site=hi))

    def xy_of(s):
        return (H if s["kind"] == "home" else W if s["kind"] == "work" else SEC)["xy"][s["site"]]

    # ---- modes (tour-level; lunch sub-tour may walk) ----
    d_primary = math.hypot(*(xy_of(stops[1]) - hxy)) if work_i is None else math.hypot(*(wxy - hxy))
    tour_mode = choose_mode(rng, p, d_primary)
    modes = [tour_mode] * (len(stops) - 1)
    if pattern == ["W", "S", "W"]:
        dl = math.hypot(*(xy_of(stops[2]) - wxy))
        if dl < 1000 and rng.random() < 0.85:
            modes[1] = modes[2] = "pedestrian"

    # ---- activity durations ----
    inner = stops[1:-1]
    wd = work_duration(rng, p) if work_i is not None else 0.0
    if pattern == ["W", "S", "W"]:
        # morning part, lunch-time stop, afternoon part (work total stays ~wd)
        part1 = min(float(np.clip(rng.normal(4.0, 0.6), 2.0, 6.0)) * 3600, 0.7 * wd)
        durs = [part1, lognorm_min(rng, 40, 0.35, 20, 75), max(1.5 * 3600, wd - part1)]
    else:
        durs = []
        first_work = pattern.index("W") if "W" in pattern else None
        for k, c in enumerate(pattern):
            if c == "W":
                durs.append(wd)
            elif first_work is not None and k < first_work:       # stop before work (coffee, drop-off)
                durs.append(lognorm_min(rng, 15, 0.5, 5, 60))
            elif first_work is not None:                           # stop after work
                durs.append(lognorm_min(rng, 35, 0.6, 10, 150))
            else:                                                  # non-work tour
                durs.append(lognorm_min(rng, 60 if k == 0 else 35, 0.7, 15, 240))

    # ---- times ----
    t0 = home_departure(rng, p, work_i is not None)
    if pattern[0] == "S" and work_i is not None:
        t0 = max(t0, 6.5 * 3600)

    def timeline(t_start):
        t = t_start
        res = []
        for k, s in enumerate(inner):
            d = math.hypot(*(xy_of(s) - xy_of(stops[k])))
            arr = t + travel_time(cfg, d, modes[k])
            end = arr + durs[k]
            res.append((arr, end))
            t = end
        return res

    def window(s):
        return cfg["open_work"] if s["kind"] == "work" else cfg["open_secondary"]

    tl = timeline(t0)
    overshoot = max((end - window(s)[1] for s, (_, end) in zip(inner, tl)), default=0.0)
    undershoot = max((window(s)[0] - arr for s, (arr, _) in zip(inner, tl)), default=0.0)
    if overshoot > 0:                      # something closes before we are done: leave earlier
        t0 = max(4.5 * 3600, t0 - overshoot)
        tl = timeline(t0)
    elif undershoot > 0:                   # something is not open yet: leave later
        t0 += undershoot
        tl = timeline(t0)
    for s, (arr, end) in zip(inner, tl):   # still infeasible -> caller redraws the plan
        lo, hi_ = window(s)
        if arr < lo - 1 or end > hi_ + 1:
            return None

    return dict(stops=stops, modes=modes, t0=t0, ends=[e for _, e in tl], pattern=pattern)


# =============================================================================
# Writer
# =============================================================================
def write_population(path, persons, S):
    H, W, SEC = S["home"], S["work"], S["secondary"]
    sets = {"home": H, "work": W, "secondary": SEC}
    with open(path, "w", encoding="utf-8") as f:
        f.write("<?xml version='1.0' encoding='utf-8'?>\n")
        f.write('<!DOCTYPE population SYSTEM "http://www.matsim.org/files/dtd/population_v6.dtd">\n')
        f.write("<population>\n\n")
        for p in persons:
            plan = p["plan"]
            f.write(f'\t<person id="{p["id"]}">\n\t\t<attributes>\n')
            f.write(f'\t\t\t<attribute name="age" class="java.lang.Integer">{p["age"]}</attribute>\n')
            f.write(f'\t\t\t<attribute name="bike_avail" class="java.lang.String">{"always" if p["bike"] else "never"}</attribute>\n')
            f.write(f'\t\t\t<attribute name="car_avail" class="java.lang.String">{"always" if p["car"] else "never"}</attribute>\n')
            f.write(f'\t\t\t<attribute name="employed" class="java.lang.Boolean">{"true" if p["employed"] else "false"}</attribute>\n')
            f.write(f'\t\t\t<attribute name="sex" class="java.lang.String">{p["sex"]}</attribute>\n')
            f.write(f'\t\t\t<attribute name="household_id" class="java.lang.String">{p["hh"]}</attribute>\n')
            f.write(f'\t\t\t<attribute name="hh_vehicles" class="java.lang.Integer">{p["hh_vehicles"]}</attribute>\n')
            f.write(f'\t\t\t<attribute name="licensed" class="java.lang.Boolean">{"true" if p["licensed"] else "false"}</attribute>\n')
            f.write('\t\t</attributes>\n\t\t<plan>\n')
            stops, modes = plan["stops"], plan["modes"]
            ends = [plan["t0"]] + plan["ends"] + [None]
            for k, s in enumerate(stops):
                T = sets[s["kind"]]
                i = s["site"]
                x, y = T["xy"][i]
                end = f' end_time="{hms(ends[k])}"' if ends[k] is not None else ""
                f.write(f'\t\t\t<activity type="{s["kind"]}" link="{T["link"][i]}" facility="{T["ids"][i]}" '
                        f'x="{float(x)!r}" y="{float(y)!r}"{end}>\n\t\t\t</activity>\n')
                if k < len(modes):
                    f.write(f'\t\t\t<leg mode="{modes[k]}" />\n')
            f.write('\t\t</plan>\n\t</person>\n\n')
        f.write("</population>\n")


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--facilities", required=True)
    ap.add_argument("--population", help="original population (only for the old link fallback / snapping check)")
    ap.add_argument("--network", help="MATSim network.xml: facilities are snapped to the nearest car link")
    ap.add_argument("--chain-share", type=float, default=CFG["chain_share_workers"],
                    help="share of workers with >= 1 non-work stop (default %(default)s)")
    ap.add_argument("--taz-geojson", help="real TAZ polygons (Census TIGER 2010 TAZ / MPO), GeoJSON")
    ap.add_argument("--taz-id-field", help="property holding the TAZ id (auto-detected if omitted)")
    ap.add_argument("--rename-by-taz", action="store_true", help="rewrite facility ids as <n>_<real TAZ>")
    ap.add_argument("--lodes-xwalk", help="<st>_xwalk.csv.gz (block -> lat/lon); needed by the three options below")
    ap.add_argument("--lodes-wac", help="<st>_wac_S000_JT00_<year>.csv.gz -> job capacities of work facilities")
    ap.add_argument("--lodes-rac", help="<st>_rac_S000_JT00_<year>.csv.gz -> residential weights of home facilities")
    ap.add_argument("--lodes-od", help="<st>_od_main_JT00_<year>.csv.gz -> commute-length benchmark")
    ap.add_argument("--leblanc-trips", help="SiouxFalls_trips.tntp: work destination ZONE ~ LeBlanc OD table (keeps your 24 zones)")
    ap.add_argument("--intrazonal-share", type=float, default=CFG["leblanc_intrazonal_share"])
    ap.add_argument("--pt-scale", type=float, default=1.0, help="multiply PT weights in mode choice (1.0 = old behaviour)")
    ap.add_argument("--bike-scale", type=float, default=1.0)
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--n-persons", type=int, default=CFG["n_persons"])
    ap.add_argument("--sample-rate", type=float, help="e.g. 0.10 -> n_persons = rate x city population x 16+ share (overrides --n-persons)")
    ap.add_argument("--city-population", type=int, default=192517, help="residents of the study area [Census 2020, Sioux Falls city]")
    ap.add_argument("--seed", type=int, default=CFG["seed"])
    ap.add_argument("--tau-work-km", type=float, default=CFG["tau_work_km"])
    ap.add_argument("--tau-secondary-km", type=float, default=CFG["tau_secondary_km"])
    ap.add_argument("--no-report", action="store_true", help="skip the validation figure")
    args = ap.parse_args()

    cfg = dict(CFG)
    cfg["tau_work_km"], cfg["tau_secondary_km"] = args.tau_work_km, args.tau_secondary_km
    cfg["chain_share_workers"] = args.chain_share
    adults_full = args.city_population * cfg["adult_share_of_capacity"]
    if args.sample_rate:
        args.n_persons = int(round(args.sample_rate * adults_full))
        print(f"  sample rate {args.sample_rate:.0%} of {adults_full:,.0f} residents aged 16+ -> {args.n_persons} agents")
    MODE_SCALE["pt"], MODE_SCALE["bike"] = args.pt_scale, args.bike_scale
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    extra_meta = {}

    print("Loading inputs ...")
    fac_raw = parse_facilities(args.facilities)
    orig_pop = parse_population(args.population) if args.population else []

    # ---- network (links + study-area bbox) ----
    net_pts = net_ids = bbox = None
    if args.network:
        net_pts, net_ids, net_info = load_network_links(args.network, cfg)
        bbox = net_info["bbox"]
        extra_meta["network"] = net_info
        print(f"  network: {net_info['n_links']} links, usable access links after filters: {len(set(net_ids))}, skipped {net_info['skipped']}")

    # ---- real TAZ polygons ----
    if args.taz_geojson:
        polys = RD.load_taz_polygons(args.taz_geojson, args.taz_id_field)
        ids = list(fac_raw)
        tz = RD.assign_taz(np.array([[fac_raw[k]["x"], fac_raw[k]["y"]] for k in ids]), polys)
        outside = sum(t is None for t in tz)
        renamed = {}
        for k, t in zip(ids, tz):
            if t is not None:
                fac_raw[k]["taz_real"] = t
                if args.rename_by_taz:
                    renamed[k] = f"{k.split('_')[0]}_{t}"
        if renamed:
            fac_raw = {renamed.get(k, k): dict(v, id=renamed.get(k, k)) for k, v in fac_raw.items()}
        for v in fac_raw.values():
            if "taz_real" in v:
                v["taz"] = v["taz_real"]
        extra_meta["taz"] = dict(n_polygons=len(polys), facilities_outside_all_polygons=int(outside),
                                 n_taz_with_facilities=len({v["taz"] for v in fac_raw.values()}))
        print(f"  TAZ: {len(polys)} polygons, {outside} facilities outside all polygons keep their old zone")

    # ---- links ----
    if args.network:
        link, link_dist, link_src = snap_to_network(fac_raw, net_pts, net_ids, cfg["link_max_dist_m"])
        link_src = dict(link_src)
        if orig_pop:     # sanity check against the links that came with the original population
            known = {a["fac"]: a["link"] for p in orig_pop for a in p["seq"] if a["kind"] == "A" and a["fac"]}
            both = [k for k in known if k in link]
            link_src["agreement_with_original_links"] = round(sum(known[k] == link[k] for k in both) / max(1, len(both)), 3)
    else:
        if not orig_pop:
            raise SystemExit("need --network or --population to determine link ids")
        link, link_src = resolve_links(fac_raw, orig_pop, cfg["link_max_dist_m"])
    print("  link sources:", dict(link_src))

    # ---- LEHD LODES ----
    real, lodes_od = {}, None
    if args.lodes_wac or args.lodes_rac or args.lodes_od:
        if not (RD and args.lodes_xwalk and bbox):
            raise SystemExit("LODES options need real_data.py, --lodes-xwalk and --network (for the study-area bbox)")
        blocks = RD.load_block_points(args.lodes_xwalk, bbox)
        print(f"  LODES: {len(blocks)} census blocks inside the study area")
        if args.lodes_wac:
            real["wac"] = RD.load_lodes_area(args.lodes_wac, blocks, "w_geocode")
            print(f"    WAC: {real['wac'][1].sum():.0f} jobs in {len(real['wac'][1])} blocks")
        if args.lodes_rac:
            real["rac"] = RD.load_lodes_area(args.lodes_rac, blocks, "h_geocode")
            print(f"    RAC: {real['rac'][1].sum():.0f} resident workers in {len(real['rac'][1])} blocks")
        if args.lodes_od:
            lodes_od = RD.lodes_od_stats(args.lodes_od, blocks)
            extra_meta["lodes_od"] = lodes_od
            print(f"    OD: mean beeline commute {lodes_od['mean_m'] / 1000:.2f} km, median {lodes_od['median_m'] / 1000:.2f} km")

    fac, S, meta = build_facilities(fac_raw, link, cfg, rng, real)
    print(f"  facilities kept {len(fac)} (dropped {meta['dropped_empty']} empty) | usable homes {len(S['home']['ids'])}, "
          f"work {len(S['work']['ids'])}, secondary {len(S['secondary']['ids'])} | core at {np.round(meta['core'])}")

    W_ = S["work"]
    W_["zone_idx"] = {int(z): np.where(W_["taz"] == z)[0] for z in set(W_["taz"].tolist())}
    od_target = None
    if args.leblanc_trips:
        if args.taz_geojson:
            raise SystemExit("--leblanc-trips uses the 24 LeBlanc zones in the facility ids; do not combine with --taz-geojson")
        T = RD.load_tntp_trips(args.leblanc_trips)
        s_in = args.intrazonal_share
        cfg["od_probs"], od_target = {}, {}
        for o, row in T.items():
            row = {d: v for d, v in row.items() if v > 0 and d != o and d in W_["zone_idx"]}
            tot = sum(row.values())
            if tot <= 0:
                continue
            pz = {d: (1 - s_in) * v / tot for d, v in row.items()}
            if s_in > 0 and o in W_["zone_idx"]:
                pz[o] = pz.get(o, 0.0) + s_in
            cfg["od_probs"][o] = pz
            od_target[o] = pz
        print(f"  LeBlanc OD: {len(T)} origin zones, total flow {sum(sum(r.values()) for r in T.values()):.0f}; "
              f"work zones without work facilities: {sorted(set(range(1, 25)) - set(W_['zone_idx']))}")

    print("Building households and persons ...")
    persons = build_persons(args.n_persons, cfg, rng)

    # homes: dwelling slots proportional to capacity; a slot hosts one household
    H = S["home"]
    slots = np.maximum(1, np.round(H["cap"] / cfg["persons_per_dwelling_slot"])).astype(int)
    free = slots.copy()
    hh_home = {}
    for hh in dict.fromkeys(p["hh"] for p in persons):
        for _ in range(50):
            i = pick(rng, free.astype(float))
            if free[i] > 0:
                free[i] -= 1
                hh_home[hh] = i
                break
        else:
            hh_home[hh] = int(np.argmax(free))
    for p in persons:
        p["home_idx"] = hh_home[p["hh"]]

    print("Generating plans ...")
    keep, failed = [], 0
    for p in persons:
        plan = None
        for _ in range(5):
            plan = build_plan(rng, cfg, p, S)
            if plan is not None:
                break
        if plan is None:
            failed += 1
            continue
        p["plan"] = plan
        keep.append(p)
    print(f"  plans generated for {len(keep)} persons ({failed} failed after retries)")

    # ---- plan statistics (what share of plans is multi-activity, trip rates, commute length) ----
    nact = collections.Counter()
    legs_w, legs_n, beel = [], [], []
    for p in keep:
        kinds = [s_["kind"] for s_ in p["plan"]["stops"]]
        nact[len(kinds) - 2] += 1
        (legs_w if "work" in kinds else legs_n).append(len(kinds) - 1)
        if "work" in kinds:
            wi = kinds.index("work")
            hxy = S["home"]["xy"][p["plan"]["stops"][0]["site"]]
            wxy = S["work"]["xy"][p["plan"]["stops"][wi]["site"]]
            beel.append(math.dist(hxy, wxy))
    n = len(keep)
    mode_cnt = collections.Counter(m for p in keep for m in p["plan"]["modes"])
    tot_legs = sum(mode_cnt.values())
    zone_flow = collections.defaultdict(collections.Counter)
    for p in keep:
        kinds = [s_["kind"] for s_ in p["plan"]["stops"]]
        if "work" in kinds:
            hz = int(S["home"]["taz"][p["plan"]["stops"][0]["site"]])
            wz = int(S["work"]["taz"][p["plan"]["stops"][kinds.index("work")]["site"]])
            zone_flow[hz][wz] += 1
    if od_target:        # total-variation distance between realised and LeBlanc destination-zone shares (rows with >= 20 workers)
        tv = []
        for o, c in zone_flow.items():
            nrow = sum(c.values())
            if nrow >= 20 and o in od_target:
                zs = set(c) | set(od_target[o])
                tv.append((nrow, 0.5 * sum(abs(c.get(z, 0) / nrow - od_target[o].get(z, 0.0)) for z in zs)))
        extra_meta["leblanc_zone_tv_distance_weighted"] = round(sum(a * b for a, b in tv) / sum(a for a, _ in tv), 3)
    inter = collections.Counter()
    for p in keep:
        st = p["plan"]["stops"]
        for a, b in zip(st, st[1:]):
            za, zb = int(S[a["kind"]]["taz"][a["site"]]), int(S[b["kind"]]["taz"][b["site"]])
            key = "work" if "work" in (a["kind"], b["kind"]) else "secondary"
            inter[key + "_legs"] += 1; inter["all_legs"] += 1
            if za != zb:
                inter[key + "_inter"] += 1; inter["all_inter"] += 1
    interzonal = {k: round(inter[k + "_inter"] / max(1, inter[k + "_legs"]), 3) for k in ("all", "work", "secondary")}
    plan_stats = dict(
        interzonal_leg_share=interzonal,
        mode_share_legs={k: round(v / tot_legs, 3) for k, v in sorted(mode_cnt.items())},
        share_two_activity_plans=round(nact[1] / n, 3),
        share_multi_activity_plans=round(1 - nact[1] / n, 3),
        share_with_work_and_extra_stop=round(sum(1 for p in keep if len(p["plan"]["stops"]) > 3
                                                 and any(s_["kind"] == "work" for s_ in p["plan"]["stops"])) / n, 3),
        activities_per_plan_histogram={int(k): v for k, v in sorted(nact.items())},
        mean_trips_work_plans=round(float(np.mean(legs_w)), 2), mean_trips_nonwork_plans=round(float(np.mean(legs_n)), 2),
        mean_work_beeline_km=round(float(np.mean(beel)) / 1000, 2), median_work_beeline_km=round(float(np.median(beel)) / 1000, 2))
    extra_meta["plan_stats"] = plan_stats
    eff_rate = len(keep) / adults_full
    meta["sample_rate"] = eff_rate
    extra_meta["matsim_qsim"] = dict(flowCapacityFactor=round(eff_rate, 4), storageCapacityFactor=round(eff_rate, 4))
    with open(os.path.join(args.out_dir, "matsim_qsim_snippet.xml"), "w") as f:
        f.write(f'<module name="qsim">\n  <param name="flowCapacityFactor" value="{eff_rate:.4f}" />\n'
                f'  <param name="storageCapacityFactor" value="{eff_rate:.4f}" />\n</module>\n')
    print(f"  effective sample rate {eff_rate:.1%} of {adults_full:,.0f} residents 16+ -> set qsim flow/storageCapacityFactor to {eff_rate:.3f}")
    print("  plan statistics:", json.dumps(plan_stats))
    if lodes_od:
        print(f"  commute beeline: synthetic mean {plan_stats['mean_work_beeline_km']} km vs LODES {lodes_od['mean_m'] / 1000:.2f} km")

    fac_path = os.path.join(args.out_dir, "facilities_synthetic.xml.gz")
    pop_path = os.path.join(args.out_dir, "population_synthetic.xml")
    with open(os.path.join(args.out_dir, "facility_links.csv"), "w") as f:
        f.write("facility,x,y,taz,link\n")
        for k, v in fac.items():
            f.write(f"{k},{v['x']:.2f},{v['y']:.2f},{v['taz']},{link.get(k, '')}\n")
    write_facilities(fac_path, fac, cfg, len(keep), meta)
    write_population(pop_path, keep, S)
    print("Wrote", fac_path, "and", pop_path)

    with open(os.path.join(args.out_dir, "generation_meta.json"), "w") as f:
        json.dump(dict(link_sources=dict(link_src), **meta, **extra_meta, n_persons=len(keep), seed=args.seed), f, indent=2, default=str)

    if not args.no_report:
        try:
            from validation_report import make_report
            make_report(args.out_dir, keep, S, fac_raw, orig_pop, cfg)
        except ImportError:
            print("validation_report.py not found next to this script; skipping the figure.")


if __name__ == "__main__":
    main()
