#!/usr/bin/env python3
"""
real_data.py  --  optional government-data layer for generate_synthetic_siouxfalls.py
=====================================================================================

Everything here is plain numpy/scipy/matplotlib (no geopandas needed). Inputs are files
you download once from the public sources listed in REFERENCES.md:

* LEHD LODES 8 (US Census Bureau)   <st>_wac_S000_JT00_<year>.csv.gz   jobs by WORK block
                                    <st>_rac_S000_JT00_<year>.csv.gz   workers by HOME block
                                    <st>_od_main_JT00_<year>.csv.gz    home-block -> work-block flows
                                    <st>_xwalk.csv.gz                  block -> lat/lon of block internal point
* Census TIGER/Line TAZ (2010) or the Sioux Falls MPO TAZ layer, converted to GeoJSON
  (ogr2ogr -f GeoJSON taz.geojson tl_2010_46_taz10.shp  -- or export from QGIS)

All coordinates are returned in UTM zone 14N (EPSG:32614), the CRS of your network/facilities.
"""
import csv
import gzip
import json
import math

import numpy as np

# --------------------------------------------------------------------------- projection
_A, _F = 6378137.0, 1 / 298.257223563           # WGS84 (NAD83 differs by < 2 m here)
_K0 = 0.9996


def lonlat_to_utm(lon, lat, zone=14):
    """Standard transverse-Mercator series (Snyder), vectorised. Returns (x, y)."""
    lon, lat = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    lon0 = math.radians(-183 + 6 * zone)
    e2 = _F * (2 - _F)
    ep2 = e2 / (1 - e2)
    n = _A / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    t, c, a = np.tan(lat) ** 2, ep2 * np.cos(lat) ** 2, (lon - lon0) * np.cos(lat)
    m = _A * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * lat
              - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * np.sin(2 * lat)
              + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * np.sin(4 * lat)
              - (35 * e2 ** 3 / 3072) * np.sin(6 * lat))
    x = _K0 * n * (a + (1 - t + c) * a ** 3 / 6 + (5 - 18 * t + t ** 2 + 72 * c - 58 * ep2) * a ** 5 / 120) + 500000.0
    y = _K0 * (m + n * np.tan(lat) * (a ** 2 / 2 + (5 - t + 9 * c + 4 * c ** 2) * a ** 4 / 24
                                       + (61 - 58 * t + t ** 2 + 600 * c - 330 * ep2) * a ** 6 / 720))
    return x, y


def _open(path):
    return gzip.open(path, "rt", encoding="utf-8", newline="") if str(path).endswith(".gz") else open(path, encoding="utf-8", newline="")


# --------------------------------------------------------------------------- LODES
def load_block_points(xwalk_path, bbox):
    """block geocode -> (x, y) UTM14N of the block internal point, restricted to bbox=(x0,y0,x1,y1)."""
    geo, xs, ys = [], [], []
    with _open(xwalk_path) as f:
        rd = csv.DictReader(f)
        key = next(k for k in rd.fieldnames if k.startswith("tabblk"))
        for r in rd:
            try:
                lon, lat = float(r["blklondd"]), float(r["blklatdd"])
            except (ValueError, KeyError):
                continue
            if not (-98 < lon < -95 and 42 < lat < 45):      # cheap pre-filter around SD/Sioux Falls
                continue
            geo.append(r[key]); xs.append(lon); ys.append(lat)
    x, y = lonlat_to_utm(xs, ys)
    keep = (x >= bbox[0]) & (x <= bbox[2]) & (y >= bbox[1]) & (y <= bbox[3])
    return {g: (float(a), float(b)) for g, a, b, k in zip(geo, x, y, keep) if k}


def load_lodes_area(path, block_xy, geocode_col):
    """WAC (geocode_col='w_geocode') or RAC ('h_geocode') -> xy array, jobs array, plus NAICS-free totals."""
    xy, val = [], []
    with _open(path) as f:
        for r in csv.DictReader(f):
            g = r[geocode_col]
            if g in block_xy and float(r["C000"]) > 0:
                xy.append(block_xy[g]); val.append(float(r["C000"]))
    return np.array(xy), np.array(val)


def lodes_od_stats(od_path, block_xy):
    """Job-weighted mean / median straight-line commute (m) for flows whose two ends are inside the study area,
    and the share of all home-in-area flows that stay in the area."""
    d, w, in_home, both = [], [], 0.0, 0.0
    with _open(od_path) as f:
        for r in csv.DictReader(f):
            h, wk, s = r["h_geocode"], r["w_geocode"], float(r["S000"])
            if h in block_xy:
                in_home += s
                if wk in block_xy:
                    both += s
                    d.append(math.dist(block_xy[h], block_xy[wk])); w.append(s)
    d, w = np.array(d), np.array(w)
    order = np.argsort(d)
    cw = np.cumsum(w[order])
    return dict(mean_m=float((d * w).sum() / w.sum()), median_m=float(d[order][np.searchsorted(cw, cw[-1] / 2)]),
                share_both_in_area=float(both / in_home), n_flows=int(len(d)))


def allocate_to_blocks(fac_xy, fac_raw_weight, block_xy, block_val):
    """Distribute each block's total (jobs / workers) over the facilities that are closest to it.

    * every facility belongs to its nearest block (with value > 0)
    * inside a block the total is split in proportion to ``fac_raw_weight``
    * a block with no facility of its own gives its whole total to the nearest facility
    Returns (allocated_value_per_facility, share_of_total_that_needed_the_fallback)."""
    from scipy.spatial import cKDTree
    bt = cKDTree(block_xy)
    _, owner = bt.query(fac_xy)
    out = np.zeros(len(fac_xy))
    wsum = np.bincount(owner, weights=fac_raw_weight, minlength=len(block_xy))
    ok = wsum[owner] > 0
    out[ok] = block_val[owner[ok]] * fac_raw_weight[ok] / wsum[owner[ok]]
    orphan = np.where(np.bincount(owner, minlength=len(block_xy)) == 0)[0]
    lost = 0.0
    if len(orphan):
        _, near = cKDTree(fac_xy).query(block_xy[orphan])
        np.add.at(out, near, block_val[orphan])
        lost = block_val[orphan].sum() / block_val.sum()
    return out, float(lost)


# --------------------------------------------------------------------------- LeBlanc OD table (TNTP)
def load_tntp_trips(path):
    """Parse SiouxFalls_trips.tntp (LeBlanc et al. 1975, as distributed by Transportation Networks for Research).
    Format:  'Origin 3' followed by 'dest : flow;' pairs. Returns {origin: {dest: flow}} (zone ids as int)."""
    import re
    txt = open(path, encoding="utf-8", errors="replace").read()
    out = {}
    for block in txt.split("Origin")[1:]:
        head, _, body = block.partition("\n")
        o = int(head.split()[0])
        out[o] = {int(d): float(v) for d, v in re.findall(r"(\d+)\s*:\s*([0-9.eE+-]+)", body)}
    return out


# --------------------------------------------------------------------------- TAZ polygons
def load_taz_polygons(geojson_path, id_field=None):
    """GeoJSON (WGS84/NAD83 lon-lat, or already UTM14N) -> list of (taz_id, [ring arrays in UTM14N])."""
    with open(geojson_path, encoding="utf-8") as f:
        gj = json.load(f)
    feats = gj["features"]
    cand = [id_field] if id_field else ["TAZCE10", "TAZ", "TAZ_ID", "TAZ_NUM", "ZONE", "id", "ID"]
    props0 = feats[0]["properties"]
    field = next((c for c in cand if c in props0), None)
    if field is None:
        raise SystemExit(f"TAZ id field not found. Properties are {list(props0)}; pass --taz-id-field.")
    out = []
    for ft in feats:
        geom = ft["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        rings = []
        for poly in polys:
            ring = np.array(poly[0], float)[:, :2]             # outer ring; holes are ignored
            if np.abs(ring).max() <= 180:                      # lon/lat -> UTM
                ring = np.column_stack(lonlat_to_utm(ring[:, 0], ring[:, 1]))
            rings.append(ring)
        out.append((ft["properties"][field], rings))
    return out


def assign_taz(xy, polys):
    """TAZ id per point (None if outside every polygon)."""
    from matplotlib.path import Path
    res = np.full(len(xy), None, dtype=object)
    for tid, rings in polys:
        for ring in rings:
            todo = np.where([r is None for r in res])[0]
            if not len(todo):
                return res
            inside = Path(ring).contains_points(xy[todo])
            res[todo[inside]] = tid
    return res
