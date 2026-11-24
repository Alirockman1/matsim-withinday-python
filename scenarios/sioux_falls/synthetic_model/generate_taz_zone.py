import json, pandas as pd, geopandas as gpd
from shapely.geometry import MultiPoint, box

f = pd.read_csv("scenarios\\sioux_falls\\synthetic_model\\output\\facility_links.csv").drop_duplicates(["x", "y"])
pts = gpd.GeoDataFrame(f, geometry=gpd.points_from_xy(f.x, f.y), crs="EPSG:26914")

# network bbox from generation_meta.json (so every network location gets a zone)
x0, y0, x1, y1 = json.load(open("scenarios\\sioux_falls\\synthetic_model\\output\\generation_meta.json"))["network"]["bbox"]
extent = box(x0, y0, x1, y1)

vor = gpd.GeoSeries(
    list(__import__("shapely").voronoi_polygons(MultiPoint(list(pts.geometry)), extend_to=extent).geoms),
    crs=pts.crs)
cells = gpd.sjoin(gpd.GeoDataFrame(geometry=vor), pts[["taz", "geometry"]], predicate="contains")

zones = cells.dissolve(by="taz").reset_index()
zones["geometry"] = zones.geometry.intersection(extent)
zones["TAZ_ID"] = "ZONE_" + zones.taz.astype(str)       # or just taz.astype(str)
zones[["TAZ_ID", "geometry"]].to_file("scenarios\\sioux_falls\\synthetic_model\\output\\sioux_falls_taz.shp")

# sanity check: every facility must fall inside its own zone
chk = gpd.sjoin(pts, zones[["taz", "geometry"]].rename(columns={"taz": "taz_poly"}), predicate="within")
print("mismatches:", (chk.taz != chk.taz_poly).sum(), "of", len(pts))