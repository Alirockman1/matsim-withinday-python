import gzip
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box

def parse_facilities(file_path):
    print(f"Parsing facilities from {file_path}...")
    coords = []
    
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for event, elem in ET.iterparse(f, events=('end',)):
            if elem.tag.endswith('facility'):
                x = float(elem.get('x'))
                y = float(elem.get('y'))
                coords.append((x, y))
                elem.clear()
                
    return np.array(coords)

def voronoi_finite_polygons_2d(vor, radius=100000.0):
    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            t = vor.points[p2] - vor.points[p1] 
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            
            far_point = vor.vertices[v2] + direction * radius
            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        vs = np.array([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = [v for _, v in sorted(zip(angles, new_region))]
        new_regions.append(new_region)

    return new_regions, np.array(new_vertices)

def generate_and_visualize_zones(facilities_xml_path, output_shp_path, n_clusters=16):
    # 1. Extract coordinates
    coords = parse_facilities(facilities_xml_path)
    print(f"Extracted {len(coords)} facility coordinates.")

    # Define strict bounding box of the facilities with a small buffer (e.g., 1 km)
    min_x, min_y = coords.min(axis=0) - 1000
    max_x, max_y = coords.max(axis=0) + 1000
    network_box = box(min_x, min_y, max_x, max_y)

    # 2. Run K-Means Clustering (using your desired cluster count, e.g., 16)
    print(f"Running K-Means clustering with K={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(coords)
    centers = kmeans.cluster_centers_

    # 3. Generate Voronoi Polygons & Clip them to the network bounding box
    print("Generating and clipping Voronoi zones...")
    vor = Voronoi(centers)
    regions, vertices = voronoi_finite_polygons_2d(vor, radius=50000)

    polygons = []
    for region in regions:
        poly = Polygon(vertices[region])
        # Clip the polygon so it doesn't bleed out infinitely past the network bounds
        clipped_poly = poly.intersection(network_box)
        if not clipped_poly.is_empty:
            polygons.append(clipped_poly)

    # 4. Create GeoDataFrame and save
    df = pd.DataFrame({'TAZ_ID': [f"ZONE_{i}" for i in range(len(polygons))]})
    gdf = gpd.GeoDataFrame(df, geometry=polygons, crs="EPSG:32614")
    
    import os
    os.makedirs(os.path.dirname(output_shp_path), exist_ok=True)
    gdf.to_file(output_shp_path, driver='ESRI Shapefile')
    print(f"Successfully generated TAZ shapefile at: {output_shp_path}")

    # 5. Visualization Plot (zoomed tightly to the facilities)
    print("Generating map visualization...")
    fig, ax = plt.subplots(figsize=(10, 10))

    gdf.plot(ax=ax, column='TAZ_ID', cmap='tab20', edgecolor='black', alpha=0.5, linewidth=1.2)
    ax.scatter(coords[:, 0], coords[:, 1], color='red', s=3, alpha=0.6, label='Facilities')
    ax.scatter(centers[:, 0], centers[:, 1], color='blue', marker='x', s=120, linewidth=2, label='Zone Centers')

    # Force axis limits to match the tight facility bounds rather than infinite space
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    plt.title(f"Tightly Clipped TAZ Zones (K={n_clusters}) for Sioux Falls", fontsize=14)
    plt.xlabel("X Coordinate (Meters)", fontsize=12)
    plt.ylabel("Y Coordinate (Meters)", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig("taz_zones_visualization.png", dpi=300, bbox_inches='tight')
    print("Visualization saved as 'taz_zones_visualization.png'")
    plt.show()

if __name__ == "__main__":
    FACILITIES_PATH = "C:/ResearchWork/Matsim_integration/Local/matsim-withinday-python/scenarios/sioux-falls/input/facilities.xml.gz" 
    OUTPUT_SHP = "C:/ResearchWork/Matsim_integration/Local/matsim-withinday-python/scenarios/sioux-falls/input/sioux_falls_taz.shp"
    
    generate_and_visualize_zones(FACILITIES_PATH, OUTPUT_SHP, n_clusters=16)