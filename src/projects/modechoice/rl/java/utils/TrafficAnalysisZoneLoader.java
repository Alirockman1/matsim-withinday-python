package modechoice.rl.java.utils;

import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import org.matsim.api.core.v01.Coord;
import org.matsim.api.core.v01.Scenario;
import org.matsim.core.utils.gis.ShapeFileReader;
import org.matsim.facilities.ActivityFacility;
import org.matsim.withinday.utils.StateEngine;
import org.geotools.api.feature.Feature;
import org.geotools.api.feature.simple.SimpleFeature;
import org.locationtech.jts.geom.Coordinate;
import org.locationtech.jts.geom.Geometry;
import org.locationtech.jts.geom.GeometryFactory;
import org.locationtech.jts.geom.Point;
import org.locationtech.jts.index.strtree.STRtree;

import java.util.*;

public class TrafficAnalysisZoneLoader {
    private static final Logger log = LogManager.getLogger(TrafficAnalysisZoneLoader.class);
    private static final STRtree spatialIndex = new STRtree();
    private static final Map<Geometry, String> geometryToZoneIdMap = new HashMap<>();
    private static final Map<String, Integer> zoneIdToBinMap = new HashMap<>();
    private static final GeometryFactory geometryFactory = new GeometryFactory();

    private TrafficAnalysisZoneLoader() {
        throw new UnsupportedOperationException("Utility class cannot be instantiated");
    }

    /**
     * Professional helper container class for holding zone metadata during spatial sorting.
     */
    private static class ZoneData {
        private final String zoneId;
        private final Geometry geometry;
        private final double distanceFromCenter;

        public ZoneData(String zoneId, Geometry geometry, double distanceFromCenter) {
            this.zoneId = zoneId;
            this.geometry = geometry;
            this.distanceFromCenter = distanceFromCenter;
        }

        public String getZoneId() {
            return zoneId;
        }

        public Geometry getGeometry() {
            return geometry;
        }

        public double getDistanceFromCenter() {
            return distanceFromCenter;
        }
    }

/**
     * Loads TAZ geometries from a shapefile, calculates their distance from the network center,
     * and initializes the spatial binning index. If loading fails or the file path is invalid,
     * it automatically falls back to generating synthetic zones using facility K-means clustering.
     *
     * @param filePath             Path to the shapefile (.shp)
     * @param zoneIdAttributeName  Attribute name for the Zone ID within the shapefile
     * @param centerX              X-coordinate of the network/city center
     * @param centerY              Y-coordinate of the network/city center
     * @param scenario             MATSim scenario used for facility-based fallback clustering if needed
     */
    public static synchronized void loadZonesSortedByDistanceFromCenter(String filePath, 
        String zoneIdAttributeName, Scenario scenario) {
        
        double centerX = StateEngine.getNetworkCentroid().getX();
        double centerY = StateEngine.getNetworkCentroid().getY();

        try {
            ShapeFileReader shapeFileReader = new ShapeFileReader();
            shapeFileReader.readFileAndInitialize(filePath);
            Collection<SimpleFeature> features = shapeFileReader.getFeatureSet();

            if (features == null || features.isEmpty()) {
                throw new IllegalStateException("Shapefile feature set is empty or missing.");
            }

            List<ZoneData> zoneDataList = new ArrayList<>();
            Point networkCenter = geometryFactory.createPoint(new Coordinate(centerX, centerY));

            for (Feature feature : features) {
                Object zoneIdAttr = feature.getProperty(zoneIdAttributeName) != null
                        ? feature.getProperty(zoneIdAttributeName).getValue()
                        : null;

                if (zoneIdAttr == null) {
                    continue;
                }
                String zoneId = String.valueOf(zoneIdAttr);

                Object geometryObject = feature.getProperty("the_geom") != null
                        ? feature.getProperty("the_geom").getValue()
                        : null;

                if (!(geometryObject instanceof Geometry) && feature.getType().getGeometryDescriptor() != null) {
                    geometryObject = feature.getProperty(
                            feature.getType().getGeometryDescriptor().getName()
                    ).getValue();
                }

                if (!(geometryObject instanceof Geometry)) {
                    continue;
                }

                Geometry geometry = (Geometry) geometryObject;
                Point centroid = geometry.getCentroid();
                double distanceFromCenter = centroid.distance(networkCenter);

                zoneDataList.add(new ZoneData(zoneId, geometry, distanceFromCenter));
            }

            if (zoneDataList.isEmpty()) {
                throw new IllegalStateException("No valid geometries found in the shapefile.");
            }

            // Process and register the successfully loaded zones
            sortAndRegisterZones(zoneDataList);
            log.info("Successfully loaded and spatially sorted " + zoneDataList.size() + " TAZs from shapefile.");

        } catch (Exception e) {
            log.error("Warning: Failed to load TAZ shapefile (" + e.getMessage() + "). Falling back to K-means synthetic zone generation.");
            initializeSyntheticZonesFromFacilities(scenario, centerX, centerY);
        }
    }

    /**
     * Sorts a collection of zone data records by their distance to the network center 
     * and populates the internal spatial index and mapping dictionaries.
     *
     * @param zoneDataList List of raw zone data records to be sorted and registered
     */
    private static void sortAndRegisterZones(List<ZoneData> zoneDataList) {
        // Sort zones by distance from center: Closest gets Bin 0, furthest gets Bin N-1
        zoneDataList.sort(Comparator.comparingDouble(ZoneData::getDistanceFromCenter));

        int binIndex = 0;
        for (ZoneData zone : zoneDataList) {
            spatialIndex.insert(zone.getGeometry().getEnvelopeInternal(), zone.getGeometry());
            geometryToZoneIdMap.put(zone.getGeometry(), zone.getZoneId());
            zoneIdToBinMap.put(zone.getZoneId(), binIndex);
            binIndex++;
        }

        spatialIndex.build();
    }

    /**
     * Fallback mechanism: Extracts facility coordinates from the MATSim scenario, 
     * applies K-Means clustering to create 16 density-based synthetic zones, and sorts 
     * them by distance from the network center.
     *
     * @param scenario MATSim scenario containing activity facilities
     * @param centerX  X-coordinate of the network/city center
     * @param centerY  Y-coordinate of the network/city center
     */
    private static void initializeSyntheticZonesFromFacilities(Scenario scenario, double centerX, double centerY) {
        try {
            List<DoublePoint> points = new ArrayList<>();
            for (ActivityFacility facility : scenario.getActivityFacilities().getFacilities().values()) {
                double x = facility.getCoord().getX();
                double y = facility.getCoord().getY();
                points.add(new DoublePoint(new double[]{x, y}));
            }

            if (points.isEmpty()) {
                throw new IllegalStateException("No facilities available in scenario for synthetic clustering.");
            }

            int targetClusters = 16;
            KMeansPlusPlusClusterer<DoublePoint> clusterer = new KMeansPlusPlusClusterer<>(targetClusters, 50);
            List<CentroidCluster<DoublePoint>> clusters = clusterer.cluster(points);

            List<ZoneData> syntheticZoneList = new ArrayList<>();
            Point networkCenter = geometryFactory.createPoint(new Coordinate(centerX, centerY));

            int zoneCounter = 0;
            for (CentroidCluster<DoublePoint> cluster : clusters) {
                double[] centerCoord = cluster.getCenter().getPoint();
                Coordinate jtsCoord = new Coordinate(centerCoord[0], centerCoord[1]);
                Point clusterPoint = geometryFactory.createPoint(jtsCoord);

                double distanceFromCenter = clusterPoint.distance(networkCenter);
                String syntheticId = "SYNTHETIC_ZONE_" + (zoneCounter++);

                syntheticZoneList.add(new ZoneData(syntheticId, clusterPoint, distanceFromCenter));
            }

            // Reuse sorting and registration logic
            sortAndRegisterZones(syntheticZoneList);
            
            log.info("Successfully generated and sorted " + syntheticZoneList.size() + " synthetic TAZ bins using K-means clustering.");

        } catch (Exception e) {
            throw new RuntimeException("Critical Error: Failed to generate synthetic K-means TAZ fallback zones.", e);
        }
    }

    /**
     * Queries the spatial index and returns the Q-table bin index (0 to N-1) 
     * for the given raw coordinates.
     *
     * @param x X coordinate
     * @param y Y coordinate
     * @return  The integer bin index, or -1 if out of bounds
     */
    public static int getZoneBin(double x, double y) {
        Point queryPoint = geometryFactory.createPoint(new Coordinate(x, y));

        @SuppressWarnings("unchecked")
        List<Geometry> potentialMatches = spatialIndex.query(queryPoint.getEnvelopeInternal());

        for (Geometry geometry : potentialMatches) {
            if (geometry.covers(queryPoint)) {
                String zoneId = geometryToZoneIdMap.get(geometry);
                return zoneIdToBinMap.getOrDefault(zoneId, -1);
            }
        }

        return -1;
    }

    /**
     * Overloaded helper method to accept MATSim Coord objects directly.
     *
     * @param coord MATSim coordinate object
     * @return      The integer bin index, or -1 if null/out of bounds
     */
    public static int getZoneBin(Coord coord) {
        if (coord == null) return -1;
        return getZoneBin(coord.getX(), coord.getY());
    }

    /**
     * Returns the total number of loaded or synthetic zones available in the index.
     *
     * @return Total active zone count
     */
    public static int getTotalBins() {
        return zoneIdToBinMap.size();
    }
}
