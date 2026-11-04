package berlin.java.prepare;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

import org.geotools.api.feature.simple.SimpleFeature;
import org.geotools.api.feature.simple.SimpleFeatureType;
import org.geotools.feature.simple.SimpleFeatureBuilder;
import org.geotools.feature.simple.SimpleFeatureTypeBuilder;
import org.locationtech.jts.geom.Coordinate;
import org.locationtech.jts.geom.Geometry;
import org.locationtech.jts.geom.MultiPolygon;
import org.locationtech.jts.geom.Polygon;
import org.locationtech.jts.geom.util.GeometryTransformer;
import org.locationtech.jts.geom.CoordinateSequence;
import org.locationtech.jts.geom.impl.CoordinateArraySequence;
import org.matsim.api.core.v01.Coord;
import org.matsim.core.utils.geometry.CoordinateTransformation;
import org.matsim.core.utils.geometry.geotools.MGC;
import org.matsim.core.utils.geometry.transformations.TransformationFactory;
import org.matsim.core.utils.gis.GeoFileReader;
import org.matsim.core.utils.gis.GeoFileWriter;

/**
 * <h2>PrepareBerlinTaz</h2>
 * <p>
 * One-shot preparation of the traffic analysis zones the reinforcement learning state space needs
 * for Open Berlin. {@code TrafficAnalysisZoneLoader} reads a shapefile with a {@code TAZ_ID}
 * attribute and queries it with scenario coordinates, so the zones have to carry that attribute
 * and be stored in the scenario's coordinate system.
 * </p>
 * <p>
 * The source is the district and federal state layer from public-svn: Berlin's twelve districts
 * plus the surrounding federal states, which together cover the whole Open Berlin network. Full
 * coverage matters, because a coordinate outside every zone yields bin {@code -1}, and
 * {@code StateEngine.convertToBitStateRepresentation} turns that into all-ones - the same bit
 * pattern as a real zone.
 * </p>
 * <p>
 * Usage (from the repository root):
 * </p>
 * <pre>
 * java -cp a7-rl-testbed-0.0.1-SNAPSHOT.jar berlin.java.prepare.PrepareBerlinTaz
 * </pre>
 */
public final class PrepareBerlinTaz {

    private static final String DEFAULT_SOURCE =
            "https://svn.vsp.tu-berlin.de/repos/public-svn/matsim/scenarios/countries/de/berlin/geo/Bezirke_Laender_shp.zip";
    private static final String DEFAULT_SHAPEFILE_IN_ZIP = "Bezirke_Laender_shp/Bezirke_Laender.shp";
    private static final String DEFAULT_OUTPUT = "scenarios/berlin-v7.1.1/input/berlin-taz.shp";

    /** The attribute the source carries for every feature, districts and federal states alike. */
    private static final String DEFAULT_SOURCE_ATTRIBUTE = "Name";

    /** The attribute name TrafficAnalysisZoneLoader looks for. */
    private static final String TARGET_ATTRIBUTE = "TAZ_ID";

    private static final String DEFAULT_SOURCE_CRS = TransformationFactory.WGS84;
    private static final String DEFAULT_TARGET_CRS = "EPSG:25832";

    private PrepareBerlinTaz() {
    }

    public static void main(String[] args) throws IOException {
        String source = argument(args, "--source", DEFAULT_SOURCE);
        String shapefileInZip = argument(args, "--shapefile-in-zip", DEFAULT_SHAPEFILE_IN_ZIP);
        Path output = Paths.get(argument(args, "--output", DEFAULT_OUTPUT));
        String sourceAttribute = argument(args, "--source-attribute", DEFAULT_SOURCE_ATTRIBUTE);
        String sourceCrs = argument(args, "--source-crs", DEFAULT_SOURCE_CRS);
        String targetCrs = argument(args, "--target-crs", DEFAULT_TARGET_CRS);

        Path workingDirectory = Files.createTempDirectory("berlin-taz");
        Path shapefile = source.endsWith(".zip")
                ? unpack(source, shapefileInZip, workingDirectory)
                : Paths.get(source);

        Collection<SimpleFeature> sourceFeatures = new GeoFileReader().readFileAndInitialize(shapefile.toString());
        System.out.println("Read " + sourceFeatures.size() + " features from " + shapefile);

        CoordinateTransformation transformation =
                TransformationFactory.getCoordinateTransformation(sourceCrs, targetCrs);
        List<SimpleFeature> zones = transform(sourceFeatures, sourceAttribute, transformation, targetCrs);

        Files.createDirectories(output.toAbsolutePath().getParent());
        GeoFileWriter.writeGeometries(zones, output.toString());

        System.out.println();
        System.out.println("Traffic analysis zones written to " + output.toAbsolutePath());
        System.out.println("  zones: " + zones.size() + ", attribute: " + TARGET_ATTRIBUTE
                + ", coordinate system: " + targetCrs);
        System.out.println("  bounds: " + describeBounds(zones));
        System.out.println();
        System.out.println("Set this as agentModeChoice.TAZFileName: " + output.getFileName());
    }

    /**
     * Extracts the shapefile and its sidecar files from a zipped archive.
     *
     * @param url              Archive URL or local path.
     * @param shapefileInZip   Path of the .shp entry inside the archive.
     * @param targetDirectory  Directory the entries are extracted into.
     * @return The extracted .shp file.
     */
    private static Path unpack(String url, String shapefileInZip, Path targetDirectory) throws IOException {
        String base = shapefileInZip.substring(0, shapefileInZip.length() - ".shp".length());

        System.out.println("Downloading " + url);
        try (ZipInputStream zip = new ZipInputStream(open(url))) {
            ZipEntry entry;
            while ((entry = zip.getNextEntry()) != null) {
                if (entry.isDirectory() || !entry.getName().startsWith(base + ".")) {
                    continue;
                }

                Path target = targetDirectory.resolve(Paths.get(entry.getName()).getFileName().toString());
                Files.copy(zip, target, StandardCopyOption.REPLACE_EXISTING);
                System.out.println("  extracted " + target.getFileName());
            }
        }

        Path shapefile = targetDirectory.resolve(Paths.get(shapefileInZip).getFileName().toString());
        if (!Files.exists(shapefile)) {
            throw new IOException("Archive " + url + " does not contain " + shapefileInZip);
        }
        return shapefile;
    }

    /**
     * Reprojects every feature and reduces its attributes to the single {@code TAZ_ID} the zone
     * loader reads.
     *
     * @param sourceFeatures  Features as read from the source file.
     * @param sourceAttribute Attribute holding the zone name in the source.
     * @param transformation  Transformation from the source into the scenario coordinate system.
     * @param targetCrs       Coordinate system the result is declared in.
     * @return The zone features, ready to be written.
     */
    private static List<SimpleFeature> transform(Collection<SimpleFeature> sourceFeatures, String sourceAttribute,
            CoordinateTransformation transformation, String targetCrs) {

        SimpleFeatureTypeBuilder typeBuilder = new SimpleFeatureTypeBuilder();
        typeBuilder.setName("berlin_taz");
        typeBuilder.setCRS(MGC.getCRS(targetCrs));
        // A shapefile needs one concrete geometry type; the zones are (multi) polygons.
        typeBuilder.add("the_geom", MultiPolygon.class);
        typeBuilder.add(TARGET_ATTRIBUTE, String.class);
        SimpleFeatureType featureType = typeBuilder.buildFeatureType();

        SimpleFeatureBuilder featureBuilder = new SimpleFeatureBuilder(featureType);
        GeometryTransformer reprojection = new GeometryTransformer() {
            @Override
            protected CoordinateSequence transformCoordinates(CoordinateSequence sequence, Geometry parent) {
                Coordinate[] transformed = new Coordinate[sequence.size()];
                for (int i = 0; i < sequence.size(); i++) {
                    Coord projected = transformation.transform(new Coord(sequence.getX(i), sequence.getY(i)));
                    transformed[i] = new Coordinate(projected.getX(), projected.getY());
                }
                return new CoordinateArraySequence(transformed);
            }
        };

        List<SimpleFeature> zones = new ArrayList<>();
        int skipped = 0;

        for (SimpleFeature feature : sourceFeatures) {
            Object name = feature.getAttribute(sourceAttribute);
            Object geometry = feature.getDefaultGeometry();

            if (name == null || String.valueOf(name).isBlank() || !(geometry instanceof Geometry)) {
                skipped++;
                continue;
            }

            featureBuilder.add(asMultiPolygon(reprojection.transform((Geometry) geometry)));
            featureBuilder.add(String.valueOf(name));
            zones.add(featureBuilder.buildFeature(String.valueOf(zones.size())));
        }

        if (skipped > 0) {
            System.out.println("Skipped " + skipped + " features without a usable "
                    + sourceAttribute + " attribute or geometry.");
        }
        if (zones.isEmpty()) {
            throw new IllegalStateException("No usable zone found in the source file.");
        }

        return zones;
    }

    /**
     * Wraps a single polygon into a multi polygon, so that all features share the geometry type a
     * shapefile has to declare.
     *
     * @param geometry The reprojected geometry.
     * @return The geometry as a multi polygon.
     */
    private static MultiPolygon asMultiPolygon(Geometry geometry) {
        if (geometry instanceof MultiPolygon multiPolygon) {
            return multiPolygon;
        }
        if (geometry instanceof Polygon polygon) {
            return polygon.getFactory().createMultiPolygon(new Polygon[] {polygon});
        }
        throw new IllegalStateException("Zone geometry is neither a polygon nor a multi polygon: "
                + geometry.getGeometryType());
    }

    /**
     * Builds a readable bounding box over all zones, so that the result can be eyeballed against
     * the scenario network.
     *
     * @param zones The zone features.
     * @return The bounding box as "minX, minY .. maxX, maxY".
     */
    private static String describeBounds(List<SimpleFeature> zones) {
        double minX = Double.MAX_VALUE;
        double minY = Double.MAX_VALUE;
        double maxX = -Double.MAX_VALUE;
        double maxY = -Double.MAX_VALUE;

        for (SimpleFeature zone : zones) {
            Geometry geometry = (Geometry) zone.getDefaultGeometry();
            minX = Math.min(minX, geometry.getEnvelopeInternal().getMinX());
            minY = Math.min(minY, geometry.getEnvelopeInternal().getMinY());
            maxX = Math.max(maxX, geometry.getEnvelopeInternal().getMaxX());
            maxY = Math.max(maxY, geometry.getEnvelopeInternal().getMaxY());
        }

        return String.format("%.0f, %.0f .. %.0f, %.0f", minX, minY, maxX, maxY);
    }

    /**
     * Opens a URL or a local path as a stream.
     *
     * @param url Source URL or local path.
     * @return An open stream, to be closed by the caller.
     */
    private static InputStream open(String url) throws IOException {
        if (url.startsWith("http://") || url.startsWith("https://")) {
            return URI.create(url).toURL().openStream();
        }
        return Files.newInputStream(Paths.get(url));
    }

    /**
     * Reads a named command line argument.
     *
     * @param args         The raw arguments.
     * @param name         Argument name, including the leading dashes.
     * @param defaultValue Value used when the argument is absent.
     * @return The argument value or the default.
     */
    private static String argument(String[] args, String name, String defaultValue) {
        for (int i = 0; i < args.length - 1; i++) {
            if (name.equals(args[i])) {
                return args[i + 1];
            }
        }
        return defaultValue;
    }
}
