package modechoice.rl.java.utils;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.Test;
import org.matsim.api.core.v01.Coord;
import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.network.Network;
import org.matsim.core.network.NetworkUtils;
import org.matsim.withinday.utils.StateEngine;

/**
 * <h2>BerlinTrafficAnalysisZoneTest</h2>
 * <p>
 * Guards the traffic analysis zones {@code berlin.java.prepare.PrepareBerlinTaz} writes. The zone
 * lookup is a plain point-in-polygon query without any coordinate conversion, so the zones have to
 * be stored in the scenario's coordinate system - a shapefile left in WGS84 would resolve every
 * Berlin coordinate to bin {@code -1} without any error being raised.
 * </p>
 * <p>
 * The coordinates used here are activity locations taken from {@code berlin-mini.plans.xml.gz},
 * which is EPSG:25832 like the rest of the scenario.
 * </p>
 */
public class BerlinTrafficAnalysisZoneTest {

    private static final String TAZ_FILE = "scenarios/berlin-v7.1.1/input/berlin-taz.shp";

    /** Twelve Berlin districts plus the fifteen surrounding federal states. */
    private static final int EXPECTED_ZONES = 27;

    /** A home activity of agent bb_00007dd9. */
    private static final Coord BERLIN_HOME = new Coord(800908.01, 5807691.37);

    /** A work activity of the same agent, in a different district. */
    private static final Coord BERLIN_WORK = new Coord(798274.38, 5814530.0);

    @Test
    void zonesResolveScenarioCoordinates() {
        assertTrue(Files.exists(Path.of(TAZ_FILE)),
                TAZ_FILE + " is missing; run berlin.java.prepare.PrepareBerlinTaz to create it");

        // The loader reads the network centroid to order the bins by distance from the center.
        StateEngine.setNetworkCentroid(berlinNetwork());

        TrafficAnalysisZoneLoader.loadZonesSortedByDistanceFromCenter(TAZ_FILE, "TAZ_ID", null);

        // A fallback to synthetic K-means zones would show up as a different count.
        assertEquals(EXPECTED_ZONES, TrafficAnalysisZoneLoader.getTotalBins(),
                "the shapefile was not read; the loader fell back to synthetic zones");

        int homeBin = TrafficAnalysisZoneLoader.getZoneBin(BERLIN_HOME);
        int workBin = TrafficAnalysisZoneLoader.getZoneBin(BERLIN_WORK);

        assertTrue(homeBin >= 0, "a Berlin home location resolved to no zone; check the zone coordinate system");
        assertTrue(workBin >= 0, "a Berlin work location resolved to no zone; check the zone coordinate system");

        // Bins are ordered by distance from the network center, so Berlin's districts come first.
        assertTrue(homeBin < 12, "a Berlin home location resolved to a zone outside Berlin, bin " + homeBin);
        assertTrue(workBin < 12, "a Berlin work location resolved to a zone outside Berlin, bin " + workBin);
    }

    /**
     * Builds a minimal network whose bounding box center is Berlin, which is all the zone loader
     * reads from it. Reading the real network would need the mirrored scenario input.
     *
     * @return A network with two nodes around the Berlin city center.
     */
    private static Network berlinNetwork() {
        Network network = NetworkUtils.createNetwork();
        NetworkUtils.createAndAddNode(network, Id.createNodeId("southWest"), new Coord(770000, 5790000));
        NetworkUtils.createAndAddNode(network, Id.createNodeId("northEast"), new Coord(820000, 5840000));
        return network;
    }
}
