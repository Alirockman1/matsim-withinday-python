package org.matsim.withinday.environment;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.Scenario;
import org.matsim.api.core.v01.TransportMode;
import org.matsim.api.core.v01.network.Link;
import org.matsim.api.core.v01.network.Network;
import org.matsim.api.core.v01.population.Activity;
import org.matsim.api.core.v01.population.Leg;
import org.matsim.api.core.v01.population.Person;
import org.matsim.api.core.v01.population.PlanElement;
import org.matsim.core.mobsim.qsim.QSim;
import org.matsim.core.mobsim.qsim.interfaces.NetsimLink;
import org.matsim.core.population.routes.NetworkRoute;
import org.matsim.core.router.TripRouter;
import org.matsim.facilities.FacilitiesUtils;

/**
 * Estimates how congested the car network is along the likely route of a trip, BEFORE the agent
 * chooses a mode.
 * <ol>
 *   <li>A car route is computed once as the "skeleton" of the trip (the same A* idea as before;
 *       cached per origin link, destination link and 15-minute departure bin).</li>
 *   <li>The live QSim state is read along that skeleton: moving vehicles on each link compared to
 *       the link's storage space.</li>
 * </ol>
 * The result is the <b>occupancy</b> of the route: vehicles on the route divided by the vehicle
 * spaces of the route. It is scale-free (0 = empty, 1 = completely jammed), and it is corrected
 * for the sample size through the QSim storage capacity factor. A 10 % scenario holds a tenth of
 * the vehicles, so the storage space is scaled by the same factor.
 * <p>
 * Reading guide: where the flow capacity is reached, occupancy is about 0.3 on typical urban links
 * (1800 veh/h/lane at 45 km/h, 7.5 m per vehicle). Hence the default thresholds 0.10 / 0.20 / 0.35.
 * <p>
 * The live state describes "now". If the departure is later (next-trip context), treat the value
 * as the current surroundings, not as a forecast.
 */
public final class Congestion {

    private static final Logger log = LogManager.getLogger(Congestion.class);

    /** Occupancy thresholds between the levels 0 (free), 1 (light), 2 (busy), 3 (congested). */
    private static final double[] LEVEL_THRESHOLDS = {0.10, 0.20, 0.35};

    /** Number of discrete congestion levels, for example for convertToBitStateRepresentation. */
    public static final int LEVELS = LEVEL_THRESHOLDS.length + 1;

    /** Departure times are grouped into bins of this length for the route cache. */
    private static final double CACHE_TIME_BIN_S = 900.0;

    private static final Map<String, List<Id<Link>>> ROUTE_CACHE = new ConcurrentHashMap<>();
    private static final AtomicBoolean WARNED = new AtomicBoolean(false);

    private Congestion() {}

    /**
     * @param occupancy Vehicles on the route divided by the storage space of the route (0 .. about 1).
     * @param links     Number of links the value is based on; 0 means "no data" (not "free flow").
     */
    public record Estimate(double occupancy, int links) {
        public static final Estimate NONE = new Estimate(0.0, 0);

        public boolean hasData() {return links > 0;}

        /** Discrete congestion level 0 .. LEVELS-1. */
        public int level() {return discretize(occupancy);}
    }

    /** Maps an occupancy value to a discrete level 0 .. LEVELS-1. */
    public static int discretize(double occupancy) {
        int level = 0;
        for (double threshold : LEVEL_THRESHOLDS) {
            if (occupancy >= threshold) level++;
        }
        return level;
    }

    /** Empties the route cache. Call it at the start of every iteration (travel times change). */
    public static void clearRouteCache() {ROUTE_CACHE.clear();}

    /**
     * @param sim                   The running QSim (pass the current instance, never a stored one).
     * @param scenario              The scenario.
     * @param router                A trip router (the one your replanner already uses).
     * @param person                The agent's person. Pass the real person: with
     *                              vehiclesSource = modeVehicleTypesFromVehiclesData the router can fail
     *                              without one.
     * @param from                  The activity the agent is at.
     * @param to                    The next activity.
     * @param departureTime         Departure time in seconds.
     * @param storageCapacityFactor config.qsim().getStorageCapacityFactor() (0.1 for a 10 % scenario).
     * @return The estimate, or {@link Estimate#NONE} if nothing could be evaluated.
     */
    public static Estimate estimate(QSim sim, Scenario scenario, TripRouter router, Person person,
                                    Activity from, Activity to, double departureTime) {
        Network network = scenario.getNetwork();
        double storageCapacityFactor = scenario.getConfig().qsim().getStorageCapFactor();
        
        if (sim == null || network == null || router == null) {return Estimate.NONE;}

        try {
            List<Id<Link>> route = getRouteSkeleton(network, router, person, from, to, departureTime);

            double vehicles = 0.0;
            double spaces = 0.0;
            int evaluated = 0;

            for (Id<Link> linkId : route) {
                Link link = network.getLinks().get(linkId);
                NetsimLink netsimLink = sim.getNetsimNetwork().getNetsimLink(linkId);
                if (link == null || netsimLink == null) continue;

                // Moving and queueing vehicles only: parked cars are not traffic
                vehicles += netsimLink.getAllNonParkedVehicles().size();
                spaces += Math.max(1.0,
                        link.getLength() * link.getNumberOfLanes() / network.getEffectiveCellSize() * storageCapacityFactor);
                evaluated++;
            }

            if (evaluated == 0) return Estimate.NONE;
            return new Estimate(vehicles / spaces, evaluated);

        } catch (Exception e) {
            // Never fail silently: a constant 0 would look like "free flow" in the state.
            if (WARNED.compareAndSet(false, true)) {
                log.warn("Pre-routing congestion estimate failed, returning 'no data' (this warning is shown once).", e);
            }
            return Estimate.NONE;
        }
    }

    /** Link ids of the car route, including the start and end link, cached per OD pair and time bin. */
    private static List<Id<Link>> getRouteSkeleton(Network network, TripRouter router, Person person,
                                                   Activity from, Activity to, double departureTime) {
        String key = from.getLinkId() + ">" + to.getLinkId() + "@" + (int) (departureTime / CACHE_TIME_BIN_S);
        return ROUTE_CACHE.computeIfAbsent(key, k -> computeRoute(router, person, from, to, departureTime));
    }

    private static List<Id<Link>> computeRoute(TripRouter router, Person person,
                                               Activity from, Activity to, double departureTime) {
        List<? extends PlanElement> tripElements = router.calcRoute(
                TransportMode.car,
                FacilitiesUtils.wrapActivity(from),
                FacilitiesUtils.wrapActivity(to),
                departureTime,
                person,
                person.getAttributes());

        List<Id<Link>> links = new ArrayList<>();
        if (tripElements == null) return links;

        for (PlanElement element : tripElements) {
            if (element instanceof Leg leg && leg.getRoute() instanceof NetworkRoute networkRoute) {
                // getLinkIds() leaves out the start and end link: the origin link is exactly "near the agent"
                links.add(networkRoute.getStartLinkId());
                links.addAll(networkRoute.getLinkIds());
                links.add(networkRoute.getEndLinkId());
            }
        }
        return links;
    }
}