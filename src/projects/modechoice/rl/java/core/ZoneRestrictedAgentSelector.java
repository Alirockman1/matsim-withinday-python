package modechoice.rl.java.core;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.matsim.api.core.v01.Coord;
import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.Scenario;
import org.matsim.api.core.v01.network.Link;
import org.matsim.api.core.v01.population.Activity;
import org.matsim.api.core.v01.population.Person;
import org.matsim.api.core.v01.population.Plan;
import org.matsim.core.router.TripStructureUtils;
import org.matsim.core.router.TripStructureUtils.StageActivityHandling;
import org.matsim.facilities.ActivityFacility;
import org.matsim.withinday.core.AgentSelector;
import org.matsim.withinday.utils.WithinDayConfigGroup;

import com.google.inject.Inject;

import modechoice.rl.java.utils.CustomConfigGroup;
import modechoice.rl.java.utils.TrafficAnalysisZoneLoader;

/**
 * <h2>ZoneRestrictedAgentSelector</h2>
 * <p>
 * An {@link AgentSelector} that restricts the train and test pools to agents who stay
 * entirely inside a region of the network. The region is the set of traffic analysis
 * zones (TAZ) with bin indices {@code 0 .. innerZoneCount - 1}. Because zone bins are
 * sorted by distance from the network center, this region is a central core that grows
 * as {@code innerZoneCount} increases.
 * </p>
 * <p>
 * A person is eligible only if every real (non-stage) activity of the selected plan
 * lies inside the region. Every consecutive current-to-next activity pair is then
 * inside the region by construction, so no out-of-region trips are ever presented to
 * the RL agent.
 * </p>
 * <p>
 * The {@code samplingPercentage} of the parent class is applied to the <em>eligible</em>
 * pool, not to the whole population. The test set is a fixed ratio of the train set,
 * drawn from the remaining eligible persons.
 * </p>
 */
public class ZoneRestrictedAgentSelector extends AgentSelector {

    private static final Logger log = LogManager.getLogger(ZoneRestrictedAgentSelector.class);

    /** Sentinel bin value marking a person as ineligible (any activity outside all zones or unresolvable). */
    private static final int NOT_ELIGIBLE = Integer.MAX_VALUE;

    private final Scenario scenario;
    private final WithinDayConfigGroup withinDayConfig;

    /** Number of inner zones forming the region (bins 0..n-1). A value of 0 or less disables the restriction. */
    private final int innerZoneCount;

    /** Cache of the highest zone bin visited by each person, so each plan is evaluated only once. */
    private final Map<Id<Person>, Integer> maxBinByPerson = new HashMap<>();

    /**
     * Constructs the selector via dependency injection and reads the zone restriction
     * from the custom configuration group.
     *
     * @param scenario    The MATSim scenario containing the population, network and facilities.
     * @param configGroup The within-day configuration group (sampling percentage, agent lists, etc.).
     */
    @Inject
    public ZoneRestrictedAgentSelector(Scenario scenario, WithinDayConfigGroup configGroup) {
        super(scenario, configGroup);
        this.scenario = scenario;
        this.withinDayConfig = configGroup;

        CustomConfigGroup custom = (CustomConfigGroup) scenario.getConfig().getModule(CustomConfigGroup.GROUP_NAME);
        this.innerZoneCount = (custom != null) ? custom.getFilteredZonesCount() : 0;
    }

    /**
     * Indicates whether the zone restriction is active.
     *
     * @return {@code true} if {@code innerZoneCount} is greater than zero, {@code false} otherwise.
     */
    private boolean isRestricted() {return innerZoneCount > 0;}

    /**
     * Decides whether a person may enter the train or test pool. Called by the parent class
     * while building the candidate pool, before the train/test split.
     *
     * @param person The person to evaluate.
     * @return {@code true} if the restriction is disabled, or if all of the person's real
     *         activities lie within zones {@code 0 .. innerZoneCount - 1}; {@code false} otherwise
     *         (including for a {@code null} person).
     */
    @Override
    protected boolean isEligiblePerson(Person person) {
        if (!isRestricted()) return true;
        return person != null && maxBin(person) < innerZoneCount;
    }

    /**
     * Builds the fixed master pool of train and test agents. Must be called after the TAZ
     * zones are loaded (in this project, {@code initializeExternalModel()} runs before this
     * method in {@code notifyStartup}).
     * <p>
     * Logs the number of eligible agents for every possible region size, the expected
     * train and test sizes for the selected region, and a final check that both pools
     * lie fully inside the region.
     * </p>
     *
     * @throws IllegalStateException if the restriction is active but no zones are loaded, or if
     *                               fewer than two persons are eligible (a train/test split is impossible).
     */
    @Override
    public void sampleAgentsForSimulation() {
        maxBinByPerson.clear();

        if (!isRestricted()) {
            log.info("ZoneRestrictedAgentSelector: no zone restriction applied.");
            super.sampleAgentsForSimulation();
            return;
        }

        int totalBins = TrafficAnalysisZoneLoader.getTotalBins();
        if (totalBins == 0) {
            throw new IllegalStateException("No TAZ zones loaded; cannot apply the zone restriction.");
        }

        // Informational only: number of eligible agents if the region were zones 0..k-1
        int[] countByMaxBin = new int[totalBins];
        for (Person person : scenario.getPopulation().getPersons().values()) {
            int mb = maxBin(person);
            if (mb < totalBins) countByMaxBin[mb]++;
        }

        int cumulative = 0;
        int eligibleNow = 0;
        for (int k = 1; k <= totalBins; k++) {
            cumulative += countByMaxBin[k - 1];
            if (k == innerZoneCount) eligibleNow = cumulative;
            log.info("  zones 0..{} -> {} eligible agents{}", k - 1, cumulative, k == innerZoneCount ? "  <== selected" : "");
        }
        if (innerZoneCount > totalBins) eligibleNow = cumulative;

        double pct = withinDayConfig.getSamplingPercentage();
        int expectedTrain = (int) Math.max(1, Math.round(eligibleNow * pct));
        int expectedTest = (int) Math.max(1, Math.round(expectedTrain * DEFAULT_TEST_TO_TRAIN_RATIO));
        log.info("ZoneRestrictedAgentSelector: {} of {} persons eligible -> expected ~{} train, ~{} test (samplingPercentage={}).",
                eligibleNow, scenario.getPopulation().getPersons().size(), expectedTrain, expectedTest, pct);

        if (eligibleNow < 2) {
            throw new IllegalStateException("Fewer than 2 eligible agents. Increase the zone count, or check that the "
                    + "TAZ shapefile loaded (K-means fallback) and that its CRS matches the network.");
        }
        if (withinDayConfig.getAgentsPerIteration() > expectedTrain) {
            log.warn("agentsPerIteration ({}) exceeds the expected train size ({}); fewer agents will run per iteration.",
                    withinDayConfig.getAgentsPerIteration(), expectedTrain);
        }

        super.sampleAgentsForSimulation();

        // Sanity check: both pools must lie fully inside the region
        long badTrain = getAgentsFromSimulationSubspace().stream()
                .filter(id -> !isEligiblePerson(scenario.getPopulation().getPersons().get(id))).count();
        long badTest = getAgentsFromSimulationTestSubspace().stream()
                .filter(id -> !isEligiblePerson(scenario.getPopulation().getPersons().get(id))).count();
        log.info("ZoneRestrictedAgentSelector: train subspace {}, test pool {} (outside the region: {} / {}).",
                getSimulationSubspaceSize(), getSimulationTestSize(), badTrain, badTest);
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    /**
     * Returns the highest zone bin visited by a person, using a cache so each plan
     * is evaluated at most once.
     *
     * @param person The person to evaluate.
     * @return The highest zone bin over all real activities, or {@link #NOT_ELIGIBLE}
     *         if the person cannot be placed fully inside the zones.
     */
    private int maxBin(Person person) {
        return maxBinByPerson.computeIfAbsent(person.getId(), id -> computeMaxBin(person));
    }

    /**
     * Computes the highest zone bin over all real (non-stage) activities in the person's
     * selected plan. Because zone bins are ordered by distance from the center, a person
     * is inside the region of size {@code k} exactly when this value is below {@code k}.
     *
     * @param person The person whose selected plan is evaluated.
     * @return The highest zone bin (0 or greater), or {@link #NOT_ELIGIBLE} if the person has
     *         no plan, fewer than two activities (no trip), an activity whose location cannot be
     *         resolved, or an activity outside every zone.
     */
    private int computeMaxBin(Person person) {
        Plan plan = person.getSelectedPlan();
        if (plan == null) return NOT_ELIGIBLE;

        List<Activity> activities = TripStructureUtils.getActivities(plan, StageActivityHandling.ExcludeStageActivities);
        if (activities.size() < 2) return NOT_ELIGIBLE;   // no trip, nothing to learn from

        int max = 0;
        for (Activity activity : activities) {
            Coord coord = resolveCoord(activity);
            if (coord == null) return NOT_ELIGIBLE;

            int bin = TrafficAnalysisZoneLoader.getZoneBin(coord);
            if (bin < 0) return NOT_ELIGIBLE;
            max = Math.max(max, bin);
        }
        return max;
    }

    /**
     * Resolves the coordinate of an activity, trying the activity itself, then its
     * facility, then its network link.
     *
     * @param activity The activity whose location is needed.
     * @return The resolved {@link Coord}, or {@code null} if none of the three sources provides one.
     */
    private Coord resolveCoord(Activity activity) {
        if (activity.getCoord() != null) return activity.getCoord();

        if (activity.getFacilityId() != null) {
            ActivityFacility facility = scenario.getActivityFacilities().getFacilities().get(activity.getFacilityId());
            if (facility != null && facility.getCoord() != null) return facility.getCoord();
        }

        if (activity.getLinkId() != null) {
            Link link = scenario.getNetwork().getLinks().get(activity.getLinkId());
            if (link != null) return link.getCoord();
        }
        return null;
    }
}
