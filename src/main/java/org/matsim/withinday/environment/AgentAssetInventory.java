package org.matsim.withinday.environment;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.matsim.api.core.v01.Coord;
import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.Scenario;
import org.matsim.api.core.v01.network.Link;
import org.matsim.api.core.v01.population.Activity;
import org.matsim.api.core.v01.population.Person;
import org.matsim.api.core.v01.population.Plan;
import org.matsim.api.core.v01.population.PlanElement;

import org.matsim.core.mobsim.framework.MobsimAgent;
import org.matsim.core.mobsim.qsim.agents.WithinDayAgentUtils;
import org.matsim.core.network.NetworkUtils;
import org.matsim.core.router.TripStructureUtils;
import org.matsim.core.router.TripStructureUtils.Trip;
import org.matsim.withinday.utils.WithinDayConfigGroup;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Manages physical asset tracking, vehicle location inventories, and mode availability for simulation agents.
 */
public class AgentAssetInventory {
    private static final Logger log = LogManager.getLogger(AgentAssetInventory.class);
    
    // CRITICAL: Thread-safe data structures prevent data corruption during parallel QSim runs
    private static final Map<Id<Person>, Map<String, Id<Link>>> agentModeInventory = new ConcurrentHashMap<>();
    private static final Map<Id<Person>, Id<Link>> agentLastLink = new ConcurrentHashMap<>();
    private static final Map<Id<Person>, Id<Link>> startOfDayLocations = new ConcurrentHashMap<>();
    private static final Map<Id<Person>, Boolean> agentTourStatus = new ConcurrentHashMap<>();
    
    // Global Mode Sets initialized once from Config
    private static final Set<String> ALL_MODES = ConcurrentHashMap.newKeySet();
    private static final Set<String> TOUR_BASED_MODES = ConcurrentHashMap.newKeySet();

    private static final Id<Link> DUMMY_RESTRICTED_LINK = Id.createLinkId("99999999");

    /**
     * Configures simulation transport modes from the scenario configuration group.
     *
     * @param scenario The active simulation scenario.
     */
    public static void setSimulationBasedModes(Scenario scenario){
        WithinDayConfigGroup configGroup = (WithinDayConfigGroup) scenario.getConfig().getModule(WithinDayConfigGroup.GROUP_NAME);
        
        if (configGroup != null) {
            parseAndAddModes(configGroup.getModes(), ALL_MODES);
            parseAndAddModes(configGroup.getTourBasedModes(), TOUR_BASED_MODES);
        }
    }
    
    /**
     * Calculates the retrieval time required if an agent abandons a tour-based vehicle asset.
     *
     * @param agent            The simulation agent.
     * @param scenario         The simulation scenario.
     * @param currentTripIndex The index of the current trip.
     * @param log              Logger instance.
     * @return                 Total retrieval time in seconds.
     */
    public static Double getModeRetrievalTimes(MobsimAgent agent, Scenario scenario, int currentTripIndex, Logger log){

        Double modeRetrievalTime = 0.0;

        Id<Person> agentId = agent.getId();
        Id<Link> currentLink = agent.getCurrentLinkId();
        List<Trip> trips = TripStructureUtils.getTrips(WithinDayAgentUtils.getModifiablePlan(agent));
        Map<String, Id<Link>> currentInventory = getModeLocation(agentId);

        Activity firstActivity = (Activity) WithinDayAgentUtils.getModifiablePlan(agent).getPlanElements().get(0);

        for (String assetMode : TOUR_BASED_MODES) {
            Id<Link> vehicleLocation = currentInventory.get(assetMode);

            if (!vehicleLocation.toString().contains("99999999")){
                boolean isWithAgent = vehicleLocation.equals(currentLink);
                boolean isAtHome = vehicleLocation.equals(firstActivity.getLinkId());

                // Get the activity the asset was last seen on.
                String activityAtVehicleLocation = "";
                for (PlanElement pe : WithinDayAgentUtils.getModifiablePlan(agent).getPlanElements()) {
                    if (pe instanceof Activity act) {
                        String activityType = act.getType();
                        
                        if (!activityType.contains("interaction") && act.getLinkId().equals(vehicleLocation)){
                            activityAtVehicleLocation = activityType;
                            break;
                        }
                    }
                }

                // Evaluate if the trip is part of a sub-tour
                boolean returnsToAssetLocation = false;
                Activity activityToreturn = null;
                
                for (int i = currentTripIndex + 1; i < trips.size(); i++) {
                    Activity futureActivity = trips.get(i).getDestinationActivity();

                    // Match based on the activity type found at the vehicle's location
                    if (futureActivity.getLinkId().equals(vehicleLocation) && 
                        futureActivity.getType().equals(activityAtVehicleLocation)) {
                        returnsToAssetLocation = true;
                        activityToreturn = futureActivity;
                        break;
                    }
                }

                if(!isWithAgent && !isAtHome && returnsToAssetLocation){log.info("The agent " + agentId.toString() + " is on a sub-tour mode.");}

                // Asset mode left behind 
                boolean modeAbandoned = !isWithAgent && !returnsToAssetLocation && !isAtHome;

                if (modeAbandoned) {
                    // Eucleadian Distance between the agent Coordinates and vehicle location
                    Coord agentCoord = scenario.getNetwork().getLinks().get(currentLink).getCoord();
                    Coord vehicleCoord = scenario.getNetwork().getLinks().get(vehicleLocation).getCoord();
                    double distanceToVehicle = NetworkUtils.getEuclideanDistance(agentCoord, vehicleCoord);

                    double walkSpeed = scenario.getConfig().routing().getTeleportedModeSpeeds().get("walk");
                    double walkBeelineFactor = scenario.getConfig().routing().getBeelineDistanceFactors().get("walk");
                    
                    // Retrivela time of the Asset
                    double retrievalTime = (distanceToVehicle * walkBeelineFactor) / walkSpeed;
                    modeRetrievalTime += retrievalTime;

                    log.warn(assetMode.toUpperCase() + " ABANDONED at " + vehicleLocation + ". Time: " + retrievalTime);
                }else{
                    modeRetrievalTime += 0.0;
                }
            }else{
                modeRetrievalTime += 0.0;
            }
        }

        return modeRetrievalTime;
    }

    /**
     * Returns an unmodifiable set of all simulation modes.
     *
     * @return Set of mode strings.
     */
    public static Set<String> getSimulationModes() {return Collections.unmodifiableSet(ALL_MODES);}
    
    /**
     * Returns an unmodifiable set of all tour-based simulation modes.
     *
     * @return Set of tour-based mode strings.
     */
    public static Set<String> getSimulationTourBasedModes() {return Collections.unmodifiableSet(TOUR_BASED_MODES);}
    
    /**
     * Returns all simulation modes formatted as a comma-separated string.
     *
     * @return Comma-separated mode string.
     */
    public static String getSimulationModesAsString() { 
        Set<String> simulationModes = getSimulationModes();
        return String.join(",", simulationModes);
    }
    
    /**
     * Returns tour-based simulation modes formatted as a comma-separated string.
     *
     * @return Comma-separated tour-based mode string.
     */
    public static String getSimulationTourBasedModesAsString() {         
        Set<String> simulationTourBasedModes = getSimulationTourBasedModes();
        return String.join(",", simulationTourBasedModes);
    }

    /**
     * Sets the tour-based plan status for an agent.
     *
     * @param agentId The agent ID.
     * @param isTour  True if the plan is tour-based, false otherwise.
     */
    public static void setTourBasedPlan(Id<Person> agentId, boolean isTour) {agentTourStatus.put(agentId, isTour);}

    /**
     * Returns whether an agent's plan is tour-based.
     *
     * @param agentId The agent ID.
     * @return        {@code True} if tour-based, {@code False} otherwise.
     */
    public static boolean getIsTourBased(Id<Person> agentId) {return agentTourStatus.getOrDefault(agentId, false);}

    /**
     * Sets the start-of-day (home) location link ID for an agent.
     *
     * @param personId The person ID.
     * @param linkId   The starting link ID {@code Id<Link>}.
     */
    public static void setStartOfDayLocation(Id<Person> personId, Id<Link> linkId) {startOfDayLocations.putIfAbsent(personId, linkId);}

    /**
     * Retrieves the start-of-day (home) link ID for an agent.
     *
     * @param personId The person ID.
     * @return         The starting link ID {@code Id<Link>}.
     */
    public static Id<Link> getStartOfDayLocation(Id<Person> personId){return startOfDayLocations.get(personId);}

    /**
     * Sets the parked location link ID for a specific mode and agent.
     *
     * @param personId The person ID.
     * @param mode     The transport mode.
     * @param linkId   The link ID where the asset is parked.
     */
    public static void setModeLocation(Id<Person> personId, String mode, Id<Link> linkId) {
        agentLastLink.put(personId, linkId);
        agentModeInventory.computeIfAbsent(personId, k -> new ConcurrentHashMap<>()).put(mode, linkId);
    }
    
    /**
     * Retrieves the map of mode locations for a given agent.
     *
     * @param personId The person ID.
     * @return         A map of transport modes {@code String} to their respective parked link IDs {@code Id<Link>}.
     */
    public static Map<String,Id<Link>> getModeLocation(Id<Person> personId) {
        if (!agentModeInventory.containsKey(personId)) {return Collections.emptyMap();}
        return agentModeInventory.get(personId);
    }

    /**
     * Retrieves the most recent link ID associated with an agent.
     *
     * @param personId The person ID {@code Id<Person>}.
     * @return         The recent link ID {@code Id<Link>}.
     */
    public static Id<Link> getAgentLinkID(Id<Person> personId) {return agentLastLink.get(personId); }

    /**
     * Computes and returns the mode discontinuity penalty map for an agent at a given link.
     *
     * @param personId The person ID.
     * @param linkID   The target link ID.
     * @return         A map of mode penalties.
     */
    public static Map<String, Integer> getModeDiscontinuityPenalty(Id<Person> personId, Id<Link> linkID){
        Map<String, Integer> penaltyMap = modeDiscontinuityPenalty(linkID, personId);
        return penaltyMap;
    }

    /**
     * Initializes initial mode locations and geo-tags for filtered agents at the start of an iteration.
     *
     * @param scenario         The active simulation scenario.
     * @param selectedAgentsId Set of agent IDs selected for within-day replanning.
     */
    public static void initializeModeLocationTagging(Scenario scenario, Set<Id<Person>> selectedAgentsId){
        // Attach a Geo-tag for filtered individuals
        for (Id<Person> agentID : selectedAgentsId){
            Person person = scenario.getPopulation().getPersons().get(agentID);
            Plan plan = person.getSelectedPlan();

            if (plan == null || plan.getPlanElements().isEmpty()) {
                log.warn("Agent {} skipped initialization: Plan is null or empty.", agentID);
                continue;
            }

            Activity firstActivity = (Activity) plan.getPlanElements().get(0);
            Id<Link> startLinkID = firstActivity.getLinkId();

            if (startLinkID == null) {
                log.warn("Agent {} skipped initialization: First activity startLinkID is null.", agentID);
                continue;
            }

            // Verify if the key is missing OR the map is empty
            if (!agentModeInventory.containsKey(agentID) || getModeLocation(agentID).isEmpty()) {
                setStartOfDayLocation(agentID, startLinkID);
                
                for (String mode : ALL_MODES) {
                    boolean isAssetAvailable = checkAssetAvailability(person, mode);

                    if (!isAssetAvailable) {
                        setModeLocation(agentID, mode, Id.createLinkId("99999999"));
                    } else {
                        setModeLocation(agentID, mode.trim(), startLinkID);
                    }                
                }
            }

            // Check for tour based plans
            Activity lastActivity = (Activity) plan.getPlanElements().get(plan.getPlanElements().size() - 1);
            boolean isTourBasedPlan = firstActivity.getLinkId().equals(lastActivity.getLinkId());
            setTourBasedPlan(agentID, isTourBasedPlan);
        }

        log.info("Geo-Tags attached to the selected agents in the environment.");
    }
    
    /**
     * @helper Parses a comma-separated string of modes and adds them to the target set.
     *
     * @param rawString Raw string containing comma-separated mode names.
     * @param targetSet The target set to populate.
     */
    private static void parseAndAddModes(String rawString, Set<String> targetSet) {
        if (rawString == null || rawString.trim().isEmpty()) {return;}
        
        Arrays.stream(rawString.split("\\s*,\\s*"))
            .map(String::trim)
            .filter(s -> !s.isEmpty())
            .map(String::toLowerCase)
            .forEach(targetSet::add);
    }

    /**
     * Checks whether a specific transport mode asset is available for a given person.
     *
     * @param person The simulation person/agent.
     * @param mode   The transport mode to check.
     * @return       {@code true} if available or non-tour-based, {@code false} if explicitly restricted.
     */
    private static boolean checkAssetAvailability(Person person, String mode) {

        if (!TOUR_BASED_MODES.contains(mode)) {return true;}

        String attributeKey = mode + "_avail";
        String availability = (String) person.getAttributes().getAttribute(attributeKey);

        if (availability != null) {
            String available = availability.trim().toLowerCase();
            // Returns false if explicitly restricted in the population file
            return !"false".equals(available) && !"never".equals(available);
        }
        return true;
    }

    /**
     * Updates the location of vehicle assets and validates mode usage legality.
     *
     * @param agentId                  The ID of the simulation agent.
     * @param currentLink              The current link ID.
     * @param previousLink             The previous link ID.
     * @param currentModeUsed          The mode currently used by the agent.
     * @param modeDiscontinuityPenalty Penalty map for mode transitions.
     */
    public static void updateModeLocation(Id<Person> agentId, Id<Link> currentLink, Id<Link> previousLink, 
        String currentModeUsed, Map<String, Integer> modeDiscontinuityPenalty){
        Map<String, Integer> discontinuityPenaltyMap = modeDiscontinuityPenalty(previousLink, agentId);

        // Validate the mode was a legal choice
        int discontinuityFlag = discontinuityPenaltyMap.getOrDefault(currentModeUsed, 0);

        // Update all service modes
        for (String mode: ALL_MODES){
            if (!TOUR_BASED_MODES.contains(mode)) {setModeLocation(agentId, mode, currentLink);}
        }

        // Update asset modes (if Legal trip)
        if (TOUR_BASED_MODES.contains(currentModeUsed)) {
            if (discontinuityFlag == 0) {
                setModeLocation(agentId, currentModeUsed, currentLink);
            }else{
                log.warn("RL MODE CHOICE: Illegal " + currentModeUsed + " use for the agent (" + agentId.toString() + "). Asset remains at previous location.");
            }
        }
    }

    /**
     * Computes discontinuity penalties for each mode based on the current link location.
     *
     * @param currentLinkID The current link ID of the agent.
     * @param agentId       The agent ID.
     * @return              A map of mode names to integer penalty values.
     */
    private static Map<String, Integer> modeDiscontinuityPenalty(Id<Link> currentLinkID, Id<Person> agentId) {
        Map<String, Id<Link>> previousInventorySnapshot = new HashMap<>(AgentAssetInventory.getModeLocation(agentId));

        Map<String, Integer> penaltyMap = new HashMap<>();

        if (previousInventorySnapshot == null) return penaltyMap;

        for (Map.Entry<String, Id<Link>> entry : previousInventorySnapshot.entrySet()) {
            String mode = entry.getKey();
            Id<Link> location = entry.getValue();
            int penalty;

            if (location.equals(DUMMY_RESTRICTED_LINK)) {
                penalty = 1;
            } else if (location.equals(currentLinkID)) {
                penalty = 0;
            } else {
                penalty = 1;
            }

            penaltyMap.put(mode, penalty);
        }
        return penaltyMap;
    }

    /**
     * Resets and clears all internal inventory records and tracking maps.
     */
    public static synchronized void reset() {
        agentModeInventory.clear();
        startOfDayLocations.clear();
        agentLastLink.clear();
        agentTourStatus.clear();
    }
}