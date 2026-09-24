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
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

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

    public static synchronized void reset() {
        agentModeInventory.clear();
        startOfDayLocations.clear();
        agentLastLink.clear();
        agentTourStatus.clear();
    }

    // This method initialize the location of agents and the modes
    public static void initializeModeLocationTagging(Scenario scenario, Set<Id<Person>> selectedAgentsId){
        // Attach a Geo-tag for filtered individuals
        for (Id<Person> agentID : selectedAgentsId){
            // Get ID of the person
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

            // Fixed check: verifies if the key is missing OR the map is empty
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

            // Look if the plan is tour based for all agents (start activity location is same as the last activity location)
            Activity lastActivity = (Activity) plan.getPlanElements().get(plan.getPlanElements().size() - 1);
            boolean isTourBasedPlan = firstActivity.getLinkId().equals(lastActivity.getLinkId());

            setTourBasedPlan(agentID, isTourBasedPlan);
        }

        log.info("Geo-Tags attached to the selected agents in the environment.");
    }

    private static void parseAndAddModes(String rawString, Set<String> targetSet) {
        if (rawString == null || rawString.trim().isEmpty()) {
            return;
        }
        
        Arrays.stream(rawString.split("\\s*,\\s*"))
            .map(String::trim)
            .filter(s -> !s.isEmpty())
            .map(String::toLowerCase)
            .forEach(targetSet::add);
    }

    private static boolean checkAssetAvailability(Person person, String mode) {

        if (!TOUR_BASED_MODES.contains(mode)) {
            return true;
        }

        String attributeKey = mode + "_avail";
        String availability = (String) person.getAttributes().getAttribute(attributeKey);

        if (availability != null) {
            String available = availability.trim().toLowerCase();
            // Returns false if explicitly restricted in the population file
            return !"false".equals(available) && !"never".equals(available);
        }

        return true;
    }

    // This method updates the location of vehicels
    public static void updateModeLocation(Id<Person> agentId, Id<Link> currentLink, Id<Link> previousLink, 
        String currentModeUsed, Map<String, Integer> modeDiscontinuityPenalty){

        Map<String, Integer> discontinuityPenaltyMap = modeDiscontinuityPenalty(previousLink, agentId);
        //Map<String, Integer> modeDiscontinuityPenalty1 = ModeUtils.modeDiscontinuityPenalty(previousLink.toString(),previousInventorySnapshot);
        
        // Validate the mode was a legal choice
        int discontinuityFlag = discontinuityPenaltyMap.getOrDefault(currentModeUsed, 0);

        // Update all service modes
        for (String mode: ALL_MODES){
            if (!TOUR_BASED_MODES.contains(mode)) {
                setModeLocation(agentId, mode, currentLink);
            }
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

    // This method returns the penalty map for the current location
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

    // SETTER: Allocate the Simulation modes as a set of strings
    public static void setSimulationBasedModes(Scenario scenario){
        WithinDayConfigGroup configGroup = (WithinDayConfigGroup) scenario.getConfig().getModule(WithinDayConfigGroup.GROUP_NAME);
        
        if (configGroup != null) {
            parseAndAddModes(configGroup.getModes(), ALL_MODES);
            parseAndAddModes(configGroup.getTourBasedModes(), TOUR_BASED_MODES);
        }
    }
    
    // GETTER: Retrieval times incase of abandoned mode
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

                // 3. Evaluate if the trip is part of a sub-tour -> allow mode within subtour to be different
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

                // Abandoned = Not with me AND No plans to go back AND Not safely at Home
                boolean modeAbandoned = !isWithAgent && !returnsToAssetLocation && !isAtHome;

                // 4. If stranded and not sub-tour mode ... compute the retrieval time of the mode (distance(tagent, vehicle from mode inventory)) using travel parameters of walk mode
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
                    //modeRetrievalTime.put(assetMode, retrievalTime);

                    log.warn(assetMode.toUpperCase() + " ABANDONED at " + vehicleLocation + ". Time: " + retrievalTime);
                }else{
                    //modeRetrievalTime.put(assetMode, 0.0);
                    modeRetrievalTime += 0.0;
                }

            }else{
                //modeRetrievalTime.put(assetMode, 0.0);
                modeRetrievalTime += 0.0;
            }
        }

        return modeRetrievalTime;
    }

    // GETTER: All simulation modes
    public static Set<String> getSimulationModes() {return Collections.unmodifiableSet(ALL_MODES);}
    
    // GETTER: Tour based modes specified in the config
    public static Set<String> getSimulationTourBasedModes() {return Collections.unmodifiableSet(TOUR_BASED_MODES);}
    
    // GETTER: All simulation modes as a string
    public static String getSimulationModesAsString() { 
        Set<String> simulationModes = getSimulationModes();
        return String.join(",", simulationModes);
    }
    
    // GETTER: Tour based modes specified in the config as string
    public static String getSimulationTourBasedModesAsString() {         
        Set<String> simulationTourBasedModes = getSimulationTourBasedModes();
        return String.join(",", simulationTourBasedModes);
    }

    // SETTER: Update the tour plan for agents
    public static void setTourBasedPlan(Id<Person> agentId, boolean isTour) {
        agentTourStatus.put(agentId, isTour);
    }

    // GETTER: Tour based plan
    public static boolean getIsTourBased(Id<Person> agentId) {
        return agentTourStatus.getOrDefault(agentId, false);
    }

    // SETTER: Update the "Home" location for each agent
    public static void setStartOfDayLocation(Id<Person> personId, Id<Link> linkId) {
        startOfDayLocations.putIfAbsent(personId, linkId);
    }

    // GETTER: The home link ID
    public static Id<Link> getStartOfDayLocation(Id<Person> personId){
        return startOfDayLocations.get(personId);
    }

    // SETTER: Update where a specific mode is parked
    public static void setModeLocation(Id<Person> personId, String mode, Id<Link> linkId) {
        agentLastLink.put(personId, linkId);

        agentModeInventory.computeIfAbsent(personId, k -> new ConcurrentHashMap<>()).put(mode, linkId);
    }
    
    // GETTER: The location of all modes
    public static Map<String,Id<Link>> getModeLocation(Id<Person> personId) {
        if (!agentModeInventory.containsKey(personId)) {
            return Collections.emptyMap();
        }
        return agentModeInventory.get(personId);
    }

    // GETTER: The recent Link ID
    public static Id<Link> getAgentLinkID(Id<Person> personId) {
        return agentLastLink.get(personId); 
    }

    // GETTER: The recent penalty map
    public static Map<String, Integer> getModeDiscontinuityPenalty(Id<Person> personId, Id<Link> linkID){
        Map<String, Integer> penaltyMap = modeDiscontinuityPenalty(linkID, personId);

        return penaltyMap;
    }
}