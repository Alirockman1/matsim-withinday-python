package modechoice.rl.java.core;

import java.io.File;
import java.net.URISyntaxException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.Scenario;
import org.matsim.api.core.v01.network.Link;
import org.matsim.api.core.v01.population.Activity;
import org.matsim.api.core.v01.population.Leg;
import org.matsim.api.core.v01.population.Person;
import org.matsim.api.core.v01.population.Plan;
import org.matsim.api.core.v01.population.PlanElement;
import org.matsim.core.config.Config;
import org.matsim.core.controler.events.IterationEndsEvent;
import org.matsim.core.mobsim.framework.MobsimAgent;
import org.matsim.core.mobsim.qsim.QSim;
import org.matsim.core.mobsim.qsim.agents.WithinDayAgentUtils;
import org.matsim.core.router.TripRouter;
import org.matsim.core.router.TripStructureUtils;
import org.matsim.core.router.TripStructureUtils.Trip;
import org.matsim.core.utils.timing.TimeInterpretation;
import org.matsim.withinday.core.WithinDayReplanner;
import org.matsim.withinday.environment.AgentAssetInventory;
import org.matsim.withinday.environment.WithinDayObserver;
import org.matsim.withinday.environment.WithinDayRealTimeScoringEngine;
import org.matsim.withinday.networking.CommunicationManager;
import org.matsim.withinday.utils.StateEngine;
import org.matsim.withinday.utils.WithinDayAgentExperience;
import org.matsim.withinday.utils.WithinDayConfigGroup;

import modechoice.rl.java.utils.CustomConfigGroup;
import modechoice.rl.java.utils.CustomIterationEndReporting;
import modechoice.rl.java.utils.TrafficAnalysisZoneLoader;

import com.google.inject.Inject;

/**
 * Reinforcement Learning extension of WithinDayReplanner.
 * Overrides step() to compute step rewards, record experiences, and handle /feedback/score updates with Python.
 */
public class CustomRLReplanner extends WithinDayReplanner {

    protected final CustomConfigGroup customConfigGroup;
    protected final CommunicationManager communicationManager;

    private final Map<Id<Person>, Double> tripStartTimes = new ConcurrentHashMap<>();

    /**
     * Constructor for CustomRLReplanner.
     *
     * @param scenario                   The active MATSim simulation scenario.
     * @param router                     Trip router utility used by EditTrips.
     * @param timeInterpretation         Time interpretation rules for routing.
     * @param customObserver             Observer for extracting environment states and trip scores.
     * @param pythonCommunicationManager HTTP communication manager for external decision models.
     */
    @Inject
    public CustomRLReplanner(Scenario scenario, TripRouter router, TimeInterpretation timeInterpretation,
                              WithinDayObserver customObserver, CommunicationManager pythonCommunicationManager, 
                              CustomConfigGroup customConfigGroup) {
        super(scenario, router, timeInterpretation, customObserver);
        this.communicationManager = pythonCommunicationManager;
        this.customConfigGroup = customConfigGroup;
    }

    @Override
    public void initializeExternalModel() {

        if (this.customConfigGroup == null) return;

        // System Metadata
        Config config = scenario.getConfig();
        String outputDirectory = config.controller().getOutputDirectory();
        String absoluteOutputDirectory = new File(outputDirectory).getAbsolutePath();
        File fullPath = new File(absoluteOutputDirectory, WithinDayConfigGroup.getExternalModelFileName());

        String absoluteInputDirectory;
        try {
            absoluteInputDirectory = new File(config.getContext().toURI()).getParentFile().getAbsolutePath();
        } catch (URISyntaxException e) {
            throw new IllegalStateException(e);
        }
        String latestCheckpointPath = new File(absoluteInputDirectory, customConfigGroup.getLatestCheckpoint()).getAbsolutePath();

        System.out.println("Temporary Log: The saved checkpoint provided is " + latestCheckpointPath);

        Map<String, Object> jsonMap = new HashMap<>();
        jsonMap.put("modelType", customConfigGroup.getModelType());
        jsonMap.put("alpha", customConfigGroup.getAlpha());
        jsonMap.put("gamma", customConfigGroup.getGamma());
        jsonMap.put("epsilon", customConfigGroup.getEpsilon());
        jsonMap.put("epsilonDecay", customConfigGroup.getEpsilonDecay());
        jsonMap.put("epsilonMinimum", customConfigGroup.getEpsilonMinimum());
        jsonMap.put("trainingCutoffIteration", customConfigGroup.getTrainingCutoffIteration());
        //jsonMap.put("outputDirectory", scenario.getConfig().controller().getOutputDirectory());
        jsonMap.put("modes", AgentAssetInventory.getSimulationModesAsString());
        jsonMap.put("tourBasedModes", AgentAssetInventory.getSimulationTourBasedModesAsString());
        jsonMap.put("outputDirectory", absoluteOutputDirectory);
        jsonMap.put("modelFileName", fullPath.getAbsolutePath());
        jsonMap.put("checkpointModel", latestCheckpointPath);

        communicationManager.httpPost(gson.toJson(jsonMap), "/session/configure", 30, null);

        // Initialize the network traffic analysis zonal map
        String tazShapefileFullPath = "";
        try {
            String tazFileName = customConfigGroup.getTAZFileName();
            if (tazFileName != null && !tazFileName.isEmpty()) {
                tazShapefileFullPath = new File(absoluteInputDirectory, tazFileName).getAbsolutePath();

                TrafficAnalysisZoneLoader.loadZonesSortedByDistanceFromCenter(tazShapefileFullPath, "TAZ_ID", scenario);
            }
        } catch (Exception e) {
            throw new RuntimeException("Failed to resolve TAZ shapefile path from configuration.", e);
        }

    }

    @Override
    protected Map<String, Object> determineAction(MobsimAgent agent, Trip nextTrip, Map<String, Object> agentDemographics, Map<String, Object> stateObservation) {        
        boolean trainingFlag = isTrainingIteration((int) stateObservation.get("simulationIteration"));
        Plan modifiablePlan = WithinDayAgentUtils.getModifiablePlan(agent);

        stateObservation.put("simulationIteration", StateEngine.currentIteration);
        if (agentDemographics != null) {
            stateObservation.put("subpopulation", agentDemographics.getOrDefault("subpopulation", "default"));
        }

        //////
        double decisionTime = agent.getActivityEndTime();
        tripStartTimes.put(agent.getId(), decisionTime);

        if (log.isInfoEnabled() && stateObservation.containsKey("features")) {
            @SuppressWarnings("unchecked")
            Map<String, Object> features = (Map<String, Object>) stateObservation.get("features");
            Map<String, Object> rawState = (Map<String, Object>) features.get("rawStateObservation");
            int[] bitVector = (int[]) features.get("rawBitStateRepresentation");

            log.info("Temporary Log: DECISION_STATE_CHECK [Agent: {} | Iteration: {}] " +
                     "-> CurrentZone: {}, ScheduledZone: {}, DistanceBin: {}, TimeBin: {}, CongestionLevel: {}, AssetState: {} | BitVector: {}",
                agent.getId(),
                stateObservation.get("simulationIteration"),
                rawState.get("currentActivityTrafficZone"),
                rawState.get("scheduledActivityTrafficZone"),
                rawState.get("distanceBin"),
                rawState.get("discreteTimeBin"),
                rawState.get("congestionLevel"),
                rawState.get("assetState"),
                java.util.Arrays.toString(bitVector)
            );
        }
        ///////

        // Retrieve agent experience to check for pending rewards from the prior trip
        WithinDayAgentExperience experience = this.agentExperiences.computeIfAbsent(agent.getId(), 
            id -> new WithinDayAgentExperience(id, this.customObserver.getOrCreateScoringEngine(id))
        );

        List<String> experiencedModes = experience.getExperiencedModes();
        String PreviousMode = null;
        if (experiencedModes.size() != 0){
            PreviousMode = experiencedModes.get(experiencedModes.size() - 1);
        }

        stateObservation.put("previousMode", PreviousMode);

        // Request decision from serving model via HTTP POST
        String chosenMode = retrieveModeChoiceFromServingModel(stateObservation, trainingFlag);
        //chosenMode = "bike"; // OVER-RIDE VARIABLE
        
        // Update agent's executed plan in memory
        List<? extends PlanElement> newNextTrip = editTrips.replanFutureTrip(nextTrip, modifiablePlan, 
            chosenMode, agent.getActivityEndTime());
        
        // Update policy being used by the serving model via HTTP Post
        if (trainingFlag){
            double deltaQ = updatePolicyToServingModel(stateObservation);
            experience.setDeltaQ(deltaQ);
        }

        Map<String, Object> result = new HashMap<>();
        result.put("mode", chosenMode);
        result.put("newNextTrip", newNextTrip);
        return result;
    }

/**
     * Overridden step method calculating post-trip MATSim scores, step rewards, and posting feedback to Python.
     *
     * @param agent                  The MATSim agent completing the trip.
     * @param sim                    The active queue simulation instance.
     * @param simulationTime         Current simulation timestamp in seconds.
     * @param rescheduleActivityEndTime Boolean flag to reschedule activity end.
     */
    @Override
    public void step(MobsimAgent agent, QSim sim, double simulationTime, boolean rescheduleActivityEndTime) {
        double totalTripDistance = 0;
        double totalTripTravelTime = 0;
        int mainModeLegCount = 0;
        String currentModeUsed = "unknown";

        Map<String, Object> demographics = this.customObserver.getAgentDemographicRecord(agent);
        Id<Person> agentId = (Id<Person>) demographics.get("agentId");

        Trip completedTrip = getCompletedTrip(agent);
        
        if (completedTrip == null) return;

        Activity previousActivity = completedTrip.getOriginActivity();     
        List<Leg> legsInTrip = completedTrip.getLegsOnly();

        // --- EXECUTION TRACKING & LOGGING ---
        double actualDepartureTime = tripStartTimes.getOrDefault(agentId, simulationTime);
        double actualArrivalTime = simulationTime;
        double actualTravelTime = actualArrivalTime - actualDepartureTime;
        
        // Extract scheduled times if available from plans
        double scheduledArrivalTime = completedTrip.getDestinationActivity().getStartTime().isDefined() ? 
                completedTrip.getDestinationActivity().getStartTime().seconds() : actualArrivalTime;
        double scheduleError = actualArrivalTime - scheduledArrivalTime;

        log.info("DECISION_EXECUTION_OUTCOME [Agent: {} | Iteration: {}] -> ActualDep: {}s | ActualArr: {}s | ActualTT: {}s | SchedError: {}s",
            agentId,
            StateEngine.currentIteration,
            actualDepartureTime,
            actualArrivalTime,
            actualTravelTime,
            scheduleError
        );

        for (Leg leg : legsInTrip) {
            totalTripDistance += leg.getRoute().getDistance();
            totalTripTravelTime += leg.getTravelTime().orElse(0.0);

            if (!leg.getMode().contains("walk")) {
                mainModeLegCount++;
                currentModeUsed = leg.getMode();
            }
        }

        // Skipping the walk legs finds the main mode behind access and egress walking, but it leaves
        // a trip that is walking all the way without any mode. That never shows on a scenario whose
        // walking mode is named something else, such as Sioux Falls' "pedestrian"; on Open Berlin the
        // mode is "walk", so every walk trip would report "unknown" - into the asset inventory and
        // the step reward, not just into the tracking csv.
        if ("unknown".equals(currentModeUsed) && !legsInTrip.isEmpty()) {
            currentModeUsed = legsInTrip.getFirst().getMode();
        }

        int numberOfTransfers = Math.max(0, mainModeLegCount - 1);
        
        // REWARD PARAMETER COMPUTATION
        WithinDayRealTimeScoringEngine rewardCalculator = updateAndGetStepwiseScoringEngine(agent, previousActivity, currentModeUsed);
        double currentStepMatsimScore = rewardCalculator.getCurrentStepTripScore();
        double currentStepReward = rewardCalculator.getCurrentStepReward();

        WithinDayAgentExperience experience = this.agentExperiences.computeIfAbsent(
            agentId, 
            id -> new WithinDayAgentExperience(id, this.customObserver.getOrCreateScoringEngine(id))
        );

        experience.recordTrip(currentModeUsed, currentStepMatsimScore, currentStepReward);

        // --- NEXT STATE DATA ---
        Map<String, Object> nextState = this.customObserver.observeState(agent, sim, completedTrip, simulationTime, true);
        int iteration = (int) nextState.get("simulationIteration");
        boolean isEndOfDay = (boolean) nextState.get("endOfDayFlag");

        // 1. Build common payload structure for feedback/scoring
        Map<String, Object> payload = new HashMap<>();
        payload.put("agentID", agentId.toString());
        payload.put("reward", currentStepReward);
        payload.put("matsimScore", currentStepMatsimScore);
        payload.put("isTerminal", isEndOfDay);

        String jsonPayload = gson.toJson(payload);
        fireHttpCallback(jsonPayload, communicationManager.getRewardFeedbackEndpoint(), 360, agentId.toString());

        if (isEndOfDay) {
            if (isTrainingIteration(iteration)){
                double deltaQ = updatePolicyToServingModel(payload);
                experience.setDeltaQ(deltaQ);
            }
            
            customObserver.finalizeDay(agent, completedTrip.getDestinationActivity(), simulationTime);

            experience.finalizeDay(
                rewardCalculator.getAccumulatedDayReward(), 
                rewardCalculator.getAccumulatedDayScore()
            );
        }

        if (rescheduleActivityEndTime) {
            rescheduleActivityEnd(agent, sim, simulationTime, false);
        }
    }

    private List<Trip> getScheduledTrips(MobsimAgent agent){
        Plan executedPlan = WithinDayAgentUtils.getModifiablePlan(agent);
        List<Trip> trips = TripStructureUtils.getTrips(executedPlan);
        return trips;
    }

    private Trip getCompletedTrip(MobsimAgent agent){
        Trip completedTrip = null;

        for (Trip trip : getScheduledTrips(agent)) {
            // Match the trip whose destination is the current activity the agent just started
            if (trip.getDestinationActivity().equals(WithinDayAgentUtils.getCurrentPlanElement(agent))) {
                completedTrip = trip;
                break;
            }
        }

        return completedTrip;
    }

    private WithinDayRealTimeScoringEngine updateAndGetStepwiseScoringEngine(MobsimAgent agent, Activity previousActivity, String currentModeUsed){
        double modeRetrievalTime = 0.0;
        Id<Person> agentId = agent.getId();
        Id<Link> previousLinkId = previousActivity.getLinkId();
        Id<Link> currentLinkId = agent.getCurrentLinkId();
        Trip completedTrip = getCompletedTrip(agent);

        Map<String, Integer> modeDiscontinuityPenaltyMap = AgentAssetInventory.getModeDiscontinuityPenalty(agentId, previousLinkId);

        System.out.println("Temporary Log: The previous mode location at " + previousLinkId.toString() + " is: " + AgentAssetInventory.getModeLocation(agentId));

        // --- UPDATING INVENTORY ---
        AgentAssetInventory.updateModeLocation(agentId, currentLinkId, previousLinkId, currentModeUsed, modeDiscontinuityPenaltyMap);
        System.out.println("Temporary Log: The current mode location at " + currentLinkId.toString() + " is: " + AgentAssetInventory.getModeLocation(agentId));

        int currentTripIndex = getScheduledTrips(agent).indexOf(completedTrip);
        boolean isTour = AgentAssetInventory.getIsTourBased(agent.getId());

        if (isTour) modeRetrievalTime = AgentAssetInventory.getModeRetrievalTimes(agent, this.scenario, currentTripIndex, this.log);

        WithinDayRealTimeScoringEngine rewardCalculator = this.customObserver.tripEvaluationMetrics(
            agent, currentModeUsed, completedTrip, modeRetrievalTime, modeDiscontinuityPenaltyMap
        );

        return rewardCalculator;
    }

    private String retrieveModeChoiceFromServingModel(Map<String, Object> stateObservation, boolean trainingFlag){
        log.info("COMMUNICATION NET: Transmitting environment state for agent {}", stateObservation.get("agentID"));
        String jsonStateObservationPayload = gson.toJson(stateObservation);

        // Transmit state payload and query the external decision model endpoint via HTTP POST
        String rawModeResponse = fireHttpCallback(jsonStateObservationPayload, communicationManager.getDecisionEndpoint(), 360, (String) stateObservation.get("agentID"), trainingFlag);
        String chosenMode = (String) parseResponse(rawModeResponse);
        log.info("RL MODE CHOICE: Assigned mode '{}' to agent {} during {}", chosenMode.toUpperCase(), stateObservation.get("agentID"), trainingFlag?"train":"test");

        return chosenMode;
    }

    private double updatePolicyToServingModel(Map<String, Object> stateObservation){
        String jsonStateObservationPayload = gson.toJson(stateObservation);

        String updateResponse = fireHttpCallback(jsonStateObservationPayload, "/policy/update", 360, (String) stateObservation.get("agentID"));
        Double deltaQObject = (Double) parseResponse(updateResponse);

        double deltaQ = (deltaQObject != null) ? deltaQObject : 0.0;
        System.out.println("Temporary Log: The policy updated successfully with a delta Q of " + deltaQ);

        return deltaQ;
    }

    private String fireHttpCallback(String jsonPayload, String endpoint, int timeoutSeconds){
        return fireHttpCallback(jsonPayload, endpoint, timeoutSeconds, null, true);
    }

    private String fireHttpCallback(String jsonPayload, String endpoint, int timeoutSeconds, String agentIdString){
        return fireHttpCallback(jsonPayload, endpoint, timeoutSeconds, agentIdString, true);
    }

    private String fireHttpCallback(String jsonPayload, String endpoint, int timeoutSeconds, String agentIdString, boolean trainingFlag) {
        String response = null;

        if (agentIdString == null){
            response = communicationManager.httpPost(jsonPayload, endpoint, timeoutSeconds);
        }else if(trainingFlag){
            response = communicationManager.httpPost(jsonPayload, endpoint, timeoutSeconds, agentIdString);
        }else{
            response = communicationManager.httpPost(jsonPayload, endpoint, timeoutSeconds, agentIdString, "test");
        }
        
        if (response != null && !response.isEmpty()) {
            String trimmed = response.trim();
            
            // Safety check: If the response is a plain string instead of a JSON object, handle it or wrap it
            if (!trimmed.startsWith("{")) {
                log.warn("COMMUNICATION NET: Received non-object response from {} for agent {}: {}", endpoint, agentIdString, response);
                return response; 
            }

            try {
                Map<String, Object> responseMap = gson.fromJson(response, HashMap.class);

                if (responseMap == null) {
                    log.error("COMMUNICATION NET: Parsed response map was null for agent {}", agentIdString);
                    return null;
                }

                if (responseMap.containsKey("error")) {
                    log.error("COMMUNICATION NET: Backend error response for agent {}: {}", agentIdString, responseMap.get("error"));
                    return null;
                }

                return response; 

            } catch (Exception ex) {
                log.error("Failed to parse response JSON for agent {}: {}. Raw response was: {}", agentIdString, ex.getMessage(), response);
            }
        } else {
            log.error("COMMUNICATION NET: Null or empty response received from {} for agent {}", endpoint, agentIdString);
        }

        return null;
    }

    private Object parseResponse(String response) {
        Object parsedResponse = null;

        if (response != null && !response.isBlank()) {
            try {
                Map<String, Object> resMap = gson.fromJson(response, HashMap.class);
                if (resMap != null && resMap.containsKey("response")) {
                    Object val = resMap.get("response");
                    
                    if (val instanceof Number) {
                        parsedResponse = ((Number) val).doubleValue();
                    } else if (val != null) {
                        parsedResponse = val.toString();
                    }
                }
            } catch (Exception e) {
                log.error("Failed to parse JSON response: {}", response);
            }
        }
        return parsedResponse;
    }

    @Override
    protected boolean isTrainingIteration(int iteration){
        if ("test".equalsIgnoreCase(WithinDayConfigGroup.getDataSetType())) {return false;}
        
        int maxTrainingIteration = this.customConfigGroup.getTrainingCutoffIteration();
        int validationInterval = this.customConfigGroup.getValidationInterval();

        return iteration <= maxTrainingIteration && (iteration % validationInterval != 0 || iteration == 0);
    }

    @Override
    protected boolean isCheckpointInterval(int iteration){
        if (isTrainingIteration(iteration) && iteration % this.customConfigGroup.getSaveInterval() == 0){return true;}
        return false;
    }

    @Override
    public Map<Id<Person>, WithinDayAgentExperience> getAgentExperiences() {return this.agentExperiences;}

    @Override
    protected void handleIterationReporting(IterationEndsEvent event) {
        if (this.agentExperiences != null && !this.agentExperiences.isEmpty()) {
            CustomIterationEndReporting.writeAgentStatsCsv(event, this.agentExperiences);
        }
    }

    @Override
    public void reset(IterationEndsEvent event) {
        super.reset(event);
        AgentAssetInventory.reset();
        int iteration = event.getIteration();

        if (isCheckpointInterval(iteration)){
            fireHttpCallback("{}", communicationManager.getSaveEndPoint(iteration), 360);          
        }

        fireHttpCallback("{}", "/session/shared_policy/update", 360);
        fireHttpCallback("{}", communicationManager.getIterationResetEndPoint(), 360);

    }
}
