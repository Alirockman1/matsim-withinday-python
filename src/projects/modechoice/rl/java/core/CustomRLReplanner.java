package modechoice.rl.java.core;

import java.io.File;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.Scenario;
import org.matsim.api.core.v01.network.Link;
import org.matsim.api.core.v01.population.Activity;
import org.matsim.api.core.v01.population.Leg;
import org.matsim.api.core.v01.population.Person;
import org.matsim.api.core.v01.population.Plan;
import org.matsim.api.core.v01.population.PlanElement;
import org.matsim.contrib.analysis.vsp.qgis.QGisConstants.pointLayerSymbol;
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
import org.matsim.withinday.environment.StateEngine;
import org.matsim.withinday.environment.WithinDayObserver;
import org.matsim.withinday.environment.WithinDayRealTimeScoringEngine;
import org.matsim.withinday.networking.CommunicationManager;
import org.matsim.withinday.utils.WithinDayAgentExperience;
import org.matsim.withinday.utils.WithinDayConfigGroup;

import modechoice.rl.java.utils.CustomConfigGroup;
import modechoice.rl.java.utils.CustomIterationEndReporting;

import com.google.gson.Gson;
import com.google.inject.Inject;

/**
 * Reinforcement Learning extension of WithinDayReplanner.
 * Overrides step() to compute step rewards, record experiences, and handle /feedback/score updates with Python.
 */
public class CustomRLReplanner extends WithinDayReplanner {

    protected final CustomConfigGroup customConfigGroup;

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
                              WithinDayObserver customObserver, CommunicationManager pythonCommunicationManager, CustomConfigGroup customConfigGroup) {
        super(scenario, router, timeInterpretation, customObserver, pythonCommunicationManager);
        this.customConfigGroup = customConfigGroup;
    }

    @Override
    public void initializeExternalModel() {

        if (this.customConfigGroup == null) return;

        // System Metadata
        String outputDirectory = scenario.getConfig().controller().getOutputDirectory();
        String absoluteOutputDirectory = new File(outputDirectory).getAbsolutePath();
        File fullPath = new File(absoluteOutputDirectory, WithinDayConfigGroup.getExternalModelFileName());

        Map<String, Object> jsonMap = new HashMap<>();
        jsonMap.put("modelType", customConfigGroup.getModelType());
        jsonMap.put("alpha", customConfigGroup.getAlpha());
        jsonMap.put("gamma", customConfigGroup.getGamma());
        jsonMap.put("epsilon", customConfigGroup.getEpsilon());
        jsonMap.put("epsilonDecay", customConfigGroup.getEpsilonDecay());
        jsonMap.put("epsilonMinimum", customConfigGroup.getEpsilonMinimum());
        jsonMap.put("trainingCutoffIteration", customConfigGroup.getTrainingCutoffIteration());
        jsonMap.put("outputDirectory", scenario.getConfig().controller().getOutputDirectory());
        jsonMap.put("modes", AgentAssetInventory.getSimulationModesAsString());
        jsonMap.put("tourBasedModes", AgentAssetInventory.getSimulationTourBasedModesAsString());
        jsonMap.put("outputDirectory", absoluteOutputDirectory);
        jsonMap.put("saveInterval", customConfigGroup.getSaveInterval());
        jsonMap.put("modelFileName", fullPath.getAbsolutePath());

        communicationManager.httpPost(gson.toJson(jsonMap), "/session/configure", 30);
    }

    @Override
    protected Map<String, Object> determineAction(MobsimAgent agent, Trip nextTrip, Map<String, Object> agentDemographics, Map<String, Object> stateObservation) {
        Map<String, Object> payload = new HashMap<>();
        Plan modifiablePlan = WithinDayAgentUtils.getModifiablePlan(agent);
        Id<Person> agentId = agent.getId();

        stateObservation.put("simulationIteration", StateEngine.currentIteration);
        if (agentDemographics != null) {
            stateObservation.put("subpopulation", agentDemographics.getOrDefault("subpopulation", "default"));
        }

        // Retrieve agent experience to check for pending rewards from the prior trip
        WithinDayAgentExperience experience = this.agentExperiences.computeIfAbsent(agentId, 
            id -> new WithinDayAgentExperience(id, this.customObserver.getOrCreateScoringEngine(id))
        );

        // Request decision from serving model via HTTP POST
        log.info("COMMUNICATION NET: Transmitting environment state for agent {}", agentId.toString());
        String jsonStateObservationPayload = gson.toJson(stateObservation);

        // Transmit state payload and query the external decision model endpoint via HTTP POST
        String rawModeResponse = fireHttpCallback(jsonStateObservationPayload, getDecisionEndpoint(), 360, agentId);
        String chosenMode = (String) parseResponse(rawModeResponse);
        if (chosenMode==null){
            chosenMode = "pedestrian";
        }

        log.info("RL MODE CHOICE: Assigned mode '{}' to agent {}", chosenMode.toUpperCase(), agentId.toString());

        // Update agent's executed plan in memory
        List<? extends PlanElement> newNextTrip = editTrips.replanFutureTrip(nextTrip, modifiablePlan, 
            chosenMode, agent.getActivityEndTime());

        // Update the Q values based on agent experience
        String updateResponse = fireHttpCallback(jsonStateObservationPayload, "/policy/update", 360, agentId);

        Double deltaQ = (Double) parseResponse(updateResponse);
        
        if (deltaQ==null)deltaQ = 0.0;
        experience.setDeltaQ(deltaQ);

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

        for (Leg leg : legsInTrip) {
            totalTripDistance += leg.getRoute().getDistance();
            totalTripTravelTime += leg.getTravelTime().orElse(0.0);
            
            if (!leg.getMode().contains("walk")) {
                mainModeLegCount++;
                currentModeUsed = leg.getMode();
            }
        }

        int numberOfTransfers = Math.max(0, mainModeLegCount - 1);
        
        // REWARD PARAMETER COMPUTATION
        WithinDayRealTimeScoringEngine rewardCalculator = updateAndGetStepwiseScoringEngine(agent, previousActivity, currentModeUsed);
        double currentStepMatsimScore = rewardCalculator.getCurrentStepTripScore();
        double currentStepReward = rewardCalculator.getCurrentStepReward();
        System.out.println("Temporary Logs: The current step matsim score is " + currentStepMatsimScore);

        WithinDayAgentExperience experience = this.agentExperiences.computeIfAbsent(
            agentId, 
            id -> new WithinDayAgentExperience(id, this.customObserver.getOrCreateScoringEngine(id))
        );

        experience.recordTrip(currentModeUsed, currentStepMatsimScore, currentStepReward);

        // --- NEXT STATE DATA ---
        Map<String, Object> nextState = this.customObserver.observeState(agent, sim, completedTrip, simulationTime, true);
        boolean isEndOfDay = (boolean) nextState.get("endOfDayFlag");

        // 1. Build common payload structure for feedback/scoring
        Map<String, Object> payload = new HashMap<>();
        payload.put("agentID", agentId.toString());
        payload.put("reward", currentStepReward);
        payload.put("matsimScore", currentStepMatsimScore);
        payload.put("isTerminal", isEndOfDay);

        String jsonPayload = gson.toJson(payload);
        fireHttpCallback(jsonPayload, getRewardFeedbackEndpoint(), 360, agentId);

        if (isEndOfDay) {
            String updateResponse = fireHttpCallback(jsonPayload, "/policy/update", 360, agentId);
            System.out.println(updateResponse);
            Double deltaQ = (Double) parseResponse(updateResponse);
            
            if (deltaQ==null)deltaQ = 0.0;
            experience.setDeltaQ(deltaQ);

            System.out.println("RL MODE CHOICE: The policy updated successfully at the last leg with a delta Q of " + deltaQ);
            
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

        System.out.println("The previous mode location at " + previousLinkId.toString() + " is: " + AgentAssetInventory.getModeLocation(agentId));

        // --- UPDATING INVENTORY ---
        AgentAssetInventory.updateModeLocation(agentId, currentLinkId, previousLinkId, currentModeUsed, modeDiscontinuityPenaltyMap);
        System.out.println("The current mode location at " + currentLinkId.toString() + " is: " + AgentAssetInventory.getModeLocation(agentId));

        int currentTripIndex = getScheduledTrips(agent).indexOf(completedTrip);
        boolean isTour = AgentAssetInventory.getIsTourBased(agent.getId());

        if (isTour) modeRetrievalTime = AgentAssetInventory.getModeRetrievalTimes(agent, this.scenario, currentTripIndex, this.log);

        WithinDayRealTimeScoringEngine rewardCalculator = this.customObserver.tripEvaluationMetrics(
            agent, currentModeUsed, completedTrip, modeRetrievalTime, modeDiscontinuityPenaltyMap
        );

        return rewardCalculator;
    }

    private String fireHttpCallback(String jsonPayload, String endpoint, int timeoutSeconds, Id<Person> agentId) {
        String response = communicationManager.httpPost(jsonPayload, endpoint, timeoutSeconds);

        if (response != null && !response.isEmpty()) {
            String trimmed = response.trim();
            
            // Safety check: If the response is a plain string instead of a JSON object, handle it or wrap it
            if (!trimmed.startsWith("{")) {
                log.warn("COMMUNICATION NET: Received non-object response from {} for agent {}: {}", endpoint, agentId, response);
                return response; 
            }

            try {
                Map<String, Object> responseMap = gson.fromJson(response, HashMap.class);

                if (responseMap == null) {
                    log.error("COMMUNICATION NET: Parsed response map was null for agent {}", agentId);
                    return null;
                }

                if (responseMap.containsKey("error")) {
                    log.error("COMMUNICATION NET: Backend error response for agent {}: {}", agentId, responseMap.get("error"));
                    return null;
                }

                return response; 

            } catch (Exception ex) {
                log.error("Failed to parse response JSON for agent {}: {}. Raw response was: {}", agentId, ex.getMessage(), response);
            }
        } else {
            log.error("COMMUNICATION NET: Null or empty response received from {} for agent {}", endpoint, agentId);
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
    public Map<Id<Person>, WithinDayAgentExperience> getAgentExperiences() {
        return this.agentExperiences;
    }

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

        fireHttpCallback("{}", "/reset-iteration-memory", 360, null);
    }
}
