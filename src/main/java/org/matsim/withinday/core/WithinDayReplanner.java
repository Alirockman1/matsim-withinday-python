package org.matsim.withinday.core;

import com.google.gson.Gson;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.Scenario;
import org.matsim.api.core.v01.population.Activity;
import org.matsim.api.core.v01.population.Leg;
import org.matsim.api.core.v01.population.Person;
import org.matsim.api.core.v01.population.Plan;
import org.matsim.api.core.v01.population.PlanElement;
import org.matsim.core.controler.events.IterationEndsEvent;
import org.matsim.core.mobsim.framework.MobsimAgent;
import org.matsim.core.mobsim.qsim.QSim;
import org.matsim.core.mobsim.qsim.agents.WithinDayAgentUtils;
import org.matsim.core.router.TripRouter;
import org.matsim.core.router.TripStructureUtils;
import org.matsim.core.router.TripStructureUtils.Trip;
import org.matsim.core.utils.timing.TimeInterpretation;
import org.matsim.withinday.environment.AgentAssetInventory;
import org.matsim.withinday.environment.StateEngine;
import org.matsim.withinday.environment.WithinDayObserver;
import org.matsim.withinday.environment.WithinDayRealTimeScoringEngine;
import org.matsim.withinday.networking.CommunicationManager;
import org.matsim.withinday.utils.EditTrips;
import org.matsim.withinday.utils.IterationEndReporting;
import org.matsim.withinday.utils.WithinDayAgentExperience;
import org.matsim.withinday.utils.WithinDayConfigGroup;

/**
 * <h2>WithinDayReplanner</h2>
 * <p>
 * Generic base class for all within-day replanning strategies.
 * Handles observer integration, state extraction, and network transmission to Python serving models.
 * </p>
 */
public abstract class WithinDayReplanner {

    protected static final Logger log = LogManager.getLogger(WithinDayReplanner.class);

    protected final Scenario scenario;
    protected final TripRouter router;
    protected final TimeInterpretation timeInterpretation;
    protected final WithinDayObserver customObserver;
    protected final CommunicationManager communicationManager;
    protected final Gson gson = new Gson();
    protected final Map<Id<Person>, WithinDayAgentExperience> agentExperiences = new HashMap<>();

    protected EditTrips editTrips;

    /**
     * Constructor for WithinDayReplanner.
     *
     * @param scenario                   The active MATSim simulation scenario.
     * @param router                     Trip router utility used by EditTrips.
     * @param timeInterpretation         Time interpretation rules for routing.
     * @param customObserver             Observer for extracting environment states and trip scores.
     * @param pythonCommunicationManager HTTP communication manager for external decision models.
     */
    public WithinDayReplanner(Scenario scenario, TripRouter router, TimeInterpretation timeInterpretation,
                              WithinDayObserver customObserver, CommunicationManager pythonCommunicationManager) {
        this.scenario = scenario;
        this.router = router;
        this.timeInterpretation = timeInterpretation;
        this.customObserver = customObserver;
        this.communicationManager = pythonCommunicationManager;

        AgentAssetInventory.setSimulationBasedModes(scenario);
        StateEngine.setNetworkCentroid(scenario.getNetwork());
    }

    /**
     * Method to initialize model configurations or remote endpoint sessions during startup.
     */
    public abstract void initializeExternalModel();

    /**
     * The decision to make once the within-day objective is reached. Can be extended for route choice or mode choice.
     *
     * @param agent              The simulation agent.
     * @param nextTrip           The upcoming trip structure.
     * @param agentDemographics  Demographic attributes of the agent.
     * @param stateObservation   Environmental state observations.
     * @return                   A map containing the selected action / mode and payload.
     */
    protected abstract Map<String, Object> determineAction(MobsimAgent agent, Trip nextTrip, Map<String, Object> agentDemographics, Map<String, Object> stateObservation);

    /**
     * Optional accessor for agent experience tracking metrics.
     *
     * @return Map mapping Person IDs to experience records, or null if tracking is disabled.
     */
    public abstract Map<Id<Person>, WithinDayAgentExperience> getAgentExperiences();

    /**
     * Replans the agent's next trip based on current state observations and external action determination.
     *
     * @param agent          The MATSim simulation agent undergoing replanning.
     * @param sim            The active queue simulation instance.
     * @param simulationTime Current simulation timestamp in seconds from midnight.
     */
    public void replanNextTrip(MobsimAgent agent, QSim sim, double simulationTime) {
        // Validate that the current plan element is an Activity
        PlanElement currentElement = WithinDayAgentUtils.getCurrentPlanElement(agent);
        if (!(currentElement instanceof Activity)) {
            throw new IllegalStateException(String.format(
                "Expected current plan element for agent '%s' to be an Activity, but found '%s'.",
                agent.getId(), currentElement != null ? currentElement.getClass().getSimpleName() : "null"
            ));
        }

        Activity currentActivity = (Activity) currentElement;
        Plan modifiablePlan = WithinDayAgentUtils.getModifiablePlan(agent);

        // Extract upcoming trip structures
        int currentPlanElementIndex = WithinDayAgentUtils.getCurrentPlanElementIndex(agent);
        Trip unmodifiedNextTrip = EditTrips.findTripAtPlanElementIndex(agent, currentPlanElementIndex + 1);
        Trip nextTripLeg = TripStructureUtils.findTripStartingAtActivity(currentActivity, modifiablePlan);

        if (nextTripLeg == null || unmodifiedNextTrip == null) return;

        // Observe environmental state and demographics
        Map<String, Object> demographics = this.customObserver.getAgentDemographicRecord(agent);
        Map<String, Object> state = this.customObserver.observeState(agent, sim, nextTripLeg, simulationTime, false);
        
        log.info("Replanning next trip for agent {} at activity end time {}", demographics.get("agentId").toString(), agent.getActivityEndTime());

        Map<String, Object> action = determineAction(agent, unmodifiedNextTrip, demographics, state);
        String chosenMode = (String) action.get("mode");

        if (sim.getScenario().getConfig().qsim().getMainModes().contains(chosenMode)) {
            WithinDayAgentUtils.addVehicleToQSimIfNecessary(
                (List<? extends PlanElement>) action.get("newNextTrip"), 
                scenario, sim
            );
        }
    }

    /**
     * Processes post-trip simulation step feedback.
     * Can be overridden by concrete subclasses for custom reward processing.
     *
     * @param agent                     The MATSim simulation agent completing the trip.
     * @param sim                       The active queue simulation instance.
     * @param simulationTime            Current simulation timestamp in seconds from midnight.
     * @param rescheduleActivityEndTime Flag indicating whether to reschedule activity end.
     */
    public void step(MobsimAgent agent, QSim sim, double simulationTime, boolean rescheduleActivityEndTime) {
        Map<String, Object> demographics = this.customObserver.getAgentDemographicRecord(agent);
        Id<Person> agentId = (Id<Person>) demographics.get("agentId");

        // Create agent experience database
        WithinDayAgentExperience experience = this.agentExperiences.computeIfAbsent(
            agentId, 
            id -> new WithinDayAgentExperience(id, this.customObserver.getOrCreateScoringEngine(id))
        );

        // Get trip metrics from completed trip
        Trip completedTrip = findCompletedTrip(agent);
        if (completedTrip == null) return;

        Map<String, Object> metrics = extractTripMetrics(completedTrip);
        String currentModeUsed = (String) metrics.get("mode");

        // Get the trip Kai-nagel score
        WithinDayRealTimeScoringEngine rewardCalculator = this.customObserver.tripEvaluationMetrics(
            agent, currentModeUsed, completedTrip);
        double currentStepMatsimScore = rewardCalculator.getCurrentStepTripScore();

        Map<String, Object> nextState = this.customObserver.observeState(agent, sim, completedTrip, simulationTime, true);
        if ((boolean) nextState.get("endOfDayFlag")){
            this.customObserver.finalizeDay(agent, completedTrip.getDestinationActivity(), simulationTime);
            experience.finalizeDay(rewardCalculator.getAccumulatedDayReward(), rewardCalculator.getAccumulatedDayScore());
        }

        experience.recordTrip(currentModeUsed, currentStepMatsimScore, 0);
        
        if (rescheduleActivityEndTime){
            rescheduleActivityEnd(agent, sim, simulationTime, false);
        }
    }

    /**
     * Reschedules the active activity's end time in the QSim queue.
     * Preserves original duration for flexible activities, or fixed end times for rigid activities.
     *
     * @param agent             The MATSim simulation agent.
     * @param sim               The active queue simulation instance.
     * @param activityStartTime The actual timestamp (in seconds) when the agent started this activity.
     */
    public void rescheduleActivityEnd(MobsimAgent agent, QSim sim, double activityStartTime, boolean isFlexible) {
        PlanElement currentElement = WithinDayAgentUtils.getCurrentPlanElement(agent);
        if (!(currentElement instanceof Activity)) {
            log.warn("Cannot reschedule activity end for agent {}: current element is not an Activity.", agent.getId());
            return;
        }

        Activity currentActivity = (Activity) currentElement;
        int currentElementIndex = WithinDayAgentUtils.getCurrentPlanElementIndex(agent);

        // Fetch original unmodifiable activity from initial selected plan
        Person person = WithinDayAgentUtils.getModifiablePlan(agent).getPerson();
        Plan originalPlan = person.getSelectedPlan();
        Activity originalActivity = (Activity) originalPlan.getPlanElements().get(currentElementIndex);

        double oldEndTime = currentActivity.getEndTime().orElse(-1.0);
        double targetEndTime = oldEndTime;

        if (isFlexible) {
            if (originalActivity.getEndTime().isDefined() && originalActivity.getStartTime().isDefined()) {
                double plannedDuration = originalActivity.getEndTime().seconds() - originalActivity.getStartTime().seconds();
                targetEndTime = activityStartTime + Math.max(0.0, plannedDuration);
            } else if (originalActivity.getMaximumDuration().isDefined()) {
                targetEndTime = activityStartTime + originalActivity.getMaximumDuration().seconds();
            } 
        } else {
            if (originalActivity.getEndTime().isDefined()) {
                targetEndTime = originalActivity.getEndTime().seconds();
            }
        }

        // Apply update in QSim queue
        currentActivity.setEndTime(targetEndTime);
        WithinDayAgentUtils.rescheduleActivityEnd(agent, sim);

        log.info("RESCHEDULE ACTIVITY ({}) : Agent {} end time updated from {}s to {}s (Actual Start: {}s)",
                isFlexible ? "FLEXIBLE" : "FIXED", agent.getId(), oldEndTime, targetEndTime, activityStartTime);
    }

    /**
     * Determines whether the current simulation iteration is designated for model training.
     * 
     * @param iteration The current simulation iteration index.
     * @return True if training is active for this iteration, false otherwise.
     */
    protected boolean isTrainingIteration(int iteration) {return true;}

    /**
     * Determines whether model state checkpoints or weights should be saved during this iteration.
     * 
     * @param iteration The current simulation iteration index.
     * @return True if a checkpoint should be saved, false otherwise.
     */
    protected boolean isCheckpointInterval(int iteration){return false;}

    /**
     * Identifies and retrieves the most recently completed trip for a given simulation agent 
     * by comparing destination activities against the agent's current plan position.
     * 
     * @param agent The MATSim simulation agent.
     * @return The completed {@link Trip} object, or null if no matching trip is found.
     */
    protected Trip findCompletedTrip(MobsimAgent agent) {
        Plan executedPlan = WithinDayAgentUtils.getModifiablePlan(agent);
        List<Trip> trips = TripStructureUtils.getTrips(executedPlan);

        for (Trip trip : trips) {
            if (trip.getDestinationActivity().equals(WithinDayAgentUtils.getCurrentPlanElement(agent))) {return trip;}
        }
        return null;
    }

    /**
     * Extracts operational performance metrics (distance, travel time, active mode, and transfers) 
     * from a completed trip leg structure.
     * 
     * @param completedTrip The completed trip instance to analyze.
     * @return A map containing aggregated trip metrics.
     */
    protected Map<String, Object> extractTripMetrics(Trip completedTrip) {
        double totalTripDistance = 0;
        double totalTripTravelTime = 0;
        String currentModeUsed = "unknown";
        
        List<Leg> legsInTrip = completedTrip.getLegsOnly();
        for (Leg leg : legsInTrip) {
            totalTripDistance += leg.getRoute().getDistance();
            totalTripTravelTime += leg.getTravelTime().orElse(0.0);
            if (!leg.getMode().contains("walk")) {currentModeUsed = leg.getMode();}
        }

        Map<String, Object> metrics = new HashMap<>();
        metrics.put("distance", totalTripDistance);
        metrics.put("travelTime", totalTripTravelTime);
        metrics.put("mode", currentModeUsed);
        metrics.put("transfers", Math.max(0, completedTrip.getLegsOnly().size() - 2)); // Example transfer logic
        return metrics;
    }

    /**
     * Default iteration reporting hook. Subclasses can override this method to route 
     * performance statistics to custom reporting engines or standard loggers.
     * 
     * @param event The iteration ends event containing iteration metadata.
     */
    protected void handleIterationReporting(IterationEndsEvent event) {
        if (this.agentExperiences != null && !this.agentExperiences.isEmpty()) {
            IterationEndReporting.writeAgentStatsCsv(event, this.agentExperiences);
        }
    }

    /**
     * Utility used for modifying upcoming agent itineraries prior to activity completion.
     */
    public void initEditTrips() {
        // lazy instantiation of EditTrips
        if (editTrips == null) {
            editTrips = new EditTrips(router, scenario, null, timeInterpretation);
        }
    }

    /**
     * Resets internal replanner states, clears experience caches, and triggers end-of-iteration 
     * reporting procedures.
     * 
     * @param event The iteration ends event containing current iteration parameters.
     */
    public void reset(IterationEndsEvent event) {
        int iteration = event.getIteration();

        if (this.customObserver != null) {this.customObserver.reset();}

        handleIterationReporting(event);
        this.agentExperiences.clear();
        log.info("RL PLANNER: Agent experiences cleared for iteration {}", iteration);
    }
}
