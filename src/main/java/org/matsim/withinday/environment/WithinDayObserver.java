package org.matsim.withinday.environment;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.logging.log4j.Logger;
import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.Scenario;
import org.matsim.api.core.v01.population.Activity;
import org.matsim.api.core.v01.population.Person;
import org.matsim.api.core.v01.population.Plan;
import org.matsim.core.mobsim.framework.MobsimAgent;
import org.matsim.core.mobsim.qsim.QSim;
import org.matsim.core.mobsim.qsim.agents.WithinDayAgentUtils;
import org.matsim.core.router.TripStructureUtils;
import org.matsim.core.router.TripStructureUtils.Trip;
import org.matsim.core.scoring.functions.ScoringParametersForPerson;
import org.matsim.withinday.core.AgentSelector;

import com.google.inject.Inject;

/**
 * WithinDayObserver returns a snapshot of the matsim environment as a state representation.
 * The class is modeled such that custom observers can be built on top
 * while automatically inheriting cached demographic profiling and step-reward scoring.
 */
public abstract class WithinDayObserver {

    protected final Scenario scenario;
    protected final Logger log;

    public final Map<Id<Person>, WithinDayRealTimeScoringEngine> agentRewardCalculators = new HashMap<>();
    private final Map<Id<Person>, Map<String, Object>> agentDemographicRegistry = new HashMap<>();
    
    @Inject protected MatsimScoreTracker scoreTracker;
    @Inject protected ScoringParametersForPerson scoringParametersForPerson;
    @Inject protected AgentSelector agentSelector;

    /**
     * Initializes the within-day observer with the simulation scenario and logger.
     *
     * @param scenario The MATSim scenario instance.
     * @param log      The logger instance for logging observer events.
     */
    public WithinDayObserver(Scenario scenario, Logger log) {
        this.scenario = scenario;
        this.log = log;
    }

    /**
     * Extracts agent core demographic parameters (saves the parameter for a single iteration)
     * 
     * @param agent The MATSim simulation agent being evaluated.
     * @return A Dictionary containing: agent_id, subpopulation, sex, and age_group.
     */
    public Map<String, Object> getAgentDemographicRecord(MobsimAgent agent) {
        Id<Person> agentId = agent.getId();

        if (agentDemographicRegistry.containsKey(agentId)) {
            return agentDemographicRegistry.get(agentId);
        }

        Person person = scenario.getPopulation().getPersons().get(agentId);
        Map<String, Object> agentProfile = new HashMap<>();

        if (person != null) {
            String subpopulation = (String) person.getAttributes().getAttribute("subpopulation");
            String sex = (String) person.getAttributes().getAttribute("sex");
            Object rawAge = person.getAttributes().getAttribute("age");

            agentProfile.put("agentId", agentId);
            agentProfile.put("subpopulation", subpopulation != null ? subpopulation : "default");
            agentProfile.put("sex", sex != null ? sex.trim().toLowerCase() : "unknown");
            
            // Discretize the raw age attribute using the internal package strategy rules
            agentProfile.put("ageGroup", StateEngine.discretizeAgeAttribute(rawAge));
        } else {
            log.warn("Agent '{}' not found in population.", agentId);
        }

        agentDemographicRegistry.put(agentId, agentProfile);
        return agentProfile;
    }

    /**
     * Retrieves or creates a new piecewise scoring engine for a given agent.
     *
     * @param agentId Unique agent-specific ID.
     * @return The piecewise scoring engine instance.
     */
    public abstract WithinDayRealTimeScoringEngine getOrCreateScoringEngine(Id<Person> agentId);

    /**
     * Provides a custom configuration group to subclasses if applicable.
     *
     * @return The custom group object, or null by default.
     */
    public Object getCustomConfigGroup(){return null;};

    /**
     * Retrieves the scoring engine for a specific agent ID if it exists.
     *
     * @param agentId The MATSim person ID.
     * @return The WithinDayRealTimeScoringEngine instance, or null if not yet created.
     */
    public WithinDayRealTimeScoringEngine getScoringEngine(Id<Person> agentId) {return this.agentRewardCalculators.get(agentId);}

    /**
     * Retrieves the scoring engine directly from a MobsimAgent.
     *
     * @param agent The MATSim simulation agent.
     * @return The WithinDayRealTimeScoringEngine instance, or null if not yet created.
     */
    public WithinDayRealTimeScoringEngine getScoringEngine(MobsimAgent agent) {return agent != null ? getScoringEngine(agent.getId()) : null;}

    /**
     * Retrieves a unique integer tag/ID for the agent via the agent selector.
     *
     * @param agent The MATSim simulation agent.
     * @return An integer representing the agent's unique tag.
     */
    public int getUniqueAgentId(MobsimAgent agent){
        String agentIdString = agent.getId().toString();
        return agentSelector.getAgentTag(agentIdString);
    }

    /**
     * Queries the environment representation for selected agents to capture state parameters.
     *
     * @param agent             The MATSim simulation agent being evaluated.
     * @param sim               The MATSim simulation engine.
     * @param trip              The future trip the agent is scheduled to undertake.
     * @param currentTime       The simulation time at which the observer is called.
     * @param isNextTripContext Flag indicating whether the evaluation is for the next trip context.
     * @return A map containing the observed state parameters. 
     */
    public abstract Map<String, Object> observeState(MobsimAgent agent, QSim sim, Trip trip, double currentTime, boolean isNextTripContext);

    /**
     * Extract the MATSim score accumulated for this agent since the previous trip.
     * Reading it consumes the read pointer.
     *
     * @param agentId The MATSim person ID.
     * @return The current trip's incremental score value.
     */
    public double consumeMatsimStepScore(Id<Person> agentId) {
        return this.scoreTracker.consumeStepScore(agentId);
    }

    /**
     * Compiles immediate execution step utilities directly with the underlying RealTimeScoringEngine.
     *
     * @param agent        The MATSim simulation agent.
     * @param executedMode The transport mode executed in the step.
     * @param trip         The current trip being evaluated.
     * @param context      Additional optional context parameters.
     * @return The WithinDayRealTimeScoringEngine instance.
     */
    public WithinDayRealTimeScoringEngine tripEvaluationMetrics(MobsimAgent agent, String executedMode, Trip trip, Object... context) {
        WithinDayRealTimeScoringEngine rewardCalculator = getOrCreateScoringEngine(agent.getId());
        rewardCalculator.compute(agent, consumeMatsimStepScore(agent.getId()), executedMode, trip, context);
        return rewardCalculator;
    }
    
    /**
     * Determines whether the agent has reached the end of their daily plan.
     *
     * @param agent             The MATSim simulation agent.
     * @param trip              The current trip being evaluated.
     * @param isNextTripContext Flag indicating whether checking against the next trip context.
     * @return A map containing a boolean completion flag and the target trip object.
     */
    protected Map<String, Object> endOfDay(MobsimAgent agent, Trip trip, boolean isNextTripContext){
        Plan executedPlan = WithinDayAgentUtils.getModifiablePlan(agent);
        List<Trip> allTrips = TripStructureUtils.getTrips(executedPlan);
        int targetTripIndex = allTrips.indexOf(trip);

        if (isNextTripContext) {targetTripIndex += 1;}

        boolean isEndOfDay = (targetTripIndex >= allTrips.size());
        Trip targetTrip = !isEndOfDay ? allTrips.get(targetTripIndex) : null;

        Map<String, Object> result = new HashMap<>();
        result.put("flag", isEndOfDay);
        result.put("targetTrip", targetTrip);
        return result;
    }

    /**
     * Extract the final MATSim score (including overnight activity and daily mode constants).
     *
     * @param agent        The MATSim simulation agent.
     * @param finalActivity The final activity reached by the agent.
     * @param arrivalTime  The simulation timestamp of the arrival.
     * @return The updated WithinDayRealTimeScoringEngine instance.
     */
    public WithinDayRealTimeScoringEngine finalizeDay(MobsimAgent agent, Activity finalActivity, double arrivalTime) {
        WithinDayRealTimeScoringEngine rewardCalculator = getOrCreateScoringEngine(agent.getId());
        rewardCalculator.finalizeDay(this.scoreTracker.finishDay(agent.getId(), finalActivity, arrivalTime));
        return rewardCalculator;
    }

    /**
     * Resets internal observation registries and resets every active scoring engine at the start of each iteration.
     */
    public void reset() {
        for (WithinDayRealTimeScoringEngine engine : this.agentRewardCalculators.values()) {
            if (engine != null) {engine.reset();}
        }

        this.agentDemographicRegistry.clear();
        this.agentRewardCalculators.clear();
    }
}