package org.matsim.withinday.environment;

import java.util.HashSet;
import java.util.Set;

import org.matsim.api.core.v01.Scenario;
import org.matsim.core.mobsim.framework.MobsimAgent;
import org.matsim.core.router.TripStructureUtils.Trip;
import org.matsim.core.scoring.functions.ScoringParameters;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Baseline class to attain a desired trip Utility value for a single agent by combining 
 * native MATSim step scores with custom penalties.
 */
public class WithinDayRealTimeScoringEngine {

    protected static final Logger localLogger = LogManager.getLogger(WithinDayRealTimeScoringEngine.class);

    protected final Scenario localScenario;
    protected final WithinDayObserver localObserver;
    protected final ScoringParameters scoringParameters;

    protected final Set<String> allModesUsedToday = new HashSet<>();
    protected String lastActiveMode = null;
    protected double accumulatedDayScore = 0.0;
    protected double accumulatedDayReward = 0.0;

    protected double currentStepTripScore;
    protected double currentStepPenalty;
    protected double currentStepReward;

    public WithinDayRealTimeScoringEngine(Scenario scenario, WithinDayObserver observer, ScoringParameters scoringParameters) {
        this.localScenario = scenario;
        this.localObserver = observer;
        this.scoringParameters = scoringParameters;
    }

    /**
     * Computes the reward for a single completed trip. By default, the reward 
     * equals the raw MATSim step score unless custom penalties are implemented in subclasses.
     *
     * @param agent                     mobsim agent
     * @param matsimStepScore           utility MATSim accumulated since the previous trip, i.e. the
     *                                  origin activity plus the trip that was just completed.
     * @param mode                      main mode the agent actually used.
     * @param trip                      the completed trip, used for the counterfactual mode comparison.
     * @param context                   variable arguments that allow subclasses to pass project-specific data (such as maps, states, or retrieval times)
     *                                  needed for custom penalty calculations
     */
    public void compute(MobsimAgent agent, double matsimStepScore, String mode, Trip trip,
                        Object... context) {
        this.allModesUsedToday.add(mode);
        this.lastActiveMode = mode;

        this.currentStepTripScore = matsimStepScore;
        this.accumulatedDayScore += matsimStepScore;

        // Compute the total penalty to be applied
        this.currentStepPenalty = calculateAdditionalPenalties(agent, mode, trip, context);

        this.currentStepReward = this.currentStepTripScore - this.currentStepPenalty;
        this.accumulatedDayReward += this.currentStepReward;

        logResults(agent);
    }

    /**
     * Closes the day with the agent's final MATSim score, which additionally contains the overnight
     * activity utility and the daily mode constants. 
     * 
     * @param matsimDayScore utility MATSim accumulated throughout the day
     */
    public void finalizeDay(double matsimDayScore) {
        double dayEndScore = matsimDayScore - this.accumulatedDayScore;

        this.accumulatedDayScore = matsimDayScore;
        this.accumulatedDayReward += dayEndScore;
    }

    /**
     * Computes the total penalty to be applied on the original MATSim score. Defaults to 0.
     * 
     * @param agent     mobsim agent
     * @param mode      main mode the agent actually used.
     * @param trip      the completed trip, used for the counterfactual mode comparison.
     * @param context   variable arguments that allow subclasses to pass project-specific data (such as maps, states, or retrieval times)
     *                  needed for custom penalty calculations
     */
    protected double calculateAdditionalPenalties(MobsimAgent agent, String mode, Trip trip, Object... context) {
        return 0.0;
    }

    protected void logResults(MobsimAgent agent) {
        String agentId = agent.getId().toString();
        localLogger.info(String.format("SCORE: Agent %s | Step Score: %.4f | Penalty: %.4f | Reward: %.4f", 
                agentId, this.currentStepTripScore, this.currentStepPenalty, this.currentStepReward));
    }

    public void reset() {
        // Clear history trackers
        this.allModesUsedToday.clear();
        this.lastActiveMode = null;

        // Reset daily cumulative trackers
        this.accumulatedDayScore = 0.0;
        this.accumulatedDayReward = 0.0;

        // Reset step cache variables
        this.currentStepTripScore = 0.0;
        this.currentStepPenalty = 0.0;
        this.currentStepReward = 0.0;
    }

    // GET step-wise rewards
    public double getCurrentStepTripScore() { return this.currentStepTripScore; }
    public double getCurrentStepPenalty() { return this.currentStepPenalty; }
    public double getCurrentStepReward() { return this.currentStepReward; }

    // GET cumalative rewards
    public double getAccumulatedDayScore() { return this.accumulatedDayScore; }
    public double getAccumulatedDayReward() { return this.accumulatedDayReward; }

}
