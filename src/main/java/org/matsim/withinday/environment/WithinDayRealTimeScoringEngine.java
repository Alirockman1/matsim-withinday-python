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
 * WithinDayRealTimeScoringEngine serves as the core evaluation and reward calculation engine 
 * for individual simulation agents. It bridges native MATSim step utilities with custom 
 * reinforcement learning penalties and rewards in real time.
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

    /**
     * Inisiates a new real-time scoring engine instance for a specific agent context.
     *
     * @param scenario          The MATSim simulation scenario.
     * @param observer          The parent within-day observer instance managing this engine.
     * @param scoringParameters The individual agent's scoring parameter configuration.
     */
    public WithinDayRealTimeScoringEngine(Scenario scenario, WithinDayObserver observer, ScoringParameters scoringParameters) {
        this.localScenario = scenario;
        this.localObserver = observer;
        this.scoringParameters = scoringParameters;
    }

    /**
     * Retrieves the raw MATSim score accumulated during the current simulation step.
     *
     * @return The step trip score.
     */
    public double getCurrentStepTripScore() { return this.currentStepTripScore; }

    /**
     * Retrieves the custom penalty applied to the MATSim score during the current simulation step.
     *
     * @return The step penalty value.
     */
    public double getCurrentStepPenalty() { return this.currentStepPenalty; }

    /**
     * Retrieves the net reward computed for the current simulation step (trip score minus penalty).
     *
     * @return The net step reward.
     */
    public double getCurrentStepReward() { return this.currentStepReward; }

    /**
     * Retrieves the cumulative MATSim score accumulated by the agent so far in the trip.
     *
     * @return The cumulative day score.
     */
    public double getAccumulatedDayScore() { return this.accumulatedDayScore; }

    /**
     * Retrieves the cumulative net reward accumulated by the agent so far today.
     *
     * @return The cumulative day reward.
     */
    public double getAccumulatedDayReward() { return this.accumulatedDayReward; }

    /**
     * Computes the step-wise reward for a single completed trip. By default, the reward 
     * matches the raw MATSim step score unless custom penalties are implemented by subclasses.
     *
     * @param agent           The MOBsim simulation agent being evaluated.
     * @param matsimStepScore The utility accumulated by MATSim since the previous trip (comprising 
     *                        the origin activity utility plus the completed trip utility).
     * @param mode            The primary transport mode actually executed by the agent.
     * @param trip            The completed trip object, utilized for counterfactual mode comparisons.
     * @param context         Optional variable arguments allowing subclasses to pass project-specific 
     *                        data (such as spatial maps, states, or lookup times) required for custom penalties.
     */
    public void compute(MobsimAgent agent, double matsimStepScore, String mode, Trip trip,
                        Object... context) {
        this.allModesUsedToday.add(mode);
        this.lastActiveMode = mode;

        this.currentStepTripScore = matsimStepScore;
        this.accumulatedDayScore += matsimStepScore;

        this.currentStepPenalty = calculateAdditionalPenalties(agent, mode, trip, context);

        this.currentStepReward = this.currentStepTripScore - this.currentStepPenalty;
        this.accumulatedDayReward += this.currentStepReward;

        logResults(agent);
    }

    /**
     * Finalizes the daily score aggregation by integrating the final MATSim score component, 
     * which includes overnight activity utility and daily mode constants.
     * 
     * @param matsimDayScore The total final utility accumulated by MATSim throughout the entire day.
     */
    public void finalizeDay(double matsimDayScore) {
        double dayEndScore = matsimDayScore - this.accumulatedDayScore;
        this.accumulatedDayScore = matsimDayScore;
        this.accumulatedDayReward += dayEndScore;
    }

    /**
     * Computes additional custom penalties to be deducted from the base MATSim score. 
     * Defaults to zero; intended to be overridden by subclasses for specific penalty models.
     * 
     * @param agent   The MOBsim simulation agent being evaluated.
     * @param mode    The primary transport mode actually executed by the agent.
     * @param trip    The completed trip object.
     * @param context Optional variable arguments for custom calculations.
     * @return The computed penalty value (non-negative).
     */
    protected double calculateAdditionalPenalties(MobsimAgent agent, String mode, Trip trip, Object... context) {return 0.0;}

    /**
     * Logs the current step metrics for debugging and auditing purposes.
     * 
     * @param agent The MOBsim simulation agent.
     */
    protected void logResults(MobsimAgent agent) {
        String agentId = agent.getId().toString();
        localLogger.info(String.format("SCORE: Agent %s | Step Score: %.4f | Penalty: %.4f | Reward: %.4f", 
                agentId, this.currentStepTripScore, this.currentStepPenalty, this.currentStepReward));
    }

    /**
     * Resets all internal history trackers, daily accumulators, and step cache variables 
     * at the beginning of a new simulation iteration.
     */
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
}
