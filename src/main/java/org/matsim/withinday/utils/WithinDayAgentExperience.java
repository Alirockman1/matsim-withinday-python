package org.matsim.withinday.utils;

import java.util.ArrayList;
import java.util.List;
import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.population.Person;
import org.matsim.withinday.environment.WithinDayRealTimeScoringEngine;

/**
 * WithinDayAgentExperience acts as an experience tracking container for an 
 * individual agent during a simulation iteration. It records per-trip attributes 
 * such as experienced transport modes, step rewards, scores values, and manages 
 * pending transition states.
 */
public class WithinDayAgentExperience {

    private final Id<Person> agentId;
    private final WithinDayRealTimeScoringEngine scoringEngine;
    
    // Per-trip tracking collections
    private final List<String> experiencedModes = new ArrayList<>();
    private final List<Double> tripRewards = new ArrayList<>();
    private final List<Double> tripScores = new ArrayList<>();
    private final List<Double> stepDeltaQs = new ArrayList<>();
    
    // Daily cumulative totals
    private double accumulatedDeltaQ = 0.0;
    private double finalDayEndReward = 0.0;
    private double finalDayEndScore = 0.0;

    // Pending Transition State Fields
    private boolean hasPending = false;
    private double pendingReward = 0.0;
    private double pendingMatsimScore = 0.0;

    /**
     * Constructs an agent experience tracker instance for the specified person and scoring engine.
     *
     * @param agentId       The unique MATSim person ID of the agent.
     * @param scoringEngine The real-time scoring engine associated with this agent.
     */
    public WithinDayAgentExperience(Id<Person> agentId, WithinDayRealTimeScoringEngine scoringEngine) {
        this.agentId = agentId;
        this.scoringEngine = scoringEngine;
    }

    /**
     * Records a step delta-Q value and accumulates it into the daily total.
     *
     * @param deltaQ The delta-Q value to record.
     */
    public void setDeltaQ(double deltaQ){
        this.stepDeltaQs.add(deltaQ);
        this.accumulatedDeltaQ += deltaQ;
    }

    /** @return The unique MATSim person ID associated with this experience. */
    public Id<Person> getAgentId() { return agentId; }

    /** @return The real-time scoring engine instance. */
    public WithinDayRealTimeScoringEngine getScoringEngine() { return scoringEngine; }

    /** @return The list of transport modes experienced throughout the day. */
    public List<String> getExperiencedModes() { return experiencedModes; }

    /** @return The list of step rewards recorded per trip. */
    public List<Double> getTripRewards() { return tripRewards; }

    /** @return The list of trip scores recorded per trip. */
    public List<Double> getTripScores() { return tripScores; }

    /** @return The list of step delta-Q values recorded. */
    public List<Double> getStepDeltaQs() { return stepDeltaQs; }

    /** @return The cumulative delta-Q value accumulated today. */
    public double getAccumulatedDeltaQ() { return accumulatedDeltaQ; }

    /** @return The final day-end reward. */
    public double getFinalDayEndReward() { return finalDayEndReward; }

    /** @return The final day-end score. */
    public double getFinalDayEndScore() { return finalDayEndScore; }

    /** @return The pending reward value. */
    public double getPendingReward() { return pendingReward; }

    /** @return The pending score value. */
    public double getPendingMatsimScore() { return pendingMatsimScore; }

    /**
     * Records the metrics for a completed trip.
     *
     * @param mode   The transport mode utilized for the trip.
     * @param score  The computed trip score.
     * @param reward The computed step reward.
     */
    public void recordTrip(String mode, double score, double reward) {
        this.experiencedModes.add(mode);
        this.tripRewards.add(reward);
        this.tripScores.add(score);
    }

    /**
     * Finalizes the day's aggregate reward and score totals at the end of the simulation day.
     *
     * @param dayEndReward The final total daily reward accumulated.
     * @param dayEndScore  The final total daily score accumulated.
     */
    public void finalizeDay(double dayEndReward, double dayEndScore) {
        this.finalDayEndReward = dayEndReward;
        this.finalDayEndScore = dayEndScore;
    }

    /**
     * Stores a pending transition reward and MATSim score awaiting state resolution.
     *
     * @param reward      The pending step reward.
     * @param matsimScore The pending MATSim score.
     */
    public void storePendingReward(double reward, double matsimScore) {
        this.pendingReward = reward;
        this.pendingMatsimScore = matsimScore;
        this.hasPending = true;
    }

    /**
     * Checks whether there is an active pending reward awaiting processing.
     *
     * @return True if a pending reward exists, false otherwise.
     */
    public boolean hasPendingReward() {return this.hasPending;}

    /**
     * Clears and resets all pending reward and score transition fields.
     */
    public void clearPendingReward() {
        this.hasPending = false;
        this.pendingReward = 0.0;
        this.pendingMatsimScore = 0.0;
    }
}
