package org.matsim.withinday.utils;

import java.util.ArrayList;
import java.util.List;
import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.population.Person;
import org.matsim.withinday.environment.WithinDayRealTimeScoringEngine;

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

    public WithinDayAgentExperience(Id<Person> agentId, WithinDayRealTimeScoringEngine scoringEngine) {
        this.agentId = agentId;
        this.scoringEngine = scoringEngine;
    }

    // --- Record-keeping methods ---
    
    public void recordTrip(String mode, double score, double reward) {
        this.experiencedModes.add(mode);
        this.tripRewards.add(reward);
        this.tripScores.add(score);
    }

    public void finalizeDay(double dayEndReward, double dayEndScore) {
        this.finalDayEndReward = dayEndReward;
        this.finalDayEndScore = dayEndScore;
    }

    // --- Pending Reward Management Methods ---

    public void storePendingReward(double reward, double matsimScore) {
        this.pendingReward = reward;
        this.pendingMatsimScore = matsimScore;
        this.hasPending = true;
    }

    public boolean hasPendingReward() {
        return this.hasPending;
    }

    public void clearPendingReward() {
        this.hasPending = false;
        this.pendingReward = 0.0;
        this.pendingMatsimScore = 0.0;
    }

    // --- Setters ---
    public void setDeltaQ(double deltaQ){
        this.stepDeltaQs.add(deltaQ);
        this.accumulatedDeltaQ += deltaQ;
    }

    // --- Getters ---

    public Id<Person> getAgentId() { return agentId; }
    public WithinDayRealTimeScoringEngine getScoringEngine() { return scoringEngine; }
    public List<String> getExperiencedModes() { return experiencedModes; }
    public List<Double> getTripRewards() { return tripRewards; }
    public List<Double> getTripScores() { return tripScores; }
    public List<Double> getStepDeltaQs() { return stepDeltaQs; }
    public double getAccumulatedDeltaQ() { return accumulatedDeltaQ; }
    public double getFinalDayEndReward() { return finalDayEndReward; }
    public double getFinalDayEndScore() { return finalDayEndScore; }

    public double getPendingReward() { return pendingReward; }
    public double getPendingMatsimScore() { return pendingMatsimScore; }
}
