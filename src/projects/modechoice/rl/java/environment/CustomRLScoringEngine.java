package modechoice.rl.java.environment;

import org.matsim.api.core.v01.Scenario;
import org.matsim.api.core.v01.TransportMode;
import org.matsim.api.core.v01.population.Leg;
import org.matsim.core.config.groups.RoutingConfigGroup.TeleportedModeParams;
import org.matsim.core.mobsim.framework.MobsimAgent;
import org.matsim.core.router.TripStructureUtils.Trip;
import org.matsim.core.scoring.functions.ModeUtilityParameters;
import org.matsim.core.scoring.functions.ScoringParameters;
import org.matsim.withinday.environment.WithinDayRealTimeScoringEngine;
import org.matsim.withinday.environment.WithinDayObserver;

import modechoice.rl.java.utils.CustomConfigGroup;

import java.util.List;
import java.util.Map;

/**
 * Reinforcement-learning specific scoring engine extending the baseline 
 * {@link WithinDayRealTimeScoringEngine} to incorporate domain-specific penalties, 
 * including mode discontinuity penalties and abandoned vehicle retrieval costs.
 */
public class CustomRLScoringEngine extends WithinDayRealTimeScoringEngine {

    /**
     * Constructs a new reinforcement learning scoring engine instance.
     *
     * @param scenario          The active MATSim simulation scenario.
     * @param observer          The within-day observer handling custom configuration metrics.
     * @param scoringParameters Parsed subpopulation-specific scoring parameters.
     */
    public CustomRLScoringEngine(Scenario scenario, WithinDayObserver observer, ScoringParameters scoringParameters) {
        super(scenario, observer, scoringParameters);
    }

    /**
     * Computes additional penalties specific to reinforcement learning (mode discontinuity and retrieval costs)
     * by unpacking context variables and weighting them via custom configuration parameters.
     *
     * @param agent   The mobsim agent being evaluated.
     * @param mode    The main transport mode used for the current trip.
     * @param trip    The completed trip data structure used for counterfactual comparisons.
     * @param context Optional parameters passed from the environment (expects retrieval time as Double and penalty map as Map).
     * @return The total weighted penalty value for the current step.
     */
    @Override
    protected double calculateAdditionalPenalties(MobsimAgent agent, String mode, Trip trip, Object... context) {
        double modeRetrievalTime = 0.0;
        Map<String, Integer> modeDiscontinuityPenaltyMap = null;

        if (context.length > 0 && context[0] instanceof Double) {
            modeRetrievalTime = (Double) context[0];
        }
        if (context.length > 1 && context[1] instanceof Map) {
            @SuppressWarnings("unchecked")
            Map<String, Integer> map = (Map<String, Integer>) context[1];
            modeDiscontinuityPenaltyMap = map;
        }

        double discontinuityPenalty = (modeDiscontinuityPenaltyMap != null) 
                ? calculateModeDiscontinuityPenalty(modeDiscontinuityPenaltyMap, mode, trip.getLegsOnly()) 
                : 0.0;

        double retrievalPenalty = calculateAbandonedModePenalty(modeRetrievalTime);

        CustomConfigGroup customConfigGroup = (CustomConfigGroup) this.localObserver.getCustomConfigGroup();
        double retrievalCostWeight = customConfigGroup.getModelWeights().getOrDefault("retrievalCostPenalty", 1.0);

        return Math.abs(discontinuityPenalty + (retrievalCostWeight * retrievalPenalty));
    }

    /**
     * Calculates the penalty for choosing a mode whose vehicle was not present at the origin,
     * comparing the actual trip disutility against the worst available legal alternative.
     *
     * @param modeDiscontinuityPenaltyMap Map defining modal availability (0 for available, 1 for unavailable).
     * @param legsInTrip                  List of legs comprising the completed trip.
     * @return The absolute calculated discontinuity penalty.
     */
    private double calculateModeDiscontinuityPenalty(Map<String, Integer> modeDiscontinuityPenaltyMap, String lastActiveMode, List<Leg> legsInTrip) {
        Integer usedModeUnavailable = modeDiscontinuityPenaltyMap.get(lastActiveMode);

        if (usedModeUnavailable == null || usedModeUnavailable != 1) return 0.0;
        
        double bestLegalTripDisutility = Double.MAX_VALUE;
        boolean foundLegalMode = false;

        for (Map.Entry<String, Integer> entry : modeDiscontinuityPenaltyMap.entrySet()) {
            if (entry.getValue() != 0) continue;

            ModeUtilityParameters legalModeParams = this.scoringParameters.modeParams.get(entry.getKey());
            if (legalModeParams == null) continue;

            double prospectiveTripDisutility = tripDisutility(legsInTrip, legalModeParams);
            if (prospectiveTripDisutility < bestLegalTripDisutility) {
                bestLegalTripDisutility = prospectiveTripDisutility;
                foundLegalMode = true;
            }
        }

        // Fallback safety penalty if no legal modes are found in the configuration map
        if (!foundLegalMode) return 100.0;

        // Compute the actual disutility of the restricted/unavailable mode used
        ModeUtilityParameters usedModeParams = this.scoringParameters.modeParams.get(this.lastActiveMode);
        double actualTripDisutility = (usedModeParams != null) 
                ? tripDisutility(legsInTrip, usedModeParams) 
                : bestLegalTripDisutility;

        // Proportional Cost Markup with Adaptive Weighting Penalty
        CustomConfigGroup customConfigGroup = (CustomConfigGroup) this.localObserver.getCustomConfigGroup();
        double costMarkupAlpha = customConfigGroup.getModelWeights().getOrDefault("discontinuityPenalty", 1.0);

        // Scales proportionally with the best legal alternative to prevent short/long trip distortions
        double proportionalBasePenalty = costMarkupAlpha * bestLegalTripDisutility;
        double relativeRegret = Math.max(0.0, actualTripDisutility - bestLegalTripDisutility);
        double modeDiscontinuityPenalty = proportionalBasePenalty + relativeRegret;
        
        return modeDiscontinuityPenalty;
    }

    /**
     * Evaluates the standard Charypar-Nagel travel disutility for a collection of legs given specific mode parameters.
     *
     * @param legsInTrip List of legs to evaluate.
     * @param modeParams The utility parameters associated with the target transport mode.
     * @return The total computed trip disutility.
     */
    private double tripDisutility(List<Leg> legsInTrip, ModeUtilityParameters modeParams) {
        double distanceBeta = modeParams.marginalUtilityOfDistance_m
                + (modeParams.monetaryDistanceCostRate * this.scoringParameters.marginalUtilityOfMoney);

        double tripDisutility = modeParams.constant;

        for (Leg leg : legsInTrip) {
            tripDisutility += leg.getTravelTime().orElse(0.0) * modeParams.marginalUtilityOfTraveling_s;

            if (leg.getRoute() != null) {
                double distance = leg.getRoute().getDistance();
                if (!Double.isNaN(distance)) {
                    tripDisutility += distance * distanceBeta;
                }
            }
        }

        return tripDisutility;
    }

    /**
     * Computes the cost associated with walking back (worst option) to retrieve a vehicle left stranded at a previous location.
     *
     * @param modeRetrievalTime The duration in seconds required to retrieve the vehicle.
     * @return The total penalty incurred for vehicle retrieval.
     */
    private double calculateAbandonedModePenalty(double modeRetrievalTime) {
        // Factor to stimulate agent fatigue while walking
        double fatigueFactor = 1.2;
        
        if (modeRetrievalTime <= 0.0) return 0.0;

        // Fetch walking parameters
        ModeUtilityParameters walkParams = this.scoringParameters.modeParams.get(TransportMode.walk);
        double betaDistanceDefaultMode = walkParams != null ? walkParams.marginalUtilityOfDistance_m : 0.0;
        double betaTimeDefaultMode = walkParams != null ? walkParams.marginalUtilityOfTraveling_s : 0.0;

        var teleportedModeParams = super.localScenario.getConfig().routing().getTeleportedModeParams();
        double teleportedSpeed = 1.11; // Default fallback (~4 km/h in m/s)
        if (teleportedModeParams != null && teleportedModeParams.get(TransportMode.walk) != null) {
            Double configuredSpeed = teleportedModeParams.get(TransportMode.walk).getTeleportedModeSpeed();
            if (configuredSpeed != null) {
                teleportedSpeed = configuredSpeed;
            }
        }

        double retrievalCostPerSecond = Math.abs((teleportedSpeed * betaDistanceDefaultMode)
            + betaTimeDefaultMode + (2.0 * this.scoringParameters.marginalUtilityOfPerforming_s));

        double linearRetrievalCost = modeRetrievalTime * retrievalCostPerSecond;
        double nonLinearRetrievalPenalty = Math.pow(linearRetrievalCost, fatigueFactor);

        return nonLinearRetrievalPenalty;
    }

    /**
     * Resets the internal state variables and tracking histories between simulation iterations.
     */
    @Override
    public void reset() {
        super.reset();
    }
}
