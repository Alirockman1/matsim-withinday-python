package modechoice.rl.java.core;

import java.lang.Thread.State;
import java.net.URL;
import java.nio.file.Paths;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.NavigableMap;
import java.util.TreeMap;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.Scenario;
import org.matsim.api.core.v01.population.Activity;
import org.matsim.api.core.v01.population.Person;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigGroup;
import org.matsim.core.mobsim.framework.MobsimAgent;
import org.matsim.core.mobsim.qsim.QSim;
import org.matsim.core.router.TripRouter;
import org.matsim.core.router.TripStructureUtils.Trip;
import org.matsim.withinday.environment.AgentAssetInventory;
import org.matsim.withinday.environment.Congestion.Estimate;
import org.matsim.withinday.environment.WithinDayObserver;
import org.matsim.withinday.environment.WithinDayRealTimeScoringEngine;
import org.matsim.withinday.utils.StateEngine;
import org.matsim.withinday.utils.StateEngine.GridPosition;

import modechoice.rl.java.environment.CustomRLScoringEngine;
import modechoice.rl.java.utils.CustomConfigGroup;
import modechoice.rl.java.utils.TrafficAnalysisZoneLoader;

import com.google.inject.Inject;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

public class CustomRLObserver extends WithinDayObserver {
    private static final Logger log = LogManager.getLogger(CustomRLObserver.class);
    private static final NavigableMap<Double, Integer> CUSTOM_TIME_BIN_LOOKUP = new TreeMap<>();
    private final Config config;
    private OrtEnvironment onnxEnvironment;
    private OrtSession autoEncoderSession;
    private String inputNodeName;

    /**
     * Maps high-fidelity continuous clock seconds into discrete temporal partitions.
     * Peak rush hours are allocated tightly, while off-peak intervals are compressed.
     * * @see Stevanovic, A., Stevanovic, J., & Kergaye, C. (2009) "Optimization of 
     * time-of-day traffic signal plans based on dynamic evolution of traffic patterns."
    */
    static {
        CUSTOM_TIME_BIN_LOOKUP.put(4.0, 0); CUSTOM_TIME_BIN_LOOKUP.put(6.0, 1); CUSTOM_TIME_BIN_LOOKUP.put(7.0, 2); 
        CUSTOM_TIME_BIN_LOOKUP.put(7.5, 3); CUSTOM_TIME_BIN_LOOKUP.put(8.0, 4); CUSTOM_TIME_BIN_LOOKUP.put(8.5, 5); 
        CUSTOM_TIME_BIN_LOOKUP.put(9.0, 6); CUSTOM_TIME_BIN_LOOKUP.put(10.0, 7); CUSTOM_TIME_BIN_LOOKUP.put(12.0, 8); 
        CUSTOM_TIME_BIN_LOOKUP.put(14.0, 9); CUSTOM_TIME_BIN_LOOKUP.put(15.5, 10); CUSTOM_TIME_BIN_LOOKUP.put(16.5, 11); 
        CUSTOM_TIME_BIN_LOOKUP.put(17.5, 12); CUSTOM_TIME_BIN_LOOKUP.put(18.5, 13); CUSTOM_TIME_BIN_LOOKUP.put(20.0, 14); 
        CUSTOM_TIME_BIN_LOOKUP.put(22.0, 15);
    }

    @Inject private CustomConfigGroup customConfigGroup;
    @Inject private TripRouter router;
    
    @Inject
    public CustomRLObserver(Scenario scenario) {
        super(scenario, log);
        this.config = scenario.getConfig();
        StateEngine.setCustomTimeBinLookup(CUSTOM_TIME_BIN_LOOKUP);

        CustomConfigGroup customConfigGroup = (CustomConfigGroup) scenario.getConfig().getModule(CustomConfigGroup.GROUP_NAME);

        if (customConfigGroup != null) {
            String encoderModelPath = customConfigGroup.getEncoderModel();

            if (encoderModelPath != null && !encoderModelPath.trim().isEmpty()) {
                initializeAutoEncoder(scenario.getConfig().getContext(), encoderModelPath);
            } else {
                log.info("No AutoEncoder model specified in config. Skipping ONNX initialization.");
            }
        }
    }

    /**
     * Overrides the baseline scoring engine creator to instantiate the 
     * project-specific CustomRLScoringEngine.
     */
    @Override
    public WithinDayRealTimeScoringEngine getOrCreateScoringEngine(Id<Person> agentId) {
        return this.agentRewardCalculators.computeIfAbsent(agentId, id -> {
            Person person = this.scenario.getPopulation().getPersons().get(id);
            return new CustomRLScoringEngine(this.scenario, this, this.scoringParametersForPerson.getScoringParameters(person));
        });
    }

    /**
     * Overrides the baseline observation state construction.
     */
    @Override
    public Map<String, Object> observeState(MobsimAgent agent, QSim sim, Trip trip, double timeInSeconds, boolean isNextTripContext) {
        Map<String, Object> features = new HashMap<>();
        
        double departureTimeSeconds = isNextTripContext 
            ? StateEngine.getPredictedDepartureTime(trip.getDestinationActivity(), timeInSeconds).seconds() : timeInSeconds;

        Map<String, Object> endOfDayResult = endOfDay(agent, trip, isNextTripContext);
        boolean isEndOfDay = (boolean) endOfDayResult.get("flag");
        Trip targetTrip = (Trip) endOfDayResult.get("targetTrip");

        // Feature Extraction Stage
        Map<String, Object> rawStateSpace = extractRawStateFeatures(agent, sim, trip, targetTrip, departureTimeSeconds, isEndOfDay);

        // Encode to a binary from raw discrete feature space
        int[] rawBitStateSpace = constructBitStateVector(rawStateSpace, isEndOfDay);

        // Compress to latent space (Optional)
        float[] latentBitStateSpace = null;
        int[] latentBitStateSpaceInt = null;
        if (!isEndOfDay && hasAutoEncoder()) {
            latentBitStateSpace = encodeBitStateWithONNX(rawBitStateSpace);
            if (latentBitStateSpace != null) {
                latentBitStateSpaceInt = new int[latentBitStateSpace.length];
                for (int i = 0; i < latentBitStateSpace.length; i++) {
                    latentBitStateSpaceInt[i] = Math.round(latentBitStateSpace[i]);
                }
            }
            features.put("latentBitStateRepresentation", latentBitStateSpaceInt);
        }

        // Encapsulation
        Map<String, Object> observation = new HashMap<>();
        observation.put("simulationIteration", StateEngine.currentIteration);
        observation.put("agentID", agent.getId().toString());
        observation.put("possibleModeSet", new ArrayList<>(AgentAssetInventory.getSimulationModes()));
        observation.put("endOfDayFlag", isEndOfDay);

        features.put("rawStateObservation", rawStateSpace);
        features.put("rawBitStateRepresentation", rawBitStateSpace);

        observation.put("features", features);
        return observation;
    }

    /**
     * Extracts all raw spatial, temporal, and environment state variables for an agent step.
     * 
     * @param agent                The active agent instance.
     * @param sim                  The active simulation instance.
     * @param trip                 The current trip context.
     * @param targetTrip           The target trip context evaluated for end-of-day logic.
     * @param departureTimeSeconds The calculated departure timestamp in seconds.
     * @param isEndOfDay           Flag indicating if the agent has completed all scheduled trips.
     * 
     * @return Map containing all extracted state key-value pairs.
     */
    private Map<String, Object> extractRawStateFeatures(MobsimAgent agent, QSim sim, Trip trip, Trip targetTrip, 
            double departureTimeSeconds, boolean isEndOfDay) {
        
        Map<String, Object> state = new HashMap<>();
        Person person = scenario.getPopulation().getPersons().get(agent.getId());
        Activity currentActivity = trip.getOriginActivity();

        // Origin Spatial Indexing
        Map<String, Object> currentActivityLocation = StateEngine.getActivityLocation(currentActivity);
        int currentZoneBin = TrafficAnalysisZoneLoader.getZoneBin(
            (double) currentActivityLocation.get("x"), 
            (double) currentActivityLocation.get("y")
        );

        // Asset Availability
        int assetState = StateEngine.getAssetStateOnHand(agent.getId(), agent.getCurrentLinkId());
        

        state.put("currentActivityTrafficZone", currentZoneBin);
        state.put("assetState", assetState);
        state.put("isEndOfDay", isEndOfDay);

        if (!isEndOfDay && trip.getDestinationActivity() != null) {
            Activity nextActivity = trip.getDestinationActivity();
            Map<String, Object> nextActivityLocation = StateEngine.getActivityLocation(nextActivity);
            
            int nextZoneBin = TrafficAnalysisZoneLoader.getZoneBin(
                (double) nextActivityLocation.get("x"), 
                (double) nextActivityLocation.get("y")
            );

            // Spatial and Congestion Metrics
            double beelineDistance = StateEngine.estimatePreRoutingDistance(currentActivity, nextActivity);
            int distanceBin = StateEngine.descritizeDistanceFromContinuous(beelineDistance);
            int discreteTimeBin = StateEngine.discretizeTimeFromContinous(departureTimeSeconds, "demand_based");

            double congestionMetric = 0.0;
            Estimate congestionEstimate = StateEngine.estimatePreRoutingCongestion(
                sim, scenario, router, person, currentActivity, nextActivity, departureTimeSeconds
            );
            
            if (congestionEstimate.hasData()) {
                congestionMetric = congestionEstimate.occupancy();
            }
            int congestionLevel = StateEngine.discretizeCongestionFromContinuous(congestionMetric);

            state.put("scheduledActivityTrafficZone", nextZoneBin);
            state.put("distanceToCoverInMeters", beelineDistance);
            state.put("distanceBin", distanceBin);
            state.put("departureTimeSeconds", departureTimeSeconds);
            state.put("discreteTimeBin", discreteTimeBin);
            state.put("congestionMetric", congestionMetric);
            state.put("congestionLevel", congestionLevel);
        }

        return state;
    }

    /**
     * Converts a raw feature map into a sequential binary bit-array representation.
     * 
     * @param rawState   Map of extracted raw feature variables.
     * @param isEndOfDay Flag indicating if end-of-day zero-padding should be applied.
     * @return Formatted integer array containing binary state representation.
     */
    private int[] constructBitStateVector(Map<String, Object> rawState, boolean isEndOfDay) {
        int currentZone = (int) rawState.get("currentActivityTrafficZone");
        int assetState = (int) rawState.get("assetState");

        int[] currentZoneBits = StateEngine.convertToBitStateRepresentation(currentZone, StateEngine.getZoneBinSize());
        int[] assetBits = StateEngine.convertToBitStateRepresentation(assetState, StateEngine.getAssetBinSize());

        if (isEndOfDay) {
            int[] emptyNextZoneBits = new int[currentZoneBits.length];
            int[] timeBits = StateEngine.convertToBitStateRepresentation(0, StateEngine.getTimeBinSize("demand_based"));
            int[] distanceBits = StateEngine.convertToBitStateRepresentation(0, StateEngine.getDistanceBinSize());
            int[] congestionBits = StateEngine.convertToBitStateRepresentation(0, StateEngine.getCongestionBinSize());

            return assembleVector(currentZoneBits, emptyNextZoneBits, timeBits, distanceBits, assetBits, congestionBits);
        } else {
            int nextZone = (int) rawState.get("scheduledActivityTrafficZone");
            int timeBin = (int) rawState.get("discreteTimeBin");
            int distanceBin = (int) rawState.get("distanceBin");
            int congestionLevel = (int) rawState.get("congestionLevel");

            int[] nextZoneBits = StateEngine.convertToBitStateRepresentation(nextZone, StateEngine.getZoneBinSize());
            int[] timeBits = StateEngine.convertToBitStateRepresentation(timeBin, StateEngine.getTimeBinSize("demand_based"));
            int[] distanceBits = StateEngine.convertToBitStateRepresentation(distanceBin, StateEngine.getDistanceBinSize());
            int[] congestionBits = StateEngine.convertToBitStateRepresentation(congestionLevel, StateEngine.getCongestionBinSize());

            return assembleVector(currentZoneBits, nextZoneBits, timeBits, distanceBits, assetBits, congestionBits);
        }
    }

    // AUTO-ENCODER (OPTIONAL) //

    /**
     * Method to initialize the ONNX session once.
     */
    private void initializeAutoEncoder(URL configContext, String encoderModelPath) {
        try {
            URL absoluteModelUrl = ConfigGroup.getInputFileURL(configContext, encoderModelPath);
            String finalModelPath = Paths.get(absoluteModelUrl.toURI()).toAbsolutePath().toString();

            // Create ONNX Environment and Session
            this.onnxEnvironment = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            this.autoEncoderSession = this.onnxEnvironment.createSession(finalModelPath, options);
            
            // Fetch input node name dynamically
            if (!this.autoEncoderSession.getInputNames().isEmpty()) {
                this.inputNodeName = this.autoEncoderSession.getInputNames().iterator().next();
            } else {
                this.inputNodeName = "input_136";
            }
            
            log.info("Successfully initialized ONNX AutoEncoder using input node: {}", this.inputNodeName);

        } catch (Exception e) {
            log.error("Failed to load ONNX AutoEncoder model from path: " + encoderModelPath, e);
            this.autoEncoderSession = null;
        }
    }

    /**
     * Encodes the raw int[] bit vector directly into the latent space float vector.
     */
    public float[] encodeBitStateWithONNX(int[] rawBitState) {
        if (!hasAutoEncoder() || rawBitState == null || rawBitState.length == 0) {
            return null;
        }

        try {
            int length = rawBitState.length;
            float[] floatInput = new float[length];

            // Convert int bits to float tensor elements directly
            for (int i = 0; i < length; i++) {
                floatInput[i] = (float) rawBitState[i];
            }

            long[] inputShape = new long[]{1, length};
            FloatBuffer floatBuffer = FloatBuffer.wrap(floatInput);

            try (OnnxTensor inputTensor = OnnxTensor.createTensor(this.onnxEnvironment, floatBuffer, inputShape)) {
                try (OrtSession.Result results = this.autoEncoderSession.run(Collections.singletonMap(this.inputNodeName, inputTensor))) {
                    float[][] outputMatrix = (float[][]) results.get(0).getValue();
                    return outputMatrix[0];
                }catch (Exception e) {
                    e.printStackTrace();
                    return null;
                }
            }
        } catch (Exception e) {
            log.error("Error during ONNX AutoEncoder transform: " + e.getMessage(), e);
            return null;
        }
    }

    // HELPER FUNCTION //

    /**
     * Helper method to concatenate individual feature bit arrays into a single sequential vector.
     */
    private int[] assembleVector(int[]... bitGroups) {
        int totalLength = 0;
        for (int[] group : bitGroups) {totalLength += group.length;}

        int[] vector = new int[totalLength];
        int offset = 0;
        for (int[] group : bitGroups) {offset = appendBits(vector, offset, group);}

        return vector;
    }

    /**
     * Helper function to append arrays of bits into a destination bit vector starting at the specified offset.
     * <p>
     * Uses {@link System#arraycopy} for native memory copying performance without allocating 
     * intermediate objects.
     * </p>
     *
     * @param dest   The target binary state array being populated.
     * @param offset The current write position index in the destination array.
     * @param src    The source bit array to copy (e.g., position bits, time bits, or asset bits).
     * @return The updated offset index pointing to the next available position in {@code dest}.
     */
    private int appendBits(int[] dest, int offset, int[] src) {
        System.arraycopy(src, 0, dest, offset, src.length);
        return offset + src.length;
    }

    /**
    * Optional helper to check if ONNX is active
    */ 
    public boolean hasAutoEncoder() {return this.autoEncoderSession != null;}

    @Override
    public CustomConfigGroup getCustomConfigGroup() {return this.customConfigGroup;}
}
