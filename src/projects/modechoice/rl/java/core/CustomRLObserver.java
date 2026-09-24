package modechoice.rl.java.core;

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
import org.matsim.core.router.TripStructureUtils.Trip;
import org.matsim.withinday.environment.AgentAssetInventory;
import org.matsim.withinday.environment.StateEngine;
import org.matsim.withinday.environment.StateEngine.GridPosition;
import org.matsim.withinday.environment.WithinDayObserver;
import org.matsim.withinday.environment.WithinDayRealTimeScoringEngine;

import modechoice.rl.java.environment.CustomRLScoringEngine;
import modechoice.rl.java.utils.CustomConfigGroup;

import com.google.inject.Inject;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

public class CustomRLObserver extends WithinDayObserver {
    private static final Logger log = LogManager.getLogger(CustomRLObserver.class);
    private static final NavigableMap<Double, Integer> CUSTOM_TIME_BIN_LOOKUP = new TreeMap<>();
    private final Config config;
    private final String[] tourBasedModes;
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
        CUSTOM_TIME_BIN_LOOKUP.put(4.0, 0); CUSTOM_TIME_BIN_LOOKUP.put(6.0, 1); CUSTOM_TIME_BIN_LOOKUP.put(6.5, 2);
        CUSTOM_TIME_BIN_LOOKUP.put(7.0, 3); CUSTOM_TIME_BIN_LOOKUP.put(7.5, 4); CUSTOM_TIME_BIN_LOOKUP.put(8.0, 5);
        CUSTOM_TIME_BIN_LOOKUP.put(8.5, 6); CUSTOM_TIME_BIN_LOOKUP.put(9.0, 7); CUSTOM_TIME_BIN_LOOKUP.put(10.0, 8);
        CUSTOM_TIME_BIN_LOOKUP.put(12.0, 9); CUSTOM_TIME_BIN_LOOKUP.put(14.0, 10); CUSTOM_TIME_BIN_LOOKUP.put(15.5, 11);
        CUSTOM_TIME_BIN_LOOKUP.put(16.5, 12); CUSTOM_TIME_BIN_LOOKUP.put(17.5, 13); CUSTOM_TIME_BIN_LOOKUP.put(18.5, 14);
        CUSTOM_TIME_BIN_LOOKUP.put(20.0, 15); CUSTOM_TIME_BIN_LOOKUP.put(22.0, 16);
    }

    @Inject private CustomConfigGroup customConfigGroup;
    
    @Inject
    public CustomRLObserver(Scenario scenario) {
        super(scenario, log);
        this.config = scenario.getConfig();
        this.tourBasedModes = AgentAssetInventory.getSimulationTourBasedModes().toArray(new String[0]);
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

    @Override
    public Map<String, Object> observeState(MobsimAgent agent, QSim sim, Trip trip, double timeInSeconds, boolean isNextTripContext) {
        Map<String, Object> observation = new HashMap<>();
        Map<String, Object> rawStateSpace = new HashMap<>();
        int[] rawBitStateSpace = null;
        double departureTimeSeconds = timeInSeconds;

        if (isNextTripContext) {
            departureTimeSeconds = StateEngine.getPredictedDepartureTime(trip.getDestinationActivity(), timeInSeconds).seconds();
        }

        Map<String, Object> endOfDayResult = endOfDay(agent, trip, isNextTripContext);
        boolean isEndOfDay = (boolean) endOfDayResult.get("flag");
        Trip targetTrip = (Trip) endOfDayResult.get("targetTrip");

        // Mode set for the agent
        List<String> availableModes = new ArrayList<>(AgentAssetInventory.getSimulationModes());

        // Current activity location
        Activity currentActivity = trip.getOriginActivity();
        Map<String, Object> currentActivityLocation = StateEngine.getActivityLocation(currentActivity);
        double xCurrent = (double) currentActivityLocation.get("x");
        double yCurrent = (double) currentActivityLocation.get("y");

        // Asset spatial positioning
        int assetState = StateEngine.getAssetStateOnHand(agent.getId(), agent.getCurrentLinkId(), tourBasedModes);

        if (!isEndOfDay) {
            rawStateSpace = buildActiveTripState(targetTrip, xCurrent, yCurrent, departureTimeSeconds, assetState);

            // Discretize continuous variables
            int discreteTimeBin = StateEngine.discretizeTimeFromContinous(departureTimeSeconds, "demand_based");
            GridPosition currentDiscretePosition = StateEngine.discretizePositionFromContinuous(xCurrent, yCurrent, 8);
            GridPosition nextDiscretePosition = StateEngine.discretizePositionFromContinuous(
                    (double) rawStateSpace.get("scheduledActivityLocationX"), 
                    (double) rawStateSpace.get("scheduledActivityLocationY"), 8);

            // Bit state representation
            int[] timeBits = StateEngine.convertToBitStateRepresentation(discreteTimeBin, StateEngine.getTimeBinSize("demand_based"));
            int[] assetBits = StateEngine.convertToBitStateRepresentation(assetState, StateEngine.getAssetBinSize());
            int[] currentPositionBits = StateEngine.oneHotEncodePosition(currentDiscretePosition.cellIndex(), currentDiscretePosition.gridShape());
            int[] nextPositionBits = StateEngine.oneHotEncodePosition(nextDiscretePosition.cellIndex(), nextDiscretePosition.gridShape());

            // Sequential Vector Assembly
            int totalLength = currentPositionBits.length + nextPositionBits.length + timeBits.length + 1 + assetBits.length;
            int offset = 0;             
            
            rawBitStateSpace = new int[totalLength];
            offset = appendBits(rawBitStateSpace, offset, currentPositionBits);
            offset = appendBits(rawBitStateSpace, offset, nextPositionBits);
            offset = appendBits(rawBitStateSpace, offset, timeBits);
            offset = appendBit(rawBitStateSpace, offset, 1);
            offset = appendBits(rawBitStateSpace, offset, assetBits);

        } else {
            rawStateSpace = buildTerminalState(xCurrent, yCurrent, assetState);

            // Discretize continuous variables
            GridPosition currentDiscretePosition = StateEngine.discretizePositionFromContinuous(xCurrent, yCurrent, 8);

            // Bit state representation
            int[] assetBits = StateEngine.convertToBitStateRepresentation(assetState, StateEngine.getAssetBinSize());
            int[] currentPositionBits = StateEngine.oneHotEncodePosition(currentDiscretePosition.cellIndex(), currentDiscretePosition.gridShape());

            // Sequential Vector Assembly
            int totalLength = currentPositionBits.length + currentPositionBits.length + StateEngine.getTimeBinSize("demand_based") + 1 + assetBits.length;
            int offset = 0;
            
            rawBitStateSpace = new int[totalLength];
            offset = appendBits(rawBitStateSpace, offset, currentPositionBits);
            offset += currentDiscretePosition.gridShape() + StateEngine.getTimeBinSize("demand_based") + 1;
            offset = appendBits(rawBitStateSpace, offset, assetBits);
        }

        // Compressed Latent bit state
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
        }

        // Top-Level Framework Metadata
        observation.put("simulationIteration", StateEngine.currentIteration);
        observation.put("agentID", agent.getId().toString());
        observation.put("possibleModeSet", availableModes);
        observation.put("endOfDayFlag", isEndOfDay);

        // Encapsulated Dynamic Features Container
        Map<String, Object> features = new HashMap<>();
        features.put("rawStateObservation", rawStateSpace);
        features.put("rawBitStateRepresentation", rawBitStateSpace);
        features.put("latentBitStateRepresentation", latentBitStateSpaceInt);

        observation.put("features", features);
        return observation;
    }

    @Override
    public CustomConfigGroup getCustomConfigGroup() {
        return this.customConfigGroup;
    }

    /**
     * Compiles continuous variables for an active intra-day trip request.
     */
    private Map<String, Object> buildActiveTripState(Trip trip, double xCurrent, double yCurrent, double timeInSeconds, int assetState) {
        
        // Next scheduled activity location
        Map<String, Object> state = new HashMap<>();
        Activity nextActivity = trip.getDestinationActivity();
        Map<String, Object> nextLocation = StateEngine.getActivityLocation(nextActivity);

        state.put("currentActivityLocationX", xCurrent);
        state.put("currentActivityLocationY", yCurrent);
        state.put("scheduledActivityLocationX", nextLocation.get("x"));
        state.put("scheduledActivityLocationY", nextLocation.get("y"));
        state.put("departureTimeSeconds", timeInSeconds);        
        state.put("scheduledActivityFlexibility", isActivityFlexible(nextActivity) ? 1 : 0);
        state.put("assetState", assetState);
        return state;
    }

    /**
     * Compiles continuous variables for terminal end-of-day scenarios.
     */
    private Map<String, Object> buildTerminalState(double xCurrent, double yCurrent, int assetState) {
        Map<String, Object> state = new HashMap<>();
        state.put("currentActivityLocationX", xCurrent);
        state.put("currentActivityLocationY", yCurrent);
        state.put("assetState", assetState);
        return state;
    }

    /**
     * Utility method evaluating plan flexibility properties.
     */
    private boolean isActivityFlexible(Activity activity) {
        String type = activity.getType();
        return type.contentEquals("secondary") || type.contentEquals("leisure") || type.contentEquals("home");
    }

    /**
     * Appends a source array of bits into a destination bit vector starting at the specified offset.
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
     * Appends a single scalar bit value into a destination bit vector at the specified offset.
     *
     * @param dest     The target binary state array being populated.
     * @param offset   The current write position index in the destination array.
     * @param bitValue The single integer bit value (e.g., 0 or 1 for flexibility).
     * @return The updated offset index incremented by 1.
     */
    private int appendBit(int[] dest, int offset, int bitValue) {
        dest[offset] = bitValue;
        return offset + 1;
    }

    /**
     * Helper method to resolve path and initialize the ONNX session once.
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

    // Optional helper to check if ONNX is active
    public boolean hasAutoEncoder() {
        return this.autoEncoderSession != null;
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
}
