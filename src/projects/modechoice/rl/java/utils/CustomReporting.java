package modechoice.rl.java.utils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Collection;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.StringJoiner;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.Scenario;
import org.matsim.api.core.v01.population.Person;
import org.matsim.core.controler.events.IterationEndsEvent;
import org.matsim.withinday.utils.IterationEndReporting;
import org.matsim.withinday.utils.WithinDayAgentExperience;

public class CustomReporting extends IterationEndReporting{

    private static final Logger log = LogManager.getLogger(CustomReporting.class);

    public static void writeAgentStatsCsv(IterationEndsEvent event, Map<Id<Person>, WithinDayAgentExperience> agentExperiences) {
        String outputDirectory = event.getServices().getConfig().controller().getOutputDirectory();
        int iteration = event.getIteration();

        long seed = event.getServices().getConfig().global().getRandomSeed();

        String heading = "seed,iteration,agent_id,experienced_modes,trip_rewards,trip_scores,step_delta_qs,total_day_reward,total_day_kainagel_score,accumulated_delta_q";

        for (Map.Entry<Id<Person>, WithinDayAgentExperience> entry : agentExperiences.entrySet()) {
            Id<Person> agentId = entry.getKey();
            WithinDayAgentExperience exp = entry.getValue();

            Person person = event.getServices().getScenario().getPopulation().getPersons().get(agentId);
        
            if (person != null && person.getSelectedPlan() != null && exp != null) {

                // Extract array representations formatted as "val1;val2;val3"
                List<String> modeChain = exp.getExperiencedModes();
                List<Double> tripRewardsArray = exp.getTripRewards();
                List<Double> tripScoresArray = exp.getTripScores();
                List<Double> stepDeltaQsArray = exp.getStepDeltaQs();

                // Daily Totals
                double totalDayReward = exp.getFinalDayEndReward();
                double totalDayKaiNagelScore = exp.getFinalDayEndScore();
                double accumulatedDeltaQ = exp.getAccumulatedDeltaQ();


                // Save data row
                saveRowToCsv(
                    outputDirectory,
                    "agent_tracking_" + seed + ".csv",
                    heading,
                    seed,
                    iteration, 
                    agentId.toString(), 
                    modeChain, 
                    tripRewardsArray, 
                    tripScoresArray, 
                    stepDeltaQsArray, 
                    totalDayReward, 
                    totalDayKaiNagelScore, 
                    accumulatedDeltaQ
                );
            }
        }
    }

    public static void writeStepResultsToCsv(Scenario scenario, int iteration, Id<Person> agentId, WithinDayAgentExperience experience) {
        String directory = scenario.getConfig().controller().getOutputDirectory();

        String heading = "iteration,agent_id,step,mode,score,origin_taz,dest_taz,time_bin,dist_bin,asset,cong_bin";

        Map<String, Object> currentRawState = experience.getPendingRawState();
        if (currentRawState != null) {
            int stepIndex = experience.getExperiencedModes().size() - 1;

            // Extract decoded state variables (S_t)
            int originTaz = (int) currentRawState.getOrDefault("currentActivityTrafficZone", -1);
            int destTaz = (int) currentRawState.getOrDefault("scheduledActivityTrafficZone", -1);
            int timeBin = (int) currentRawState.getOrDefault("discreteTimeBin", -1);
            int distBin = (int) currentRawState.getOrDefault("distanceBin", -1);
            int assetState = (int) currentRawState.getOrDefault("assetState", -1);
            int congestionLevel = (int) currentRawState.getOrDefault("congestionLevel", -1);

            String currentMode = experience.getExperiencedModes().getLast();
            double currentScore = experience.getTripScores().getLast();

            saveRowToCsv(
                directory,
                "state_varience.csv",
                heading,
                iteration,
                agentId.toString(),
                stepIndex,
                currentMode,
                currentScore,
                originTaz,
                destTaz,
                timeBin,
                distBin,
                assetState,
                congestionLevel
            );
        }
    }


    /**
     * Completely generalized CSV writer method that accepts an arbitrary number of arguments 
     * (Varargs), formats Collections/Lists as "val1;val2;val3", formats floating-point numbers 
     * cleanly, and automatically handles header creation.
     *
     * @param directory Target output directory path.
     * @param fileName  Target CSV filename (e.g., "agent_tracking.csv" or "state_log.csv").
     * @param header    Comma-separated CSV header row.
     * @param args      Variable list of values to be written into the CSV row.
     */
    public static synchronized void saveRowToCsv(String directory, String fileName, String header, Object... args) {
        String filePath = directory + File.separator + fileName;
        File csvFile = new File(filePath);
        boolean isNew = !csvFile.exists() || csvFile.length() == 0;

        StringJoiner sj = new StringJoiner(",");

        for (Object arg : args) {
            if (arg == null) {
                sj.add("");
            } else if (arg instanceof Collection<?>) {
                StringJoiner listJ = new StringJoiner(";");

                for (Object item : (Collection<?>) arg) {
                    if (item instanceof Double || item instanceof Float) {
                        listJ.add(String.format(Locale.US, "%.4f", item));
                    } else {
                        listJ.add(String.valueOf(item));
                    }
                }

                sj.add(listJ.toString());
            } else if (arg instanceof Double || arg instanceof Float) {
                sj.add(String.format(Locale.US, "%.14f", arg));
            } else {
                sj.add(String.valueOf(arg));
            }
        }

        try (FileWriter fw = new FileWriter(csvFile, true)) {
            if (isNew) {
                fw.write(header.endsWith("\n") ? header : header + "\n");
            }
            fw.write(sj.toString() + "\n");
        } catch (IOException e) {
            log.error("Failed to write CSV row to file '{}': {}", filePath, e.getMessage());
        }
    }
}

