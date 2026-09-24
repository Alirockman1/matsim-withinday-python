package modechoice.rl.java.utils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Map;
import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.population.Person;
import org.matsim.core.controler.events.IterationEndsEvent;
import org.matsim.withinday.utils.IterationEndReporting;
import org.matsim.withinday.utils.WithinDayAgentExperience;

public class CustomIterationEndReporting extends IterationEndReporting{

    public static void writeAgentStatsCsv(IterationEndsEvent event, Map<Id<Person>, WithinDayAgentExperience> agentExperiences) {
        
        String outputDirectory = event.getServices().getConfig().controller().getOutputDirectory();
        int iteration = event.getIteration();

        for (Map.Entry<Id<Person>, WithinDayAgentExperience> entry : agentExperiences.entrySet()) {
            Id<Person> agentId = entry.getKey();
            WithinDayAgentExperience exp = entry.getValue();

            Person person = event.getServices().getScenario().getPopulation().getPersons().get(agentId);
        
            if (person != null && person.getSelectedPlan() != null && exp != null) {

                // Extract array representations formatted as "val1;val2;val3"
                String modeChain = formatListAsString(exp.getExperiencedModes());
                String tripRewardsArray = formatListAsString(exp.getTripRewards());
                String tripScoresArray = formatListAsString(exp.getTripScores());
                String stepDeltaQsArray = formatListAsString(exp.getStepDeltaQs());

                // Daily Totals
                double totalDayReward = exp.getFinalDayEndReward();
                double totalDayKaiNagelScore = exp.getFinalDayEndScore();
                double accumulatedDeltaQ = exp.getAccumulatedDeltaQ();

                // Save data row
                saveRowToCsv(
                    outputDirectory, 
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

    private static void saveRowToCsv(String directory, int iteration, String id, 
                                     String modes, String tripRewards, String tripScores, 
                                     String stepDeltaQs,double totalDayReward, 
                                     double totalDayKaiNagelScore, double accumulatedDeltaQ) {
        
        String filePath = directory + File.separator + "agent_tracking.csv";
        File csvFile = new File(filePath);

        try (FileWriter fw = new FileWriter(csvFile, true)) {
            // Write standard headers if the file is fresh/new
            if (csvFile.length() == 0) {
                fw.write("iteration,agent_id,experienced_modes,trip_rewards,trip_scores,step_delta_qs,total_day_reward,total_day_kainagel_score,accumulated_delta_q\n");
            }

            // Append data row
            fw.write(String.format("%d,%s,%s,%s,%s,%s,%.14f,%.14f,%.14f\n", 
                iteration, 
                id, 
                modes, 
                tripRewards, 
                tripScores, 
                stepDeltaQs, 
                totalDayReward, 
                totalDayKaiNagelScore, 
                accumulatedDeltaQ
            ));
        } catch (IOException e) {
            log.error("Failed to write agent tracking metrics to CSV for agent {}: {}", id, e.getMessage());
        }
    }
}
