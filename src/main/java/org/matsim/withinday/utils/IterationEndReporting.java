package org.matsim.withinday.utils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.population.Person;
import org.matsim.core.controler.events.IterationEndsEvent;
import org.matsim.withinday.utils.WithinDayAgentExperience;

public class IterationEndReporting {

    protected static final Logger log = LogManager.getLogger(IterationEndReporting.class);

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
                String tripScoresArray = formatListAsString(exp.getTripScores());
                double totalDayKaiNagelScore = exp.getFinalDayEndScore();

                // Save data row
                saveRowToCsv(
                    outputDirectory, 
                    iteration, 
                    agentId.toString(), 
                    modeChain, 
                    tripScoresArray, 
                    totalDayKaiNagelScore
                );
            }
        }
    }

    /**
     * Helper method to convert a List of values into a semicolon-delimited string (e.g. "1.2;3.4;5.6")
     */
    protected static <T> String formatListAsString(List<T> list) {
        if (list == null || list.isEmpty()) {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < list.size(); i++) {
            sb.append(list.get(i));
            if (i < list.size() - 1) {
                sb.append(";");
            }
        }
        return sb.toString();
    }

    private static void saveRowToCsv(String directory, int iteration, String id, 
                                     String modes, String tripScores, double totalDayKaiNagelScore) {
        
        String filePath = directory + File.separator + "agent_tracking.csv";
        File csvFile = new File(filePath);

        try (FileWriter fw = new FileWriter(csvFile, true)) {
            // Write standard headers if the file is fresh/new
            if (csvFile.length() == 0) {
                fw.write("iteration,agent_id,experienced_modes,trip_scores,total_day_kainagel_score\n");
            }

            // Append data row
            fw.write(String.format("%d,%s,%s,%s,%.14f\n", 
                iteration, 
                id, 
                modes, 
                tripScores, 
                totalDayKaiNagelScore
            ));
        } catch (IOException e) {
            log.error("Failed to write agent tracking metrics to CSV for agent {}: {}", id, e.getMessage());
        }
    }
}
