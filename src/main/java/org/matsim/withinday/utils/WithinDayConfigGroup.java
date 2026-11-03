package org.matsim.withinday.utils;

import java.io.File;
import java.net.URISyntaxException;
import java.util.Collection;
import java.util.HashSet;

import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.population.Person;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigGroup;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.config.ReflectiveConfigGroup;

public class WithinDayConfigGroup extends ReflectiveConfigGroup {
    public static final String GROUP_NAME = "withinday";

    private static String externalModelFileName = "";
    private static String dataset = "train";
    private static String modes;
    private static String tourModesList;

    private int agentsPerIteration = 1;
    private String trainAgentIDList = "";
    private double samplingPercentage = 1.0;

    private String replanner = "default";
    private String observer = "default";

    public WithinDayConfigGroup() {
        super(GROUP_NAME);
    }

    public Collection<Id<Person>> getAgentIdsAsCollection() {
        Collection<Id<Person>> ids = new HashSet<>();
        
        // Get the original string
        String rawList = this.trainAgentIDList; 

        if (rawList != null && !rawList.isEmpty()) {
            // Split by comma
            String[] parts = rawList.split(",");
            for (String part : parts) {
                // Trim whitespace and create the ID
                String cleanId = part.trim();
                if (!cleanId.isEmpty()) {
                    ids.add(Id.createPersonId(cleanId));
                }
            }
        }
        return ids;
    }

    @StringGetter("externalModelFileName")
    public static String getExternalModelFileName() { return externalModelFileName; }

    @StringSetter("externalModelFileName")
    public static void setExternalModelFileName(String modelFileName) { externalModelFileName = modelFileName; }

    @StringGetter("dataset")
    public static String getDataSetType() { return dataset;}

    @StringSetter("dataset")
    public void setDataSetType(String dataSetType) { dataset = dataSetType; }

    @StringGetter("agentsPerIteration")
    public int getAgentsPerIteration() { return agentsPerIteration; }

    @StringSetter("agentsPerIteration")
    public void setAgentsPerIteration(int agentsPerIteration) { this.agentsPerIteration = agentsPerIteration; }

    @StringGetter("trainAgentIDList")
    public String getTrainAgentIDList() { return trainAgentIDList; }

    @StringSetter("trainAgentIDList")
    public void setTrainAgentIDList(String trainAgentIDList) { this.trainAgentIDList = trainAgentIDList;}

    @StringGetter("samplingPercentage")
    public double getSamplingPercentage() { return samplingPercentage; }

    @StringSetter("samplingPercentage")
    public void setSamplingPercentage(double samplingPercentage) { this.samplingPercentage = samplingPercentage; }

    @StringGetter("modes")
    public static String getModes() { return modes; }

    @StringSetter("modes")
    public void setModes(String modes) { this.modes = modes; }  

    @StringGetter("tourBasedModes")
    public static String getTourBasedModes() { return tourModesList; }

    @StringSetter("tourBasedModes")
    public void setTourBasedModes(String modeList) { this.tourModesList = modeList; }

    @StringGetter("replanner")
    public String getReplanner() { return replanner; }

    @StringSetter("replanner")
    public void setReplanner(String replanner) { this.replanner = replanner; }

    @StringGetter("observer")
    public String getObserver() { return observer; }

    @StringSetter("observer")
    public void setObserver(String observer) { this.observer = observer; }

    /**
     * Reads the config file named by {@code withinday.externalModelFileName} - the hyperparameters
     * of an external model - into the given, already registered config groups. The file is resolved
     * next to the scenario config. A missing file is not an error: the groups then keep their
     * defaults, and the launcher can still set every value via {@code --config:}.
     *
     * @param config  The configuration being prepared; its context points at the scenario config.
     * @param targets The config groups the values are read into.
     */
    public static void loadExternalModelConfig(Config config, ConfigGroup... targets) {
        String modelFileName = getExternalModelFileName();
        if (modelFileName == null || modelFileName.isBlank() || config.getContext() == null) {
            return;
        }

        File scenarioDirectory;
        try {
            scenarioDirectory = new File(config.getContext().toURI()).getParentFile();
        } catch (URISyntaxException e) {
            throw new IllegalStateException("Cannot resolve the scenario directory from " + config.getContext(), e);
        }

        File modelFile = new File(scenarioDirectory, modelFileName);
        if (!modelFile.exists()) {
            return;
        }

        // Parses into the group instances that are already registered on config, so the values land there.
        ConfigUtils.loadConfig(modelFile.getAbsolutePath(), targets);
    }
}
