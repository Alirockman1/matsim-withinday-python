package org.matsim.withinday.utils;

import java.util.Collection;
import java.util.HashSet;

import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.population.Person;
import org.matsim.core.config.ReflectiveConfigGroup;

public class WithinDayConfigGroup extends ReflectiveConfigGroup {
    public static final String GROUP_NAME = "withinday";

    private static String externalModelFileName = "";
    private String agentFilterList = "";
    private boolean isAllowAllAgents = false;
    private double samplingPercentage = 1.0;
    private static String modes;
    private static String tourModesList;
    private String replanner = "default";
    private String observer = "default";

    public WithinDayConfigGroup() {
        super(GROUP_NAME);
    }

        public Collection<Id<Person>> getAgentIdsAsCollection() {
        Collection<Id<Person>> ids = new HashSet<>();
        
        // Get the original string
        String rawList = this.agentFilterList; 

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


    @StringGetter("agentFilterList")
    public String getAgentFilterList() { return agentFilterList; }

    @StringSetter("agentFilterList")
    public void setAgentFilterList(String agentFilterList) { 
        this.agentFilterList = agentFilterList;
            
            if (agentFilterList == null || agentFilterList.trim().isEmpty()) {
                this.isAllowAllAgents = true;
            } else {
                this.isAllowAllAgents = false;
            }
    }

    @StringGetter("allowAllAgents")
    public boolean getIsAllowAllAgents() {
        if (this.agentFilterList == null || this.agentFilterList.trim().isEmpty()) {
            return true;
        }
        return this.isAllowAllAgents;
    }

    @StringSetter("allowAllAgents")
    public void setIsAllowAllAgents(boolean isAllowAllAgents) { this.isAllowAllAgents = isAllowAllAgents; }

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
}