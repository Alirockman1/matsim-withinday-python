package org.matsim.withinday.core;

import com.google.inject.Inject;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.Scenario;
import org.matsim.api.core.v01.events.ActivityStartEvent;
import org.matsim.api.core.v01.events.handler.ActivityStartEventHandler;
import org.matsim.api.core.v01.network.Link;
import org.matsim.core.controler.listener.IterationEndsListener;
import org.matsim.core.controler.events.IterationEndsEvent;
import org.matsim.api.core.v01.population.Person;
import org.matsim.core.controler.events.IterationStartsEvent;
import org.matsim.core.controler.events.StartupEvent;
import org.matsim.core.controler.listener.IterationStartsListener;
import org.matsim.core.controler.listener.StartupListener;
import org.matsim.core.mobsim.framework.MobsimAgent;
import org.matsim.core.mobsim.framework.events.MobsimAfterSimStepEvent;
import org.matsim.core.mobsim.framework.events.MobsimBeforeSimStepEvent;
import org.matsim.core.mobsim.framework.listeners.MobsimAfterSimStepListener;
import org.matsim.core.mobsim.framework.listeners.MobsimBeforeSimStepListener;
import org.matsim.core.mobsim.qsim.QSim;
import org.matsim.core.router.TripRouter;
import org.matsim.withinday.environment.AgentAssetInventory;
import org.matsim.withinday.environment.MatsimScoreTracker;
import org.matsim.withinday.environment.StateEngine;
import org.matsim.withinday.environment.WithinDayObserver;
import org.matsim.withinday.networking.CommunicationManager;
import org.matsim.withinday.utils.WithinDayConfigGroup;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class WithinDayModeChoiceListener implements StartupListener, IterationStartsListener,
IterationEndsListener, MobsimBeforeSimStepListener,
MobsimAfterSimStepListener, ActivityStartEventHandler {

    private static final Logger log = LogManager.getLogger(WithinDayModeChoiceListener.class);

    @Inject TripRouter router;
    @Inject Scenario scenario;
    @Inject CommunicationManager pythonCommunicationManager;
    @Inject WithinDayReplanner customReplanner;
    @Inject WithinDayObserver customObserver;
    @Inject WithinDayConfigGroup configGroup;
    @Inject MatsimScoreTracker scoreTracker;

    private AgentSelector agentSelector;
    private Map<Id<Person>, Double> activityStartTimeByAgent = new HashMap<>();
    
    @Override
    public double priority() {return 10.0;}


    @Override
    public void notifyStartup(StartupEvent event) {
        this.agentSelector = new AgentSelector(scenario, 42L, log);
        this.customReplanner.initializeExternalModel();
    }


    @Override
    public void notifyIterationStarts(IterationStartsEvent event) {
        StateEngine.currentIteration = event.getIteration();

        // Sample agents from the population
        double samplingPercentage = this.configGroup.getSamplingPercentage();

        Collection<String> fixedAgentIds = new ArrayList<>();
        if (this.configGroup!= null && this.configGroup.getAgentFilterList() != null) {
            fixedAgentIds = Arrays.asList(this.configGroup.getAgentFilterList().split("\\s*,\\s*"));
        }

        this.agentSelector.sampleAgentsForIteration(event.getIteration(), samplingPercentage, fixedAgentIds);
        Set<Id<Person>> selectedAgents = this.agentSelector.getSelectedAgents();

        this.scoreTracker.beginIteration(selectedAgents);

        // Start tagging each agents mode geo location [CHANGE TO ONLY THE FILTED AGENTS]
        log.info("Initializing Mode Location Tagging for the filtrered agents at start of iteration {}", event.getIteration());
        AgentAssetInventory.initializeModeLocationTagging(this.scenario, selectedAgents);

        selectedAgents.forEach(agentId -> {
            Map<String, Id<Link>> location = AgentAssetInventory.getModeLocation(agentId);
            System.out.println("Mode location for " + agentId + " is: " + location);
        });

        // Initialize the tour based modes
        String[] tourBasedModes = AgentAssetInventory.getSimulationTourBasedModes().toArray(String[]::new);
        System.out.println("Tour modes are: " + AgentAssetInventory.getSimulationTourBasedModesAsString());
        StateEngine.setModeAvailabilityLookup(tourBasedModes);
    }

    /**
     * Captures the exact simulation timestamp when an agent begins an activity.
     */
    @Override
    public void handleEvent(ActivityStartEvent event) {
        if (this.agentSelector.contains(event.getPersonId())) {
            this.activityStartTimeByAgent.put(event.getPersonId(), event.getTime());
        }
    }


    @Override
    public void notifyMobsimBeforeSimStep(MobsimBeforeSimStepEvent e) {
        this.customReplanner.initEditTrips();
        QSim sim = (QSim) e.getQueueSimulation();
        double currentTime = e.getSimulationTime();

        // Pick all agents that end their activity at the current simulation time
        // and whose activity is going to end in this time step.
        sim.getAgents().values().stream()
                .filter(p -> p.getState() == MobsimAgent.State.ACTIVITY)
                .filter(p -> p.getActivityEndTime() == currentTime)
                .filter(this.agentSelector::shouldReplan)
                .forEach(p -> {
                    this.customReplanner.replanNextTrip(p, sim, currentTime);
                });
    }


    @Override
    public void notifyMobsimAfterSimStep(MobsimAfterSimStepEvent e) {
        this.customReplanner.initEditTrips();
        QSim sim = (QSim) e.getQueueSimulation();
        double currentTime = e.getSimulationTime();

        // Pick all agents that started their activity in the current step
        // interaction activities are automatically filtered out.
        // In the same time step => the state of such an agent is LEG after the sim step.
        sim.getAgents().values().stream()
                .filter(p -> p.getState() == MobsimAgent.State.ACTIVITY)
                .filter(p -> activityStartTimeByAgent.containsKey(p.getId()))
                .filter(p -> activityStartTimeByAgent.get(p.getId()) == currentTime)
                .filter(p -> this.agentSelector.contains(p.getId()))
                .forEach(p -> {
                    this.customReplanner.step(p, sim, currentTime, false);
                });
    }


    @Override
    public void notifyIterationEnds(IterationEndsEvent event){
        log.info("Cleaning up listener cache and inventory for iteration {}", event.getIteration());

        // Get the q_table for the agent
        //try {
        //    String sessionMetrics = pythonCommunicationManager.httpGet("/session/metrics", 360);
        //}catch (Exception e) {
        //    log.error("COMMUNICATION NET: Failed retriveing Q-Table. " + e.getMessage());
        //}
  
        this.activityStartTimeByAgent.clear();
        this.customReplanner.reset(event);
        this.agentSelector.reset();
    }


    @Override
    public void reset(int iteration) {
        this.activityStartTimeByAgent.clear();
    }
   
}
