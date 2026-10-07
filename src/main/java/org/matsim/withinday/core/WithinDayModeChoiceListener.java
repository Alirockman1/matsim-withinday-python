package org.matsim.withinday.core;

import com.google.inject.Inject;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.Scenario;
import org.matsim.api.core.v01.events.ActivityStartEvent;
import org.matsim.api.core.v01.events.handler.ActivityStartEventHandler;
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

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * <h2>WithinDayModeChoiceListener</h2>
 * <p>
 * Central listener orchestrating within-day replanning for each trip in an agents plan in
 * the simulation lifecycle.
 * </p>
 */
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
    @Inject AgentSelector agentSelector;

    private Map<Id<Person>, Double> activityStartTimeByAgent = new HashMap<>();
    
    /**
     * Defines the execution priority of this listener relative to other MATSim controler listeners.
     * 
     * @return Priority weight value.
     */
    @Override
    public double priority() {return 10.0;}

    /**
     * Triggered upon simulation startup to initialize agent based replanner and agent sampling.
     * 
     * @param event The startup event container.
     */
    @Override
    public void notifyStartup(StartupEvent event) {
        this.customReplanner.initializeExternalModel();
        this.agentSelector.sampleAgentsForSimulation();
    }

    /**
     * Triggered at the beginning of each simulation iteration to reset tracking metrics,
     * sample active agents, initialize location tagging, and set available transport modes.
     * 
     * @param event The iteration starts event container.
     */
    @Override
    public void notifyIterationStarts(IterationStartsEvent event) {
        StateEngine.currentIteration = event.getIteration();

        // Sample agents from the subspace
        this.agentSelector.sampleAgentsForIteration(event.getIteration());
        Set<Id<Person>> selectedAgents = this.agentSelector.getSelectedAgents();

        // Matsim score tracker
        this.scoreTracker.beginIteration(selectedAgents);

        // Start tagging each agents mode geo location [CHANGE TO ONLY THE FILTED AGENTS]
        log.info("Initializing Mode Location Tagging for the filtrered agents at start of iteration {}", event.getIteration());
        AgentAssetInventory.initializeModeLocationTagging(this.scenario, selectedAgents);
        
        //selectedAgents.forEach(agentId -> {
        //    int assignedTag = this.agentSelector.getAgentTag(agentId);
        //    System.out.println("Assigned Tag for " + agentId + " is: " + assignedTag);
        //});

        // Initialize the tour based modes
        String[] tourBasedModes = AgentAssetInventory.getSimulationTourBasedModes().toArray(String[]::new);
        StateEngine.setModeAvailabilityLookup(tourBasedModes);
    }

    /**
     * Captures the exact simulation timestamp when a filtered agent begins an activity.
     * 
     * @param event The activity start event.
     */
    @Override
    public void handleEvent(ActivityStartEvent event) {
        if (this.agentSelector.contains(event.getPersonId())) {
            this.activityStartTimeByAgent.put(event.getPersonId(), event.getTime());
        }
    }

    /**
     * Executed before each QSim time step to intercept agents ending their activities 
     * and trigger within-day trip replanning when applicable.
     * 
     * @param e The mobsim before sim step event.
     */
    @Override
    public void notifyMobsimBeforeSimStep(MobsimBeforeSimStepEvent e) {
        this.customReplanner.initEditTrips();
        QSim sim = (QSim) e.getQueueSimulation();
        double currentTime = e.getSimulationTime();

        // Pick all agents that end their activity at the current simulation time.
        sim.getAgents().values().stream()
            .filter(p -> this.agentSelector.contains(p.getId()))
            .filter(p -> p.getState() == MobsimAgent.State.ACTIVITY)
            .filter(p -> p.getActivityEndTime() == currentTime)
            .filter(this.agentSelector::shouldReplan)
            .forEach(p -> this.customReplanner.replanNextTrip(p, sim, currentTime));
    }

    /**
     * Executed after each QSim time step to process agents that started their activities 
     * during the current simulation step.
     * 
     * @param e The mobsim after sim step event.
     */
    @Override
    public void notifyMobsimAfterSimStep(MobsimAfterSimStepEvent e) {
        this.customReplanner.initEditTrips();
        QSim sim = (QSim) e.getQueueSimulation();
        double currentTime = e.getSimulationTime();

        // Pick all agents that started their activity in the current step
        sim.getAgents().values().stream()
                .filter(p -> p.getState() == MobsimAgent.State.ACTIVITY)
                .filter(p -> activityStartTimeByAgent.containsKey(p.getId()))
                .filter(p -> activityStartTimeByAgent.get(p.getId()) == currentTime)
                .filter(p -> this.agentSelector.contains(p.getId()))
                .forEach(p -> {
                    this.customReplanner.step(p, sim, currentTime, false);
                });
    }

    /**
     * Cleans up listener caches, inventory states, and replanner data at the conclusion 
     * of each iteration.
     * 
     * @param event The iteration ends event container.
     */
    @Override
    public void notifyIterationEnds(IterationEndsEvent event){
        log.info("Cleaning up listener cache and inventory for iteration {}", event.getIteration());
        this.activityStartTimeByAgent.clear();
        this.customReplanner.reset(event);
        this.agentSelector.reset();
    }

    /**
     * Resets internal tracking maps and caches for a given iteration.
     * 
     * @param iteration The iteration index being reset.
     */
    @Override
    public void reset(int iteration) {
        this.activityStartTimeByAgent.clear();
    }
   
}
