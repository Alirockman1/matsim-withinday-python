package org.matsim.withinday.environment;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.Scenario;
import org.matsim.api.core.v01.events.ActivityStartEvent;
import org.matsim.api.core.v01.events.handler.ActivityStartEventHandler;
import org.matsim.api.core.v01.population.Activity;
import org.matsim.api.core.v01.population.Person;
import org.matsim.api.core.v01.population.Plan;
import org.matsim.api.core.v01.population.PlanElement;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.config.groups.ReplanningConfigGroup;
import org.matsim.core.config.groups.RoutingConfigGroup;
import org.matsim.core.config.groups.ScoringConfigGroup;
import org.matsim.core.controler.AbstractModule;
import org.matsim.core.controler.Controller;
import org.matsim.core.controler.ControllerUtils;
import org.matsim.core.controler.OutputDirectoryHierarchy.OverwriteFileSetting;
import org.matsim.core.controler.events.IterationEndsEvent;
import org.matsim.core.controler.events.IterationStartsEvent;
import org.matsim.core.controler.listener.IterationEndsListener;
import org.matsim.core.controler.listener.IterationStartsListener;
import org.matsim.core.mobsim.framework.events.MobsimAfterSimStepEvent;
import org.matsim.core.mobsim.framework.listeners.MobsimAfterSimStepListener;
import org.matsim.core.population.PopulationUtils;
import org.matsim.core.replanning.strategies.DefaultPlanStrategiesModule;
import org.matsim.core.router.StageActivityTypeIdentifier;
import org.matsim.core.scenario.ScenarioUtils;
import org.matsim.withinday.utils.WithinDayConfigGroup;

import modechoice.rl.java.utils.CustomConfigGroup;

import com.google.inject.Inject;

/**
 * The point of {@link MatsimScoreTracker} is that the score it exposes during the mobsim is MATSim's
 * own score, not a re-implementation of it. This test checks exactly that: it closes each agent's day
 * the moment the agent reaches its final activity -- the same instant the within-day replanner does --
 * and compares the resulting score against the score MATSim writes to the selected plan afterwards.
 */
public class MatsimScoreTrackerTest {

    private static final String CONFIG = "scenarios/sioux-falls/input/config.xml";
    private static final int AGENTS_UNDER_TEST = 25;

    @Test
    void trackerReproducesMatsimScore(@TempDir Path outputDirectory) {
        Config config = ConfigUtils.loadConfig(CONFIG, new CustomConfigGroup(), new WithinDayConfigGroup());

        config.controller().setOutputDirectory(outputDirectory.toString());
        config.controller().setOverwriteFileSetting(OverwriteFileSetting.deleteDirectoryIfExists);
        config.controller().setFirstIteration(0);
        config.controller().setLastIteration(0);
        config.controller().setWritePlansInterval(0);
        config.controller().setWriteEventsInterval(0);
        config.controller().setWriteSnapshotsInterval(0);
        config.controller().setCreateGraphsInterval(0);
        config.controller().setDumpDataAtEnd(false);

        // Same framework settings the within-day runner applies
        config.routing().setNetworkRouteConsistencyCheck(RoutingConfigGroup.NetworkRouteConsistencyCheck.disable);
        config.scoring().addModeParams(new ScoringConfigGroup.ModeParams("walk"));
        config.replanning().clearStrategySettings();
        config.replanning().addStrategySettings(new ReplanningConfigGroup.StrategySettings()
                .setStrategyName(DefaultPlanStrategiesModule.DefaultSelector.KeepLastSelected)
                .setWeight(1.0));

        Scenario scenario = ScenarioUtils.loadScenario(config);
        Set<Id<Person>> trackedAgents = keepFirstAgentsOnly(scenario, AGENTS_UNDER_TEST);

        Controller controller = ControllerUtils.createController(scenario);
        ScoreComparison comparison = new ScoreComparison(scenario, trackedAgents);

        controller.addOverridingModule(new AbstractModule() {
            @Override
            public void install() {
                bind(MatsimScoreTracker.class).asEagerSingleton();
                addEventHandlerBinding().to(MatsimScoreTracker.class);

                bind(ScoreComparison.class).toInstance(comparison);
                addControllerListenerBinding().toInstance(comparison);
                addMobsimListenerBinding().toInstance(comparison);
                addEventHandlerBinding().toInstance(comparison);
            }
        });

        controller.run();

        assertTrue(comparison.getComparedAgents() >= trackedAgents.size() / 2,
                "only " + comparison.getComparedAgents() + " of " + trackedAgents.size()
                        + " agents reached their final activity; the comparison would be vacuous");

        for (Map.Entry<Id<Person>, double[]> entry : comparison.getScores().entrySet()) {
            double trackedScore = entry.getValue()[0];
            double matsimScore = entry.getValue()[1];
            assertEquals(matsimScore, trackedScore, 1e-6,
                    "tracked score differs from MATSim's own score for agent " + entry.getKey());
        }
    }

    /**
     * Trims the population so the test runs on a handful of agents instead of the full 10% sample.
     */
    private Set<Id<Person>> keepFirstAgentsOnly(Scenario scenario, int numberOfAgents) {
        List<Id<Person>> allAgents = new ArrayList<>(scenario.getPopulation().getPersons().keySet());
        Set<Id<Person>> kept = new HashSet<>(allAgents.subList(0, Math.min(numberOfAgents, allAgents.size())));

        for (Id<Person> agentId : allAgents) {
            if (!kept.contains(agentId)) {
                scenario.getPopulation().removePerson(agentId);
            }
        }

        return kept;
    }

    /**
     * Closes each tracked agent's day on arrival at its final activity and collects both scores.
     */
    private static class ScoreComparison implements IterationStartsListener, IterationEndsListener,
            MobsimAfterSimStepListener, ActivityStartEventHandler {

        private final Scenario scenario;
        private final Set<Id<Person>> trackedAgents;

        @Inject private MatsimScoreTracker tracker;

        private final Map<Id<Person>, Integer> realActivityCount = new HashMap<>();
        private final Map<Id<Person>, Integer> activityStartsSeen = new HashMap<>();
        private final Map<Id<Person>, Activity> pendingFinalActivity = new HashMap<>();
        private final Map<Id<Person>, Double> pendingArrivalTime = new HashMap<>();
        private final Map<Id<Person>, double[]> scores = new HashMap<>();

        ScoreComparison(Scenario scenario, Set<Id<Person>> trackedAgents) {
            this.scenario = scenario;
            this.trackedAgents = trackedAgents;
        }

        @Override
        public void notifyIterationStarts(IterationStartsEvent event) {
            this.tracker.beginIteration(this.trackedAgents);

            this.realActivityCount.clear();
            this.activityStartsSeen.clear();
            this.scores.clear();

            for (Id<Person> agentId : this.trackedAgents) {
                Plan plan = this.scenario.getPopulation().getPersons().get(agentId).getSelectedPlan();
                int realActivities = 0;
                for (PlanElement element : plan.getPlanElements()) {
                    if (element instanceof Activity activity
                            && !StageActivityTypeIdentifier.isStageActivity(activity.getType())) {
                        realActivities++;
                    }
                }
                this.realActivityCount.put(agentId, realActivities);
            }
        }

        @Override
        public void handleEvent(ActivityStartEvent event) {
            if (StageActivityTypeIdentifier.isStageActivity(event.getActType())
                    || !this.trackedAgents.contains(event.getPersonId())) {
                return;
            }

            int seen = this.activityStartsSeen.merge(event.getPersonId(), 1, Integer::sum);

            // The agent's day starts inside its first activity, so no start event is fired for it:
            // the (N-1)th start event is the arrival at the final activity.
            if (seen == this.realActivityCount.getOrDefault(event.getPersonId(), 0) - 1) {
                Activity finalActivity = PopulationUtils.createActivityFromLinkId(event.getActType(), event.getLinkId());
                finalActivity.setFacilityId(event.getFacilityId());
                finalActivity.setStartTime(event.getTime());

                this.pendingFinalActivity.put(event.getPersonId(), finalActivity);
                this.pendingArrivalTime.put(event.getPersonId(), event.getTime());
            }
        }

        /**
         * Deferred to after the sim step so that all events of that step -- including the trip scoring
         * triggered by the arrival itself -- have been dispatched, exactly as the within-day listener does.
         */
        @Override
        public void notifyMobsimAfterSimStep(MobsimAfterSimStepEvent e) {
            if (this.pendingFinalActivity.isEmpty()) {
                return;
            }

            for (Map.Entry<Id<Person>, Activity> entry : this.pendingFinalActivity.entrySet()) {
                double dayScore = this.tracker.finishDay(entry.getKey(), entry.getValue(),
                        this.pendingArrivalTime.get(entry.getKey()));
                this.scores.put(entry.getKey(), new double[] { dayScore, Double.NaN });
            }

            this.pendingFinalActivity.clear();
            this.pendingArrivalTime.clear();
        }

        @Override
        public void notifyIterationEnds(IterationEndsEvent event) {
            for (Map.Entry<Id<Person>, double[]> entry : this.scores.entrySet()) {
                Plan plan = this.scenario.getPopulation().getPersons().get(entry.getKey()).getSelectedPlan();
                entry.getValue()[1] = plan.getScore();
            }
        }

        @Override
        public void reset(int iteration) {
            this.pendingFinalActivity.clear();
            this.pendingArrivalTime.clear();
        }

        int getComparedAgents() {
            return this.scores.size();
        }

        Map<Id<Person>, double[]> getScores() {
            return this.scores;
        }
    }
}