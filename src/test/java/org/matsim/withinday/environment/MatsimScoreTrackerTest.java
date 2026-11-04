package org.matsim.withinday.environment;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.Scenario;
import org.matsim.api.core.v01.population.Person;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.config.groups.ReplanningConfigGroup;
import org.matsim.core.config.groups.RoutingConfigGroup;
import org.matsim.core.config.groups.ScoringConfigGroup;
import org.matsim.core.controler.AbstractModule;
import org.matsim.core.controler.Controller;
import org.matsim.core.controler.ControllerUtils;
import org.matsim.core.controler.OutputDirectoryHierarchy.OverwriteFileSetting;
import org.matsim.core.replanning.strategies.DefaultPlanStrategiesModule;
import org.matsim.core.scenario.ScenarioUtils;
import org.matsim.withinday.utils.WithinDayConfigGroup;

import modechoice.rl.java.utils.CustomConfigGroup;

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
}
