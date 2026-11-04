package modechoice.random.java.core;

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
import org.matsim.application.MATSimApplication;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.controler.AbstractModule;
import org.matsim.core.controler.Controler;
import org.matsim.withinday.environment.MatsimScoreTracker;
import org.matsim.withinday.environment.ScoreComparison;

/**
 * <h2>BerlinScoreTrackerTest</h2>
 * <p>
 * The Sioux Falls counterpart of this test shows that {@link MatsimScoreTracker} exposes MATSim's own
 * score during the mobsim. Open Berlin scores differently - {@code BerlinScoringFunctionFactory} with
 * income dependent taste variations and a pseudo random term - so the same invariant is checked again
 * on the mini Berlin scenario, with Berlin's scoring in place.
 * </p>
 * <p>
 * The run goes through {@link RunBerlinRandomModeChoice}, but its within-day module is replaced, so
 * neither the communication manager nor the python bridge is involved.
 * </p>
 */
public class BerlinScoreTrackerTest {

    private static final String CONFIG = "scenarios/berlin-v7.1.1/input/berlin-mini.config.xml";
    private static final int AGENTS_UNDER_TEST = 25;

    @Test
    void trackerReproducesBerlinScore(@TempDir Path outputDirectory) {
        TrackingRun run = new TrackingRun(AGENTS_UNDER_TEST);

        Controler controler = MATSimApplication.prepare(run, ConfigUtils.loadConfig(CONFIG),
                "--config:controller.outputDirectory=" + outputDirectory,
                "--config:controller.lastIteration=0",
                "--config:controller.writeEventsInterval=0",
                "--config:controller.writePlansInterval=0",
                "--post-processing", "disabled");

        controler.run();

        ScoreComparison comparison = run.getComparison();
        assertTrue(comparison.getComparedAgents() >= AGENTS_UNDER_TEST / 2,
                "only " + comparison.getComparedAgents() + " of " + AGENTS_UNDER_TEST
                        + " agents reached their final activity; the comparison would be vacuous");

        for (Map.Entry<Id<Person>, double[]> entry : comparison.getScores().entrySet()) {
            double trackedScore = entry.getValue()[0];
            double matsimScore = entry.getValue()[1];
            assertEquals(matsimScore, trackedScore, 1e-6,
                    "tracked score differs from MATSim's own score for agent " + entry.getKey());
        }
    }

    /**
     * The Berlin runner with the population trimmed and the within-day module replaced by the score
     * comparison.
     */
    private static final class TrackingRun extends RunBerlinRandomModeChoice {

        private final int numberOfAgents;
        private ScoreComparison comparison;

        private TrackingRun(int numberOfAgents) {
            this.numberOfAgents = numberOfAgents;
        }

        @Override
        protected void prepareScenario(Scenario scenario) {
            super.prepareScenario(scenario);

            Set<Id<Person>> trackedAgents = keepFirstAgentsOnly(scenario, this.numberOfAgents);
            this.comparison = new ScoreComparison(scenario, trackedAgents);
        }

        @Override
        protected AbstractModule createWithinDayModule(Config config) {
            ScoreComparison scoreComparison = this.comparison;

            return new AbstractModule() {
                @Override
                public void install() {
                    bind(MatsimScoreTracker.class).asEagerSingleton();
                    addEventHandlerBinding().to(MatsimScoreTracker.class);

                    bind(ScoreComparison.class).toInstance(scoreComparison);
                    addControllerListenerBinding().toInstance(scoreComparison);
                    addMobsimListenerBinding().toInstance(scoreComparison);
                    addEventHandlerBinding().toInstance(scoreComparison);
                }
            };
        }

        private ScoreComparison getComparison() {
            return this.comparison;
        }

        /**
         * Trims the population so the test runs on a handful of agents.
         */
        private static Set<Id<Person>> keepFirstAgentsOnly(Scenario scenario, int numberOfAgents) {
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
}
