package modechoice.rl.java.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.config.groups.ScoringConfigGroup;
import org.matsim.core.replanning.strategies.DefaultPlanStrategiesModule;
import org.matsim.core.config.groups.ReplanningConfigGroup;
import org.matsim.withinday.utils.WithinDayConfigGroup;

import modechoice.rl.java.utils.CustomConfigGroup;
import picocli.CommandLine;

/**
 * <h2>RunBerlinRLModeChoiceConfigTest</h2>
 * <p>
 * Guards the config preparation of {@link RunBerlinRLModeChoice}: the Open Berlin setup and the
 * within-day settings it inherits have to survive, and the {@code agentModeChoice} hyperparameters
 * have to arrive from the second config file. Only the config is built, so no scenario input is
 * read. It lives in the runner's package to reach the protected {@code prepareConfig} hook.
 * </p>
 */
public class RunBerlinRLModeChoiceConfigTest {

    private static final String CONFIG = "scenarios/berlin-v7.1.1/input/berlin-mini.config.xml";

    @Test
    void hyperparametersAreReadFromTheExternalModelConfig() {
        Config config = prepareConfig();

        assertInstanceOf(CustomConfigGroup.class, config.getModule(CustomConfigGroup.GROUP_NAME),
                "the agentModeChoice module was not materialized into its typed config group");

        CustomConfigGroup custom = ConfigUtils.addOrGetModule(config, CustomConfigGroup.class);

        // Values that only exist in berlin-mini.modeChoiceModel.xml.
        assertEquals("QLearning", custom.getModelType());
        assertEquals(0.1, custom.getAlpha(), 1e-9);
        assertEquals(0.9, custom.getGamma(), 1e-9);
        assertEquals(0.9, custom.getEpsilon(), 1e-9);
        assertEquals(0.9995, custom.getEpsilonDecay(), 1e-9);
        assertEquals(0.05, custom.getEpsilonMinimum(), 1e-9);
        assertEquals(800, custom.getTrainingCutoffIteration());

        // The traffic analysis zones the RL state space needs.
        assertEquals("berlin-taz.shp", custom.getTAZFileName());
    }

    @Test
    void berlinAndWithinDaySettingsStillApply() {
        Config config = prepareConfig();

        // Berlin ran first: the split activity types only exist because of Activities.addScoringParams.
        ScoringConfigGroup.ActivityParams workParams = config.scoring().getActivityParams("work_37800");
        assertNotNull(workParams, "Berlin's split activity types are missing, super.prepareConfig() did not run");

        // The within-day layer replaced day-to-day replanning, once per subpopulation plus the fallback.
        assertEquals(6, config.replanning().getStrategySettings().size());
        for (ReplanningConfigGroup.StrategySettings settings : config.replanning().getStrategySettings()) {
            assertEquals(DefaultPlanStrategiesModule.DefaultSelector.KeepLastSelected, settings.getStrategyName(),
                    "unexpected replanning strategy left in the config");
        }

        WithinDayConfigGroup withinDay = ConfigUtils.addOrGetModule(config, WithinDayConfigGroup.class);
        assertTrue(withinDay.getAgentIdsAsCollection().size() >= 1, "no explicit training agent configured");
        assertEquals("berlin-mini.modeChoiceModel.xml", WithinDayConfigGroup.getExternalModelFileName());
    }

    private Config prepareConfig() {
        RunBerlinRLModeChoice application = new RunBerlinRLModeChoice();

        // Let picocli fill the command line options (e.g. --plan-selector) with their defaults.
        new CommandLine(application).parseArgs();

        return application.prepareConfig(ConfigUtils.loadConfig(CONFIG));
    }
}
