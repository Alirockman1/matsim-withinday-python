package modechoice.random.java.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.HashSet;
import java.util.Set;

import org.junit.jupiter.api.Test;
import org.matsim.api.core.v01.TransportMode;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.config.groups.QSimConfigGroup;
import org.matsim.core.config.groups.ReplanningConfigGroup;
import org.matsim.core.config.groups.RoutingConfigGroup;
import org.matsim.core.config.groups.ScoringConfigGroup;
import org.matsim.core.config.groups.VspExperimentalConfigGroup;
import org.matsim.core.controler.OutputDirectoryHierarchy;
import org.matsim.core.replanning.strategies.DefaultPlanStrategiesModule;
import org.matsim.simwrapper.SimWrapperConfigGroup;
import org.matsim.withinday.utils.WithinDayConfigGroup;

import picocli.CommandLine;

/**
 * <h2>RunBerlinModeChoiceConfigTest</h2>
 * <p>
 * Guards the config preparation of {@link RunBerlinRandomModeChoice}: the Open Berlin setup has to
 * survive, and the within-day settings have to be applied on top of it. Only the config is built,
 * so no scenario input is read - the test needs neither network access nor the mirrored input
 * files. It lives in the runner's package to reach the protected {@code prepareConfig} hook.
 * </p>
 */
public class RunBerlinModeChoiceConfigTest {

    private static final String CONFIG = "scenarios/berlin-v7.1.1/input/berlin-mini.config.xml";

    /** Subpopulations the Berlin population contains, plus the fallback without a subpopulation. */
    private static final Set<String> EXPECTED_SUBPOPULATIONS = new HashSet<>(java.util.Arrays.asList(
            null, "person", "freight", "goodsTraffic", "commercialPersonTraffic", "commercialPersonTraffic_service"));

    @Test
    void withinDaySettingsAreAppliedOnTopOfBerlin() {
        Config config = prepareConfig();

        // Berlin ran first: the split activity types only exist because of Activities.addScoringParams.
        ScoringConfigGroup.ActivityParams workParams = config.scoring().getActivityParams("work_37800");
        assertNotNull(workParams, "Berlin's split activity types are missing, super.prepareConfig() did not run");

        // Berlin's calibrated walk params must not be replaced by MATSim defaults.
        ScoringConfigGroup.ModeParams walkParams = config.scoring().getModes().get(TransportMode.walk);
        assertNotNull(walkParams, "walk mode params are missing");
        assertEquals(0.0, walkParams.getConstant(), 1e-9, "Berlin's walk constant was overwritten");
        assertEquals(0.0, walkParams.getMarginalUtilityOfTraveling(), 1e-9,
                "Berlin's marginal utility of walking was overwritten");

        // Within-day replanning replaces day-to-day replanning, once per subpopulation.
        Set<String> subpopulations = new HashSet<>();
        for (ReplanningConfigGroup.StrategySettings settings : config.replanning().getStrategySettings()) {
            assertEquals(DefaultPlanStrategiesModule.DefaultSelector.KeepLastSelected, settings.getStrategyName(),
                    "unexpected replanning strategy left in the config");
            assertEquals(1.0, settings.getWeight(), 1e-9);
            subpopulations.add(settings.getSubpopulation());
        }
        assertEquals(EXPECTED_SUBPOPULATIONS, subpopulations);
        assertFalse(config.replanningAnnealer().isActivateAnnealingModule(), "annealing has no strategies to anneal");

        // Framework settings for within-day replanning.
        assertEquals(RoutingConfigGroup.NetworkRouteConsistencyCheck.disable,
                config.routing().getNetworkRouteConsistencyCheck());
        assertEquals(QSimConfigGroup.VehiclesSource.modeVehicleTypesFromVehiclesData,
                config.qsim().getVehiclesSource());
        assertEquals(VspExperimentalConfigGroup.VspDefaultsCheckingLevel.warn,
                config.vspExperimental().getVspDefaultsCheckingLevel());
        assertEquals(SimWrapperConfigGroup.DefaultDashboardsMode.disabled,
                ConfigUtils.addOrGetModule(config, SimWrapperConfigGroup.class).getDefaultDashboards());

        // deleteDirectoryIfExists would try to remove the output directory itself, which is the
        // bind mount in the containerized layout and fails there with "Device or resource busy".
        assertEquals(OutputDirectoryHierarchy.OverwriteFileSetting.overwriteExistingFiles,
                config.controller().getOverwriteFileSetting());
    }

    @Test
    void withinDayModuleIsMaterialized() {
        Config config = prepareConfig();

        assertInstanceOf(WithinDayConfigGroup.class, config.getModule(WithinDayConfigGroup.GROUP_NAME),
                "the withinday module was not materialized into its typed config group");

        WithinDayConfigGroup withinDay = ConfigUtils.addOrGetModule(config, WithinDayConfigGroup.class);
        assertTrue(withinDay.getAgentIdsAsCollection().size() >= 1, "no explicit training agent configured");
    }

    private Config prepareConfig() {
        RunBerlinRandomModeChoice application = new RunBerlinRandomModeChoice();

        // Let picocli fill the command line options (e.g. --plan-selector) with their defaults.
        new CommandLine(application).parseArgs();

        return application.prepareConfig(ConfigUtils.loadConfig(CONFIG));
    }
}
