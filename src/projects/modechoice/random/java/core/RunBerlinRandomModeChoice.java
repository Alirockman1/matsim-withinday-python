package modechoice.random.java.core;

import java.util.List;

import org.matsim.application.MATSimApplication;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.controler.AbstractModule;
import org.matsim.core.controler.Controler;
import org.matsim.run.OpenBerlinScenario;
import org.matsim.simwrapper.SimWrapperConfigGroup;
import org.matsim.withinday.core.WithinDayFrameworkSettings;
import org.matsim.withinday.core.WithinDayModule;
import org.matsim.withinday.utils.WithinDayConfigGroup;

import picocli.CommandLine;

/**
 * <h2>RunBerlinRandomModeChoice</h2>
 * <p>
 * Runs the within-day mode choice baseline on the MATSim Open Berlin scenario. It builds on
 * {@link OpenBerlinScenario} so that the whole Berlin setup - scoring parameters including the
 * split activity types, vehicles, bicycle and emissions settings, SimWrapper - stays in place, and
 * only adds the within-day layer on top of it.
 * </p>
 */
@CommandLine.Command(header = ":: Open Berlin Scenario (within-day mode choice) ::", version = OpenBerlinScenario.VERSION,
        mixinStandardHelpOptions = true, showDefaultValues = true)
public class RunBerlinRandomModeChoice extends OpenBerlinScenario {

    /**
     * Subpopulations of the Berlin population; each one needs its own replanning strategy, because
     * MATSim resolves strategies per subpopulation.
     */
    private static final List<String> SUBPOPULATIONS = List.of(
            "person", "freight", "goodsTraffic", "commercialPersonTraffic", "commercialPersonTraffic_service");

    public static void main(String[] args) {
        MATSimApplication.execute(RunBerlinRandomModeChoice.class, args);
    }

    @Override
    protected Config prepareConfig(Config config) {
        // Berlin first: scoring params, strategies, emissions, bicycle, SimWrapper.
        super.prepareConfig(config);

        // Materialize the within-day module from the config file.
        ConfigUtils.addOrGetModule(config, WithinDayConfigGroup.class);

        // Then replace what within-day replanning has to own.
        WithinDayFrameworkSettings.apply(config, SUBPOPULATIONS);
        WithinDayFrameworkSettings.applyOutputSettings(config);

        // The dashboards analyse a full scenario run and are of no use for a within-day sample.
        ConfigUtils.addOrGetModule(config, SimWrapperConfigGroup.class)
                .setDefaultDashboards(SimWrapperConfigGroup.DefaultDashboardsMode.disabled);

        return config;
    }

    @Override
    protected void prepareControler(Controler controler) {
        // Berlin's own modules first, the within-day module last so that it wins.
        super.prepareControler(controler);
        controler.addOverridingModule(createWithinDayModule(controler.getConfig()));
    }

    /**
     * Hook for tests and for other mode choice flavours.
     *
     * @param config The loaded configuration.
     * @return The module holding the within-day bindings.
     */
    protected AbstractModule createWithinDayModule(Config config) {
        // Berlin brings its own scoring function factory, so SplitAwareScoringFunctionFactory stays off.
        return new WithinDayModule(config, RandomModeChoiceReplanner.class, RandomModeChoiceObserver.class, false);
    }
}
