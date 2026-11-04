package org.matsim.withinday.core;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.matsim.application.MATSimApplication;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.config.groups.QSimConfigGroup;
import org.matsim.core.config.groups.ReplanningConfigGroup;
import org.matsim.core.config.groups.RoutingConfigGroup;
import org.matsim.core.config.groups.ScoringConfigGroup;
import org.matsim.core.controler.AbstractModule;
import org.matsim.core.controler.Controler;
import org.matsim.core.controler.OutputDirectoryHierarchy.OverwriteFileSetting;
import org.matsim.core.replanning.strategies.DefaultPlanStrategiesModule;
import org.matsim.withinday.utils.WithinDayConfigGroup;

import picocli.CommandLine;

/**
 * <h2>RunWithinDay Abstract Runner</h2>
 * <p>
 * Base class for the within-day runners that bring their own scenario, as opposed to the ones
 * derived from an external scenario such as Open Berlin. It is a {@link MATSimApplication}, so
 * every runner in this project is launched the same way - the config file behind {@code --config}
 * and config values overridden with {@code --config:<module>.<param>=<value>} - and it fills the
 * MATSimApplication lifecycle hooks with the within-day defaults: the typed within-day config
 * group, the framework settings replacing day-to-day replanning, the output settings and the
 * within-day Guice bindings.
 * </p>
 */
@CommandLine.Command(header = ":: Within-Day Mode Choice ::", mixinStandardHelpOptions = true, showDefaultValues = true)
public abstract class RunWithinDay extends MATSimApplication {
    protected static final Logger log = LogManager.getLogger(RunWithinDay.class);

    protected RunWithinDay() {
    }

    /**
     * @param config Config given from code, e.g. from a test, instead of {@code --config}.
     */
    protected RunWithinDay(Config config) {
        super(config);
    }

    /**
     * Prepares the configuration of a within-day run. The {@code --config:} overrides are applied
     * by {@link MATSimApplication} afterwards, so the command line still wins over these defaults.
     *
     * @param config The configuration loaded from the config file.
     * @return The prepared configuration.
     */
    @Override
    protected Config prepareConfig(Config config) {
        // Materialize the within-day module from the config file.
        ConfigUtils.addOrGetModule(config, WithinDayConfigGroup.class);

        // Configure Output Directory and File Intervals
        setupOutputSettings(config);

        // Apply MATSim Framework & Replanning Settings
        applyFrameworkSettings(config);

        return config;
    }

    /**
     * Registers the within-day bindings on the controller.
     *
     * @param controler The controller being prepared.
     */
    @Override
    protected void prepareControler(Controler controler) {
        controler.addOverridingModule(createWithinDayModule(controler.getConfig()));
    }

    /**
     * Method for subclasses to supply their own Guice {@link AbstractModule}
     * containing custom bindings and listener registrations.
     *
     * @param config The loaded simulation configuration.
     * @return An {@link AbstractModule} instance for controller overriding.
     */
    protected abstract AbstractModule createWithinDayModule(Config config);

    /**
     * Sets up output directory behaviors and file-writing intervals 
     * (e.g., controlling plan, event, and graph dumping policies). 
     * Can be overridden if scenarios require different r/w intervals than custom runs.
     * 
     * @param config The simulation configuration to modify.
     */
    protected void setupOutputSettings(Config config) {
        config.controller().setOverwriteFileSetting(OverwriteFileSetting.deleteDirectoryIfExists);
        //config.controller().setWritePlansInterval(config.controller().getLastIteration());
        //config.controller().setWriteEventsInterval(config.controller().getLastIteration());
        config.controller().setWritePlansInterval(1);
        config.controller().setWriteEventsInterval(1);
        config.controller().setWriteSnapshotsInterval(0);
        config.controller().setCreateGraphsInterval(1);
        config.controller().setDumpDataAtEnd(false);
    }
    
    /**
     * Sets global routing, scoring parameters, and default strategy execution rules 
     * for the simulation framework.
     * 
     * @param config The simulation configuration to modify.
     */
    protected void applyFrameworkSettings(Config config) {
        config.routing().setNetworkRouteConsistencyCheck(RoutingConfigGroup.NetworkRouteConsistencyCheck.disable);
        config.scoring().addModeParams(new ScoringConfigGroup.ModeParams("walk"));
        config.qsim().setVehiclesSource(QSimConfigGroup.VehiclesSource.modeVehicleTypesFromVehiclesData);

        // Reset replanning strategies to pure 'KeepLastSelected'
        config.replanning().clearStrategySettings();
        config.replanning().addStrategySettings(
            new ReplanningConfigGroup.StrategySettings()
                .setStrategyName(DefaultPlanStrategiesModule.DefaultSelector.KeepLastSelected)
                .setWeight(1.0)
        );
    }
}
