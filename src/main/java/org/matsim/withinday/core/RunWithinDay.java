package org.matsim.withinday.core;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.matsim.api.core.v01.Scenario;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.config.groups.QSimConfigGroup;
import org.matsim.core.config.groups.ReplanningConfigGroup;
import org.matsim.core.config.groups.RoutingConfigGroup;
import org.matsim.core.config.groups.ScoringConfigGroup;
import org.matsim.core.controler.AbstractModule;
import org.matsim.core.controler.Controller;
import org.matsim.core.controler.ControllerUtils;
import org.matsim.core.controler.OutputDirectoryHierarchy.OverwriteFileSetting;
import org.matsim.core.replanning.strategies.DefaultPlanStrategiesModule;
import org.matsim.core.scenario.ScenarioUtils;

/**
 * <h2>RunWithinDay Abstract Runner</h2>
 * <p>
 * Serves as the extensible orchestrator and template for running multi-agent 
 * scenario simulations. It standardizes the initialization lifecycle, configuration 
 * parsing, output directory settings, framework defaults, and dependency injection 
 * hooks while allowing baseline and custom scenario runners to customize specific 
 * hooks as needed.
 * </p>
 */
public abstract class RunWithinDay {
    protected static final Logger log = LogManager.getLogger(RunWithinDay.class);

    public final void run(String[] args) {
        // Load Configurations (handled by subclass to decide if custom groups are included)
        Config config = loadConfiguration(args);

        // Configure Output Directory and File Intervals
        setupOutputSettings(config);

        // Apply MATSim Framework & Replanning Settings
        applyFrameworkSettings(config);

        // Load Scenario & Create Controller
        Scenario scenario = ScenarioUtils.loadScenario(config);
        Controller controller = ControllerUtils.createController(scenario);

        // Register Guice Bindings via Subclass Hook
        controller.addOverridingModule(createModule(scenario, config));

        // Execute Simulation
        controller.run();
    }

    /**
     * Subclasses implement this to load configuration modules specific to 
     * baseline or custom multi-agent scenario runs.
     * 
     * @param args Command-line arguments containing configuration paths or overrides.
     * @return The fully populated MATSim {@link Config} instance.
     */
    protected abstract Config loadConfiguration(String[] args);

    /**
     * Method for subclasses to supply their own Guice {@link AbstractModule} 
     * containing custom bindings and listener registrations.
     * 
     * @param scenario The loaded simulation scenario.
     * @param config   The loaded simulation configuration.
     * @return An {@link AbstractModule} instance for controller overriding.
     */
    protected abstract AbstractModule createModule(Scenario scenario, Config config);

    /**
     * Sets up output directory behaviors and file-writing intervals 
     * (e.g., controlling plan, event, and graph dumping policies). 
     * Can be overridden if scenarios require different r/w intervals than custom runs.
     * 
     * @param config The simulation configuration to modify.
     */
    protected void setupOutputSettings(Config config) {
        config.controller().setOverwriteFileSetting(OverwriteFileSetting.deleteDirectoryIfExists);
        config.controller().setWritePlansInterval(config.controller().getLastIteration());
        config.controller().setWriteEventsInterval(config.controller().getLastIteration());
        //config.controller().setWritePlansInterval(1);
        //config.controller().setWriteEventsInterval(1);
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

    /**
     * Utility method for parsing configuration file paths from command-line parameters.
     * 
     * @param args Command-line arguments array.
     * @return The resolved configuration file path string.
     */
    protected String parseConfigPath(String[] args) {
        return (args != null && args.length > 0 && args[0] != null) 
                ? args[0] 
                : "scenarios/sioux-falls/input/config.xml";
    }

    /**
     * Applies dynamic command-line overrides onto the loaded configuration object.
     * 
     * @param config The configuration to update.
     * @param args   Command-line arguments array containing override pairs.
     */
    protected void applyCommandlineOverrides(Config config, String[] args) {
        if (args != null && args.length > 1) {
            String[] overrides = new String[args.length - 1];
            System.arraycopy(args, 1, overrides, 0, overrides.length);
            ConfigUtils.applyCommandline(config, overrides);
        }
    }
}