package modechoice.discrete_choice.java.core;

import org.matsim.api.core.v01.Scenario;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.config.groups.QSimConfigGroup;
import org.matsim.core.config.groups.ReplanningConfigGroup;
import org.matsim.core.config.groups.ReplanningConfigGroup.StrategySettings;
import org.matsim.core.config.groups.RoutingConfigGroup;
import org.matsim.core.config.groups.ScoringConfigGroup;
import org.matsim.core.controler.AbstractModule;
import org.matsim.core.replanning.strategies.DefaultPlanStrategiesModule;
import org.matsim.core.scoring.ScoringFunctionFactory;
import org.matsim.withinday.core.AgentSelector;
import org.matsim.withinday.core.RunWithinDay;
import org.matsim.withinday.core.WithinDayModeChoiceListener;
import org.matsim.withinday.core.WithinDayReplanner;
import org.matsim.withinday.environment.MatsimScoreTracker;
import org.matsim.withinday.environment.SplitAwareScoringFunctionFactory;
import org.matsim.withinday.environment.WithinDayObserver;
import org.matsim.withinday.networking.CommunicationManager;
import modechoice.discrete_choice.java.core.DiscreteModeChoiceObserver;
import modechoice.discrete_choice.java.core.DiscreteModeChoiceReplanner;

import org.matsim.withinday.utils.WithinDayConfigGroup;

public class RunDiscreteModeChoiceWithinDay extends RunWithinDay {

    public static void main(String[] args) {
        new RunDiscreteModeChoiceWithinDay().run(args);
    }

    @Override
    protected Config loadConfiguration(String[] args) {
        String configPath = parseConfigPath(args);

        // Baseline only requires the WithinDay module
        WithinDayConfigGroup withindayModule = new WithinDayConfigGroup();
        Config config = ConfigUtils.loadConfig(configPath, withindayModule);

        applyCommandlineOverrides(config, args);
        return config;
    }

    @Override
    protected void applyFrameworkSettings(Config config) {
        config.routing().setNetworkRouteConsistencyCheck(RoutingConfigGroup.NetworkRouteConsistencyCheck.disable);
        config.scoring().addModeParams(new ScoringConfigGroup.ModeParams("walk"));
        config.qsim().setVehiclesSource(QSimConfigGroup.VehiclesSource.modeVehicleTypesFromVehiclesData);

        // Clear existing replanning strategies
        config.replanning().clearStrategySettings();

        ReplanningConfigGroup.StrategySettings primaryStrategy = new ReplanningConfigGroup.StrategySettings();
        primaryStrategy.setStrategyName(DefaultPlanStrategiesModule.DefaultSelector.ChangeExpBeta);
        primaryStrategy.setWeight(0.7);
        config.replanning().addStrategySettings(primaryStrategy);

        // Standard MATSim Mode Choice strategy set to 30% (weight 0.3)
        ReplanningConfigGroup.StrategySettings matsimModeChoice = new ReplanningConfigGroup.StrategySettings();
        matsimModeChoice.setStrategyName(DefaultPlanStrategiesModule.DefaultStrategy.SubtourModeChoice);
        matsimModeChoice.setWeight(0.3);
        config.replanning().addStrategySettings(matsimModeChoice);
        config.subtourModeChoice().setModes(new String[]{"car", "bike", "pt", "pedestrian"});
        config.subtourModeChoice().setChainBasedModes(new String[]{"car", "bike"});
        config.subtourModeChoice().setConsiderCarAvailability(true);
    }

    @Override
        protected AbstractModule createModule(Scenario scenario, Config config) {
            return new AbstractModule() {
                @Override
                public void install() {
                    // Selector
                    bind(AgentSelector.class).asEagerSingleton();

                    // Bind standard MATSim Within-Day components for the baseline
                    bind(WithinDayObserver.class).to(DiscreteModeChoiceObserver.class).asEagerSingleton();
                    bind(WithinDayReplanner.class).to(DiscreteModeChoiceReplanner.class).asEagerSingleton();
                    bind(CommunicationManager.class).asEagerSingleton();
                    addControllerListenerBinding().to(CommunicationManager.class);

                    // Live access to MATSim's standard scoring tracker
                    bind(ScoringFunctionFactory.class).to(SplitAwareScoringFunctionFactory.class).asEagerSingleton();
                    bind(MatsimScoreTracker.class).asEagerSingleton();
                    addEventHandlerBinding().to(MatsimScoreTracker.class);

                    // Within-Day Mode Choice Listener
                    bind(WithinDayModeChoiceListener.class).asEagerSingleton();
                    addMobsimListenerBinding().to(WithinDayModeChoiceListener.class);
                    addEventHandlerBinding().to(WithinDayModeChoiceListener.class);
                    addControllerListenerBinding().to(WithinDayModeChoiceListener.class);
                }
            };
        }
}
