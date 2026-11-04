package modechoice.discrete_choice.java.core;

import org.matsim.application.MATSimApplication;
import org.matsim.core.config.Config;
import org.matsim.core.config.groups.QSimConfigGroup;
import org.matsim.core.config.groups.ReplanningConfigGroup;
import org.matsim.core.config.groups.RoutingConfigGroup;
import org.matsim.core.config.groups.ScoringConfigGroup;
import org.matsim.core.controler.AbstractModule;
import org.matsim.core.replanning.strategies.DefaultPlanStrategiesModule;
import org.matsim.withinday.core.RunWithinDay;
import org.matsim.withinday.core.WithinDayModule;

import picocli.CommandLine;

/**
 * <h2>RunDiscreteModeChoiceWithinDay</h2>
 * <p>
 * Runs the discrete choice within-day mode choice on the scenario given behind {@code --config}.
 * In contrast to the other runners day-to-day replanning stays active here, so that MATSim's own
 * subtour mode choice can serve as the reference for the within-day decisions.
 * </p>
 */
@CommandLine.Command(header = ":: Within-Day Mode Choice (discrete choice) ::", mixinStandardHelpOptions = true, showDefaultValues = true)
public class RunDiscreteModeChoiceWithinDay extends RunWithinDay {

    public static void main(String[] args) {
        MATSimApplication.execute(RunDiscreteModeChoiceWithinDay.class, args);
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
    protected AbstractModule createWithinDayModule(Config config) {
        // The scenario has no calibrated scoring of its own, so the split aware scoring factory is bound.
        return new WithinDayModule(config, DiscreteModeChoiceReplanner.class, DiscreteModeChoiceObserver.class, true);
    }
}
