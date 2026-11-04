package modechoice.basic.java.core;

import org.matsim.application.MATSimApplication;
import org.matsim.core.config.Config;
import org.matsim.core.controler.AbstractModule;
import org.matsim.withinday.core.AgentSelector;
import org.matsim.withinday.core.RunWithinDay;
import org.matsim.withinday.core.WithinDayModeChoiceListener;
import org.matsim.withinday.core.WithinDayReplanner;
import org.matsim.withinday.environment.MatsimScoreTracker;
import org.matsim.withinday.environment.WithinDayObserver;

import picocli.CommandLine;

/**
 * <h2>RunModeChoiceWithinDay</h2>
 * <p>
 * The bare within-day skeleton on the scenario given behind {@code --config}: the mode choice
 * listener and the scoring tracker, without the python bridge. It binds the framework base classes
 * directly, so a concrete replanner and observer have to be supplied before it can run.
 * </p>
 */
@CommandLine.Command(header = ":: Within-Day Mode Choice (baseline) ::", mixinStandardHelpOptions = true, showDefaultValues = true)
public class RunModeChoiceWithinDay extends RunWithinDay {

    public static void main(String[] args) {
        MATSimApplication.execute(RunModeChoiceWithinDay.class, args);
    }

    @Override
    protected AbstractModule createWithinDayModule(Config config) {
        return new AbstractModule() {
            @Override
            public void install() {
                // Selector
                bind(AgentSelector.class).asEagerSingleton();

                // Bind standard MATSim Within-Day baseline components
                bind(WithinDayObserver.class).asEagerSingleton();
                bind(WithinDayReplanner.class).asEagerSingleton();

                // Live access to MATSim's standard scoring tracker
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
