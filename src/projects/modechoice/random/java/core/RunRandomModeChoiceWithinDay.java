package modechoice.random.java.core;

import org.matsim.api.core.v01.Scenario;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.controler.AbstractModule;
import org.matsim.withinday.core.RunWithinDay;
import org.matsim.withinday.core.WithinDayModeChoiceListener;
import org.matsim.withinday.core.WithinDayReplanner;
import org.matsim.withinday.environment.MatsimScoreTracker;
import org.matsim.withinday.environment.WithinDayObserver;
import org.matsim.withinday.networking.CommunicationManager;
import modechoice.random.java.core.RandomModeChoiceObserver;
import modechoice.random.java.core.RandomModeChoiceReplanner;

import org.matsim.withinday.utils.WithinDayConfigGroup;

public class RunRandomModeChoiceWithinDay extends RunWithinDay {

    public static void main(String[] args) {
        new RunRandomModeChoiceWithinDay().run(args);
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
        protected AbstractModule createModule(Scenario scenario, Config config) {
            return new AbstractModule() {
                @Override
                public void install() {
                    // Bind standard MATSim Within-Day components for the baseline
                    bind(WithinDayObserver.class).to(RandomModeChoiceObserver.class).asEagerSingleton();
                    bind(WithinDayReplanner.class).to(RandomModeChoiceReplanner.class).asEagerSingleton();
                    bind(CommunicationManager.class).asEagerSingleton();
                    addControllerListenerBinding().to(CommunicationManager.class);

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
