package modechoice.rl.java.core;

import java.io.File;
import java.util.Set;

import org.matsim.api.core.v01.Scenario;
import org.matsim.api.core.v01.TransportMode;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.controler.AbstractModule;
import org.matsim.core.router.util.TravelTime;
import org.matsim.withinday.core.RunWithinDay;
import org.matsim.withinday.core.WithinDayModeChoiceListener;
import org.matsim.withinday.core.WithinDayReplanner;
import org.matsim.withinday.environment.MatsimScoreTracker;
import org.matsim.withinday.environment.WithinDayObserver;
import org.matsim.withinday.networking.CommunicationManager;
import org.matsim.withinday.networking.UnixSocketCommunicationManager;
import org.matsim.withinday.trafficmonitoring.WithinDayTravelTime;
import org.matsim.withinday.utils.WithinDayConfigGroup;

import modechoice.rl.java.utils.CustomConfigGroup;

public class RunRLModeChoice extends RunWithinDay {
    public static final String REINFORCEMENT_MODE = "rl";

    public static void main(String[] args) {
        new RunRLModeChoice().run(args);
    }

    @Override
    protected Config loadConfiguration(String[] args) {
        String configPath = parseConfigPath(args);

        // Load Configurations with Custom and WithinDay Modules
        CustomConfigGroup customGroupModule = new CustomConfigGroup();
        WithinDayConfigGroup withindayModule = new WithinDayConfigGroup();
        Config config = ConfigUtils.loadConfig(configPath, customGroupModule, withindayModule);

        String modelFileName = withindayModule.getExternalModelFileName();
        if (modelFileName != null && !modelFileName.isEmpty()) {
            File secondaryParamsFile = new File(new File(configPath).getParentFile(), modelFileName);
            if (secondaryParamsFile.exists()) {
                ConfigUtils.loadConfig(secondaryParamsFile.getAbsolutePath(), customGroupModule);
            }
        }
        
        System.out.println("Model File Name: " + modelFileName);

        // Apply command-line parameter overrides
        applyCommandlineOverrides(config, args);
        return config;
    }

    @Override
    protected AbstractModule createModule(Scenario scenario, Config config) {
        // Retrieve WithinDayConfigGroup to check custom replanner/observer configurations
        WithinDayConfigGroup withindayModule = ConfigUtils.addOrGetModule(config, WithinDayConfigGroup.class);
        final String replannerClass = (withindayModule != null) ? withindayModule.getParams().get("replanner") : null;
        final String observerClass = (withindayModule != null) ? withindayModule.getParams().get("observer") : null;

        System.out.println("Replanner Class: " + replannerClass);
        System.out.println("Observer Class: " + observerClass);

        return new AbstractModule() {
            @Override
            public void install() {
                // Inter-Platform Communication Manager
                bind(CommunicationManager.class).to(UnixSocketCommunicationManager.class).asEagerSingleton();
                addControllerListenerBinding().to(UnixSocketCommunicationManager.class);

                // Observer & Replanner
                bindDynamicClass(WithinDayObserver.class, observerClass, CustomRLObserver.class);
                bindDynamicClass(WithinDayReplanner.class, replannerClass, CustomRLReplanner.class);

                // Live access to MATSim's own scoring
                bind(MatsimScoreTracker.class).asEagerSingleton();
                addEventHandlerBinding().to(MatsimScoreTracker.class);

                // Within-Day Travel Time (Tracks 'car' and 'rl' modes)
                //WithinDayTravelTime travelTime = new WithinDayTravelTime(scenario, Set.of(REINFORCEMENT_MODE, TransportMode.car));
                //bind(TravelTime.class).toInstance(travelTime);
                //addEventHandlerBinding().toInstance(travelTime);
                //addMobsimListenerBinding().toInstance(travelTime);

                // Within-Day Mode Choice Listener
                bind(WithinDayModeChoiceListener.class).asEagerSingleton();
                addMobsimListenerBinding().to(WithinDayModeChoiceListener.class);
                addEventHandlerBinding().to(WithinDayModeChoiceListener.class);
                addControllerListenerBinding().to(WithinDayModeChoiceListener.class);
            }

            /**
             * Dynamically binds a class name String to a Guice target interface with type checking and fallback.
             */
            @SuppressWarnings("unchecked")
            private <T> void bindDynamicClass(Class<T> targetInterface, String className, Class<? extends T> defaultClass) {
                if (className == null || className.isBlank() || className.equalsIgnoreCase("default")) {
                    log.info("[WITHINDAY BINDING] No custom input provided for {}. Using DEFAULT class: {}", 
                            targetInterface.getSimpleName(), defaultClass.getName());
                    bind(targetInterface).to(defaultClass).asEagerSingleton();
                    return;
                }

                try {
                    Class<?> clazz;
                    try {
                        clazz = Class.forName(className);
                    } catch (ClassNotFoundException e) {
                        String defaultPackage = defaultClass.getPackageName();
                        clazz = Class.forName(defaultPackage + "." + className);
                    }

                    if (!targetInterface.isAssignableFrom(clazz)) {
                        throw new IllegalArgumentException(String.format(
                            "Configured class '%s' does not implement required interface '%s'", 
                            className, targetInterface.getName()
                        ));
                    }

                    log.info("[WITHINDAY BINDING] Resolved CUSTOM input for {}: {}", 
                            targetInterface.getSimpleName(), clazz.getName());
                    bind(targetInterface).to((Class<? extends T>) clazz).asEagerSingleton();

                } catch (ClassNotFoundException e) {
                    throw new RuntimeException("Could not find dynamic class: " + className, e);
                }
            }
        };
    }
}