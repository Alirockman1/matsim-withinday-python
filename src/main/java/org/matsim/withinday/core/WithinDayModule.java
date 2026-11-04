package org.matsim.withinday.core;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.controler.AbstractModule;
import org.matsim.core.scoring.ScoringFunctionFactory;
import org.matsim.withinday.environment.MatsimScoreTracker;
import org.matsim.withinday.environment.SplitAwareScoringFunctionFactory;
import org.matsim.withinday.environment.WithinDayObserver;
import org.matsim.withinday.networking.CommunicationManager;
import org.matsim.withinday.utils.WithinDayConfigGroup;

/**
 * <h2>WithinDayModule</h2>
 * <p>
 * The Guice bindings a within-day run needs, extracted so that every runner - the ones derived from
 * {@link RunWithinDay} as well as the ones derived from an external scenario such as Open Berlin -
 * shares one definition. The replanner and observer implementations are resolved from the config
 * params {@code withinday.replanner} / {@code withinday.observer}, which the launchers fill from
 * the {@code REPLANNER_CLASS} / {@code OBSERVER_CLASS} environment variables.
 * </p>
 */
public class WithinDayModule extends AbstractModule {

    private static final Logger log = LogManager.getLogger(WithinDayModule.class);

    private final Class<? extends WithinDayReplanner> defaultReplanner;
    private final Class<? extends WithinDayObserver> defaultObserver;
    private final boolean bindSplitAwareScoring;
    private final String replannerClass;
    private final String observerClass;

    /**
     * @param config                The loaded configuration, used to read the replanner/observer params.
     * @param defaultReplanner      Replanner used when the config does not name one.
     * @param defaultObserver       Observer used when the config does not name one.
     * @param bindSplitAwareScoring Whether {@link SplitAwareScoringFunctionFactory} replaces the
     *                              scenario's scoring function factory. Must stay {@code false} for
     *                              scenarios that bring their own calibrated scoring.
     */
    public WithinDayModule(Config config,
                           Class<? extends WithinDayReplanner> defaultReplanner,
                           Class<? extends WithinDayObserver> defaultObserver,
                           boolean bindSplitAwareScoring) {
        this.defaultReplanner = defaultReplanner;
        this.defaultObserver = defaultObserver;
        this.bindSplitAwareScoring = bindSplitAwareScoring;

        WithinDayConfigGroup withindayModule = ConfigUtils.addOrGetModule(config, WithinDayConfigGroup.class);
        this.replannerClass = (withindayModule != null) ? withindayModule.getParams().get("replanner") : null;
        this.observerClass = (withindayModule != null) ? withindayModule.getParams().get("observer") : null;
    }

    @Override
    public void install() {
        // Inter-Platform Communication Manager
        bind(CommunicationManager.class).asEagerSingleton();
        addControllerListenerBinding().to(CommunicationManager.class);

        // Selector
        bind(AgentSelector.class).asEagerSingleton();

        // Observer & Replanner
        bindDynamicClass(WithinDayReplanner.class, this.replannerClass, this.defaultReplanner);
        bindDynamicClass(WithinDayObserver.class, this.observerClass, this.defaultObserver);

        // Live access to the scoring factory
        if (this.bindSplitAwareScoring) {
            bind(ScoringFunctionFactory.class).to(SplitAwareScoringFunctionFactory.class).asEagerSingleton();
        }
        bind(MatsimScoreTracker.class).asEagerSingleton();
        addEventHandlerBinding().to(MatsimScoreTracker.class);

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
}
