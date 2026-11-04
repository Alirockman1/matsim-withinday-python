package modechoice.rl.java.core;

import org.matsim.application.MATSimApplication;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.controler.AbstractModule;
import org.matsim.withinday.core.RunWithinDay;
import org.matsim.withinday.core.WithinDayModule;
import org.matsim.withinday.utils.WithinDayConfigGroup;

import modechoice.rl.java.utils.CustomConfigGroup;
import picocli.CommandLine;

/**
 * <h2>RunRLModeChoice</h2>
 * <p>
 * The reinforcement learning flavour of the within-day mode choice on the scenario given behind
 * {@code --config}. The hyperparameters live in a second config file, named by
 * {@code withinday.externalModelFileName} and resolved next to the scenario config. It is read in
 * {@code prepareConfig}, which {@link MATSimApplication} calls before it applies the
 * {@code --config:} overrides, so the values the launcher passes on the command line still win.
 * </p>
 */
@CommandLine.Command(header = ":: Within-Day Mode Choice (reinforcement learning) ::",
        mixinStandardHelpOptions = true, showDefaultValues = true)
public class RunRLModeChoice extends RunWithinDay {
    public static final String REINFORCEMENT_MODE = "rl";

    public static void main(String[] args) {
        MATSimApplication.execute(RunRLModeChoice.class, args);
    }

    @Override
    protected Config prepareConfig(Config config) {
        // Within-day config group, framework and output settings.
        super.prepareConfig(config);

        CustomConfigGroup customConfigGroup = ConfigUtils.addOrGetModule(config, CustomConfigGroup.class);
        WithinDayConfigGroup.loadExternalModelConfig(config, customConfigGroup);

        return config;
    }

    @Override
    protected AbstractModule createWithinDayModule(Config config) {
        // The scenario has no calibrated scoring of its own, so the split aware scoring factory is bound.
        return new WithinDayModule(config, CustomRLReplanner.class, CustomRLObserver.class, true);
    }
}
