package modechoice.random.java.core;

import org.matsim.application.MATSimApplication;
import org.matsim.core.config.Config;
import org.matsim.core.controler.AbstractModule;
import org.matsim.withinday.core.RunWithinDay;
import org.matsim.withinday.core.WithinDayModule;

import picocli.CommandLine;

/**
 * <h2>RunRandomModeChoiceWithinDay</h2>
 * <p>
 * Runs the random within-day mode choice baseline on the scenario given behind {@code --config}.
 * </p>
 */
@CommandLine.Command(header = ":: Within-Day Mode Choice (random) ::", mixinStandardHelpOptions = true, showDefaultValues = true)
public class RunRandomModeChoiceWithinDay extends RunWithinDay {

    public static void main(String[] args) {
        MATSimApplication.execute(RunRandomModeChoiceWithinDay.class, args);
    }

    @Override
    protected AbstractModule createWithinDayModule(Config config) {
        // The scenario has no calibrated scoring of its own, so the split aware scoring factory is bound.
        return new WithinDayModule(config, RandomModeChoiceReplanner.class, RandomModeChoiceObserver.class, true);
    }
}
