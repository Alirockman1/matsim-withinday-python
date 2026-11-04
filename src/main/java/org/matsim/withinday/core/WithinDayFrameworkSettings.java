package org.matsim.withinday.core;

import java.util.Collection;

import org.matsim.api.core.v01.TransportMode;
import org.matsim.core.config.Config;
import org.matsim.core.config.groups.QSimConfigGroup;
import org.matsim.core.config.groups.ReplanningConfigGroup;
import org.matsim.core.config.groups.RoutingConfigGroup;
import org.matsim.core.config.groups.VspExperimentalConfigGroup;
import org.matsim.core.controler.OutputDirectoryHierarchy.OverwriteFileSetting;
import org.matsim.core.replanning.strategies.DefaultPlanStrategiesModule;

/**
 * <h2>WithinDayFrameworkSettings</h2>
 * <p>
 * Framework defaults every within-day run needs, independent of the scenario it runs on. Within-day
 * replanning replaces day-to-day replanning, so all innovative strategies are removed; in contrast
 * to {@link RunWithinDay#applyFrameworkSettings(Config)} the strategies are registered per
 * subpopulation and existing mode scoring parameters are preserved, which is what a calibrated
 * scenario such as Open Berlin requires.
 * </p>
 */
public final class WithinDayFrameworkSettings {

    private WithinDayFrameworkSettings() {
    }

    /**
     * Applies routing, mobsim and replanning settings for a within-day run.
     *
     * @param config         The configuration to modify.
     * @param subpopulations Subpopulations present in the population; a selector strategy is
     *                       registered for each of them plus once without a subpopulation.
     */
    public static void apply(Config config, Collection<String> subpopulations) {
        config.routing().setNetworkRouteConsistencyCheck(RoutingConfigGroup.NetworkRouteConsistencyCheck.disable);
        config.qsim().setVehiclesSource(QSimConfigGroup.VehiclesSource.modeVehicleTypesFromVehiclesData);

        // Create walk params only if the scenario does not define them; addModeParams() would
        // replace calibrated values.
        config.scoring().getOrCreateDefaultModeParams(TransportMode.walk);

        // Within-day replanning is the only plan modification, so keep the executed plan.
        config.replanning().clearStrategySettings();
        config.replanning().addStrategySettings(keepLastSelected(null));
        if (subpopulations != null) {
            for (String subpopulation : subpopulations) {
                if (subpopulation != null && !subpopulation.isBlank()) {
                    config.replanning().addStrategySettings(keepLastSelected(subpopulation));
                }
            }
        }

        // Without innovative strategies there is nothing left to anneal.
        config.replanningAnnealer().setActivateAnnealingModule(false);

        // Removing ChangeExpBeta knowingly violates the VSP defaults; do not abort because of it.
        config.vspExperimental().setVspDefaultsCheckingLevel(VspExperimentalConfigGroup.VspDefaultsCheckingLevel.warn);
    }

    /**
     * Output settings for a within-day run. Write intervals are left untouched, since large
     * scenarios produce output files that are expensive to write every iteration.
     *
     * @param config The configuration to modify.
     */
    public static void applyOutputSettings(Config config) {
        // Not deleteDirectoryIfExists: in the containerized layout the output directory is the bind
        // mount itself, and MATSim's recursive delete ends by removing that directory, which fails
        // with "Device or resource busy". Stale files from an earlier, longer run are left behind
        // instead; the launcher scripts empty the directory themselves.
        config.controller().setOverwriteFileSetting(OverwriteFileSetting.overwriteExistingFiles);
        config.controller().setWriteSnapshotsInterval(0);
        config.controller().setDumpDataAtEnd(false);
    }

    private static ReplanningConfigGroup.StrategySettings keepLastSelected(String subpopulation) {
        ReplanningConfigGroup.StrategySettings settings = new ReplanningConfigGroup.StrategySettings()
                .setStrategyName(DefaultPlanStrategiesModule.DefaultSelector.KeepLastSelected)
                .setWeight(1.0);
        if (subpopulation != null) {
            settings.setSubpopulation(subpopulation);
        }
        return settings;
    }
}
