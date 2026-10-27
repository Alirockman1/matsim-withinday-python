package org.matsim.withinday.environment;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.List;
import java.util.Set;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.matsim.api.core.v01.Scenario;
import org.matsim.api.core.v01.events.Event;
import org.matsim.api.core.v01.population.Activity;
import org.matsim.api.core.v01.population.Leg;
import org.matsim.api.core.v01.population.Person;
import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigGroup;
import org.matsim.core.router.TripStructureUtils.Trip;
import org.matsim.core.scoring.ScoringFunction;
import org.matsim.core.scoring.ScoringFunctionFactory;
import org.matsim.core.scoring.functions.CharyparNagelScoringFunctionFactory;

import com.google.inject.Inject;
import com.google.inject.Singleton;

/**
 * Scoring function factory that stops MATSim from punishing <em>split</em> strict activities.
 * <p>
 * Strict activities (work, school) have time windows (latestStartTime, ...). If an agent leaves
 * work for lunch and comes back, the second part starts "very late" and would take a large
 * late-arrival penalty, and its short duration scores badly. This factory wraps the standard
 * scoring function and, for such a <em>continuation</em> part only, replaces the part's score by
 * the marginal duration utility
 * <pre>    f(D_previous + d) - f(D_previous)</pre>
 * where D_previous is the time already spent in that activity today and f is MATSim's
 * logarithmic duration utility. A work day with a lunch break therefore sums to exactly the
 * same work utility as one unsplit block, and every part still receives its score immediately
 * (so step rewards read through {@link MatsimScoreTracker} keep working).
 * <p>
 * Everything else (first parts, other activity types, legs, money, stuck) stays as in the
 * wrapped factory. {@link MatsimScoreTracker} needs no change: it receives this factory through
 * injection. The standard end-of-iteration MATSim scores use it as well, so both agree.
 * <p>
 * A part counts as a continuation of the same strict activity if the same strict type was
 * already scored earlier today, the agent has not been home since, and the break between the two
 * parts is at most {@link #MAX_BREAK_S}.
 */

@Singleton
public class SplitAwareScoringFunctionFactory implements ScoringFunctionFactory {
 
    private static final Logger log = LogManager.getLogger(SplitAwareScoringFunctionFactory.class);
 
    /** Activity types treated as strict. Only those that exist in the scoring config are used. */
    private static final Set<String> CANDIDATE_STRICT_TYPES = Set.of("work", "education", "school");
 
    /** Longest gap between two parts that still counts as one split activity (3 hours). */
    private static final double MAX_BREAK_S = 3 * 3600.0;
 
    /** MATSim defaults, used only if the config does not set the parameter. */
    private static final double DEFAULT_PERFORMING_UTILS_HR = 6.0;
    private static final double DEFAULT_WAITING_UTILS_HR = 0.0;
 
    /** Typical duration (seconds) and priority of one strict activity type. */
    private record ActivityShape(double typicalSeconds, double priority) {}
 
    private final ScoringFunctionFactory delegate;
    private final Map<String, ActivityShape> strictActivities = new HashMap<>();
    private final double performingPerSecond;
    private final double waitingPerSecond;
 
    /**
     * @param delegate The standard factory that builds the actual scoring functions.
     *                 Replace this type with your own factory if you already bind one.
     * @param config   The MATSim config (scoring module is read for activity parameters).
     */
    @Inject
    public SplitAwareScoringFunctionFactory(Scenario scenario) {
        this.delegate = new CharyparNagelScoringFunctionFactory(scenario);
        Config config = scenario.getConfig();
 
        // Generic ConfigGroup access only (no version-specific ScoringConfigGroup methods)
        ConfigGroup scoring = config.getModules().get("scoring");
        if (scoring == null) {
            scoring = config.getModules().get("planCalcScore");
        }
 
        Map<String, Map<String, String>> activityParamsByType = new HashMap<>();
        if (scoring == null) {
            log.warn("Split-aware scoring: no 'scoring' module found in the config; scores are left unchanged.");
            this.performingPerSecond = DEFAULT_PERFORMING_UTILS_HR / 3600.0;
            this.waitingPerSecond = DEFAULT_WAITING_UTILS_HR / 3600.0;
        } else {
            this.performingPerSecond = parseNumber(findScoringParam(scoring, "performing"), DEFAULT_PERFORMING_UTILS_HR) / 3600.0;
            this.waitingPerSecond = parseNumber(findScoringParam(scoring, "waiting"), DEFAULT_WAITING_UTILS_HR) / 3600.0;
            collectActivityParams(scoring, activityParamsByType);
            for (ConfigGroup defaultSet : defaultSubpopulationSets(scoring)) {
                collectActivityParams(defaultSet, activityParamsByType);
            }
        }
 
        for (String type : CANDIDATE_STRICT_TYPES) {
            Map<String, String> p = activityParamsByType.get(type);
            if (p == null) {
                continue;   // "if it exists": not defined in this scenario
            }
            double typicalSeconds = parseSeconds(p.get("typicalDuration"));
            double priority = parseNumber(p.get("priority"), 1.0);
 
            if (Double.isNaN(typicalSeconds) || typicalSeconds <= 0.0 || priority <= 0.0) {
                log.warn("Split-aware scoring: activity type '{}' has no usable typicalDuration/priority; leaving it unchanged.", type);
                continue;
            }
            this.strictActivities.put(type, new ActivityShape(typicalSeconds, priority));
        }
 
        log.info("Split-aware scoring active for activity types {} (candidates: {}).",
                this.strictActivities.keySet(), CANDIDATE_STRICT_TYPES);
    }
 
    @Override
    public ScoringFunction createNewScoringFunction(Person person) {
        return new SplitAwareScoringFunction(this.delegate.createNewScoringFunction(person),
                this.strictActivities, this.performingPerSecond, this.waitingPerSecond);
    }
 
    /** The default-subpopulation 'scoringParameters' sets nested in the scoring module, if any. */
    private static List<ConfigGroup> defaultSubpopulationSets(ConfigGroup scoring) {
        List<ConfigGroup> sets = new ArrayList<>();
        for (ConfigGroup set : scoring.getParameterSets("scoringParameters")) {
            String subpopulation = set.getParams().get("subpopulation");
            if (subpopulation == null || subpopulation.isBlank() || subpopulation.equals("null")) {
                sets.add(set);
            }
        }
        return sets;
    }
 
    /** Looks a scoring parameter up directly under the module, then in the default subpopulation set. */
    private static String findScoringParam(ConfigGroup scoring, String name) {
        String value = scoring.getParams().get(name);
        if (value != null) {
            return value;
        }
        for (ConfigGroup set : defaultSubpopulationSets(scoring)) {
            value = set.getParams().get(name);
            if (value != null) {
                return value;
            }
        }
        return null;
    }
 
    /** Collects the parameters of every 'activityParams' set held by the given group, by activity type. */
    private static void collectActivityParams(ConfigGroup holder, Map<String, Map<String, String>> result) {
        for (ConfigGroup set : holder.getParameterSets("activityParams")) {
            Map<String, String> params = set.getParams();
            String type = params.get("activityType");
            if (type != null) {
                result.putIfAbsent(type, params);
            }
        }
    }
 
    /** Parses a plain number, returning the fallback if missing or invalid. */
    private static double parseNumber(String value, double fallback) {
        if (value == null) return fallback;
        try {
            return Double.parseDouble(value.trim());
        } catch (NumberFormatException e) {
            return fallback;
        }
    }
 
    /** Parses "hh:mm:ss", "hh:mm" or plain seconds; returns NaN if missing or invalid. */
    private static double parseSeconds(String value) {
        if (value == null || value.isBlank() || value.equalsIgnoreCase("undefined")) return Double.NaN;
        try {
            String[] parts = value.trim().split(":");
            if (parts.length == 1) return Double.parseDouble(parts[0]);
            double seconds = 0.0;
            double[] factors = {3600.0, 60.0, 1.0};
            for (int i = 0; i < parts.length && i < 3; i++) {
                seconds += Double.parseDouble(parts[i]) * factors[i];
            }
            return seconds;
        } catch (NumberFormatException e) {
            return Double.NaN;
        }
    }
 
    /** Wraps a standard scoring function and corrects the score of continuation parts. */
    private static final class SplitAwareScoringFunction implements ScoringFunction {
 
        private final ScoringFunction delegate;
        private final Map<String, ActivityShape> strictActivities;
        private final double performingPerSecond;
        private final double waitingPerSecond;
 
        private final Map<String, Double> cumulativeDuration = new HashMap<>();
        private final Map<String, Double> lastEndTime = new HashMap<>();
        private final Set<String> openTypes = new HashSet<>();
 
        private double adjustment = 0.0;
 
        SplitAwareScoringFunction(ScoringFunction delegate, Map<String, ActivityShape> strictActivities,
                                  double performingPerSecond, double waitingPerSecond) {
            this.delegate = delegate;
            this.strictActivities = strictActivities;
            this.performingPerSecond = performingPerSecond;
            this.waitingPerSecond = waitingPerSecond;
        }
 
        @Override
        public void handleActivity(Activity activity) {
            // First and last activity of the day have an undefined start / end: untouched.
            if (activity.getStartTime().isUndefined() || activity.getEndTime().isUndefined()) {
                this.delegate.handleActivity(activity);
                return;
            }
 
            String type = activity.getType();
 
            // Going home ends any open strict activity (a new work visit later is a new visit).
            if (type.startsWith("home")) {
                this.openTypes.clear();
                this.delegate.handleActivity(activity);
                return;
            }
 
            if (!this.strictActivities.containsKey(type)) {
                this.delegate.handleActivity(activity);
                return;
            }
 
            double start = activity.getStartTime().seconds();
            double end = activity.getEndTime().seconds();
            double duration = Math.max(0.0, end - start);
 
            boolean continuation = this.openTypes.contains(type)
                    && (start - this.lastEndTime.getOrDefault(type, Double.NEGATIVE_INFINITY)) <= MAX_BREAK_S;
 
            double before = this.delegate.getScore();
            this.delegate.handleActivity(activity);
            double nativeDelta = this.delegate.getScore() - before;
 
            double previous = continuation ? this.cumulativeDuration.getOrDefault(type, 0.0) : 0.0;
 
            if (continuation) {
                double desired = durationUtility(type, previous + duration) - durationUtility(type, previous);
                this.adjustment += desired - nativeDelta;
            }
 
            this.cumulativeDuration.put(type, previous + duration);
            this.lastEndTime.put(type, end);
            this.openTypes.add(type);
        }
 
        /** MATSim's activity duration utility (Charypar-Nagel), without any time-window terms. */
        private double durationUtility(String type, double durationSeconds) {
            ActivityShape shape = this.strictActivities.get(type);
            if (shape == null || durationSeconds <= 0.0) {
                return 0.0;
            }
            double zeroUtilityDuration = shape.typicalSeconds()
                    * Math.exp(-10.0 / (shape.typicalSeconds() / 3600.0) / shape.priority());
 
            double utilPerforming = this.performingPerSecond * shape.typicalSeconds()
                    * Math.log(durationSeconds / zeroUtilityDuration);
            double utilWaiting = this.waitingPerSecond * durationSeconds;
 
            return Math.max(0.0, Math.max(utilPerforming, utilWaiting));
        }
 
        @Override
        public double getScore() {
            return this.delegate.getScore() + this.adjustment;
        }
 
        @Override
        public void handleLeg(Leg leg) {this.delegate.handleLeg(leg);}
 
        @Override
        public void handleTrip(Trip trip) {this.delegate.handleTrip(trip);}
 
        @Override
        public void addMoney(double amount) {this.delegate.addMoney(amount);}
 
        @Override
        public void addScore(double amount) {this.delegate.addScore(amount);}
 
        @Override
        public void agentStuck(double time) {this.delegate.agentStuck(time);}
 
        @Override
        public void finish() {this.delegate.finish();}
 
        @Override
        public void handleEvent(Event event) {this.delegate.handleEvent(event);}
    }
}
