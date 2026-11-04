package org.matsim.withinday.environment;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.Scenario;
import org.matsim.api.core.v01.events.ActivityStartEvent;
import org.matsim.api.core.v01.events.handler.ActivityStartEventHandler;
import org.matsim.api.core.v01.population.Activity;
import org.matsim.api.core.v01.population.Person;
import org.matsim.api.core.v01.population.Plan;
import org.matsim.api.core.v01.population.PlanElement;
import org.matsim.core.controler.events.IterationEndsEvent;
import org.matsim.core.controler.events.IterationStartsEvent;
import org.matsim.core.controler.listener.IterationEndsListener;
import org.matsim.core.controler.listener.IterationStartsListener;
import org.matsim.core.mobsim.framework.events.MobsimAfterSimStepEvent;
import org.matsim.core.mobsim.framework.listeners.MobsimAfterSimStepListener;
import org.matsim.core.population.PopulationUtils;
import org.matsim.core.router.StageActivityTypeIdentifier;

import com.google.inject.Inject;

/**
 * Closes each tracked agent's day on arrival at its final activity - the same instant the within-day
 * replanner does - and collects both the score {@link MatsimScoreTracker} exposes and the score
 * MATSim writes to the selected plan afterwards. Shared by the scenario specific score tracker tests.
 */
public class ScoreComparison implements IterationStartsListener, IterationEndsListener,
        MobsimAfterSimStepListener, ActivityStartEventHandler {

    private final Scenario scenario;
    private final Set<Id<Person>> trackedAgents;

    @Inject private MatsimScoreTracker tracker;

    private final Map<Id<Person>, Integer> realActivityCount = new HashMap<>();
    private final Map<Id<Person>, Integer> activityStartsSeen = new HashMap<>();
    private final Map<Id<Person>, Activity> pendingFinalActivity = new HashMap<>();
    private final Map<Id<Person>, Double> pendingArrivalTime = new HashMap<>();
    private final Map<Id<Person>, double[]> scores = new HashMap<>();

    public ScoreComparison(Scenario scenario, Set<Id<Person>> trackedAgents) {
        this.scenario = scenario;
        this.trackedAgents = trackedAgents;
    }

    @Override
    public void notifyIterationStarts(IterationStartsEvent event) {
        this.tracker.beginIteration(this.trackedAgents);

        this.realActivityCount.clear();
        this.activityStartsSeen.clear();
        this.scores.clear();

        for (Id<Person> agentId : this.trackedAgents) {
            Plan plan = this.scenario.getPopulation().getPersons().get(agentId).getSelectedPlan();
            int realActivities = 0;
            for (PlanElement element : plan.getPlanElements()) {
                if (element instanceof Activity activity
                        && !StageActivityTypeIdentifier.isStageActivity(activity.getType())) {
                    realActivities++;
                }
            }
            this.realActivityCount.put(agentId, realActivities);
        }
    }

    @Override
    public void handleEvent(ActivityStartEvent event) {
        if (StageActivityTypeIdentifier.isStageActivity(event.getActType())
                || !this.trackedAgents.contains(event.getPersonId())) {
            return;
        }

        int seen = this.activityStartsSeen.merge(event.getPersonId(), 1, Integer::sum);

        // The agent's day starts inside its first activity, so no start event is fired for it:
        // the (N-1)th start event is the arrival at the final activity.
        if (seen == this.realActivityCount.getOrDefault(event.getPersonId(), 0) - 1) {
            Activity finalActivity = PopulationUtils.createActivityFromLinkId(event.getActType(), event.getLinkId());
            finalActivity.setFacilityId(event.getFacilityId());
            finalActivity.setStartTime(event.getTime());

            this.pendingFinalActivity.put(event.getPersonId(), finalActivity);
            this.pendingArrivalTime.put(event.getPersonId(), event.getTime());
        }
    }

    /**
     * Deferred to after the sim step so that all events of that step -- including the trip scoring
     * triggered by the arrival itself -- have been dispatched, exactly as the within-day listener does.
     */
    @Override
    public void notifyMobsimAfterSimStep(MobsimAfterSimStepEvent e) {
        if (this.pendingFinalActivity.isEmpty()) {
            return;
        }

        for (Map.Entry<Id<Person>, Activity> entry : this.pendingFinalActivity.entrySet()) {
            double dayScore = this.tracker.finishDay(entry.getKey(), entry.getValue(),
                    this.pendingArrivalTime.get(entry.getKey()));
            this.scores.put(entry.getKey(), new double[] { dayScore, Double.NaN });
        }

        this.pendingFinalActivity.clear();
        this.pendingArrivalTime.clear();
    }

    @Override
    public void notifyIterationEnds(IterationEndsEvent event) {
        for (Map.Entry<Id<Person>, double[]> entry : this.scores.entrySet()) {
            Plan plan = this.scenario.getPopulation().getPersons().get(entry.getKey()).getSelectedPlan();
            entry.getValue()[1] = plan.getScore();
        }
    }

    @Override
    public void reset(int iteration) {
        this.pendingFinalActivity.clear();
        this.pendingArrivalTime.clear();
    }

    public int getComparedAgents() {
        return this.scores.size();
    }

    public Map<Id<Person>, double[]> getScores() {
        return this.scores;
    }
}
