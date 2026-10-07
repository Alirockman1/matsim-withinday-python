package org.matsim.withinday.environment;

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.events.ActivityStartEvent;
import org.matsim.api.core.v01.events.PersonMoneyEvent;
import org.matsim.api.core.v01.events.PersonScoreEvent;
import org.matsim.api.core.v01.events.PersonStuckEvent;
import org.matsim.api.core.v01.events.handler.ActivityStartEventHandler;
import org.matsim.api.core.v01.events.handler.PersonMoneyEventHandler;
import org.matsim.api.core.v01.events.handler.PersonScoreEventHandler;
import org.matsim.api.core.v01.events.handler.PersonStuckEventHandler;
import org.matsim.api.core.v01.population.Activity;
import org.matsim.api.core.v01.population.Person;
import org.matsim.api.core.v01.population.Plan;
import org.matsim.api.core.v01.population.Population;
import org.matsim.core.population.PopulationUtils;
import org.matsim.core.router.StageActivityTypeIdentifier;
import org.matsim.core.router.TripStructureUtils;
import org.matsim.core.router.TripStructureUtils.Trip;
import org.matsim.core.scoring.EventsToActivities;
import org.matsim.core.scoring.EventsToLegs;
import org.matsim.core.scoring.PersonExperiencedActivity;
import org.matsim.core.scoring.PersonExperiencedLeg;
import org.matsim.core.scoring.ScoringFunction;
import org.matsim.core.scoring.ScoringFunctionFactory;

import com.google.inject.Inject;
import com.google.inject.Singleton;

/**
 * Instantiates and manages native MATSim ScoringFunctions per individual agent, 
 * leveraging individual experience (real arrival times, real travel times) 
 * via official event streams.
 */
@Singleton
public class MatsimScoreTracker implements ActivityStartEventHandler, PersonMoneyEventHandler,
        PersonScoreEventHandler, PersonStuckEventHandler {

    private static final Logger log = LogManager.getLogger(MatsimScoreTracker.class);

    private final ScoringFunctionFactory scoringFunctionFactory;
    private final Population population;

    private final Map<Id<Person>, ScoringFunction> scoringFunctions = new HashMap<>();
    private final Map<Id<Person>, Double> lastReadScore = new HashMap<>();
    private final Map<Id<Person>, Plan> tripRecords = new HashMap<>();

    private final Set<Id<Person>> finishedAgents = new HashSet<>();

    /**
     * Initializes the score tracker with required MATSim factories, population references, 
     * and event feed singletons.
     *
     * @param scoringFunctionFactory Native Matsim Scoring Factory to build scoring functions per agent.
     * @param population             The complete simulation population container.
     * @param eventsToActivities     Singleton handling experienced activity callbacks.
     * @param eventsToLegs           Singleton handling experienced leg callbacks.
     */
    @Inject
    public MatsimScoreTracker(ScoringFunctionFactory scoringFunctionFactory, Population population,
                              EventsToActivities eventsToActivities, EventsToLegs eventsToLegs) {
        this.scoringFunctionFactory = scoringFunctionFactory;
        this.population = population;

        eventsToActivities.addActivityHandler(this::handleExperiencedActivity);
        eventsToLegs.addLegHandler(this::handleExperiencedLeg);
    }

    /**
     * Retrieve the active scoring function for an agent if their day is not yet finished.
     *
     * @param agentId The ID of the agent.
     * @return The active ScoringFunction instance, or null if finished or untracked.
     */
    private ScoringFunction getActiveScoringFunction(Id<Person> agentId) {return this.finishedAgents.contains(agentId) ? null : this.scoringFunctions.get(agentId);}

    /**
     * Retrieve the current accumulated MATSim score for a tracked agent without consuming 
     * scores or closing their day.
     *
     * @param agentId The ID of the agent.
     * @return The current accumulated score, or Double.NaN if untracked.
     */
    public double getDayScore(Id<Person> agentId) {
        ScoringFunction scoringFunction = this.scoringFunctions.get(agentId);
        return scoringFunction != null ? scoringFunction.getScore() : Double.NaN;
    }

    /**
     * Builds a fresh scoring function for every agent to be tracked in the upcoming iteration.
     * Called once per iteration, before the mobsim starts.
     *
     * @param trackedAgents Collection of agent IDs {@code Id<Person>} being actively monitored in the current iteration.
     */
    public void beginIteration(Collection<Id<Person>> trackedAgents) {
        this.scoringFunctions.clear();
        this.lastReadScore.clear();
        this.tripRecords.clear();
        this.finishedAgents.clear();

        for (Id<Person> agentId : trackedAgents) {
            Person person = this.population.getPersons().get(agentId);
            if (person == null) {
                log.warn("Cannot track score of '{}': not in population.", agentId);
                continue;
            }
            this.scoringFunctions.put(agentId, this.scoringFunctionFactory.createNewScoringFunction(person));
            this.lastReadScore.put(agentId, 0.0);
            this.tripRecords.put(agentId, PopulationUtils.createPlan());
        }

        log.info("Tracking MATSim scores for {} agents.", this.scoringFunctions.size());
    }

    /**
     * Calculates and returns the score accumulated by an agent since the previous check, 
     * representing the utility of recently completed plan elements, while consuming the read pointer.
     *
     * @param agentId The ID of the agent whose step score is being consumed.
     * @return The incremental score change since the last read.
     */
    public double consumeStepScore(Id<Person> agentId) {
        ScoringFunction scoringFunction = this.scoringFunctions.get(agentId);
        if (scoringFunction == null) {
            return 0.0;
        }

        double currentScore = scoringFunction.getScore();
        double previousScore = this.lastReadScore.getOrDefault(agentId, 0.0);
        this.lastReadScore.put(agentId, currentScore);

        System.out.println("The matsim score upto this leg from the inbuilt scorer is: " + currentScore);
        System.out.println("The score for the previous leg was: " + previousScore);
        
        return currentScore - previousScore;
    }

    /**
     * Closes the agent's day early by synthesizing an open-ended overnight activity at 
     * their current arrival location, finishing their scoring function, and returning their total final score.
     *
     * @param agentId      The ID of the agent finishing their day.
     * @param lastActivity The final activity template reached by the agent.
     * @param arrivalTime  The simulation timestamp of the arrival.
     * @return The final day score computed for the agent.
     */
    public double finishDay(Id<Person> agentId, Activity lastActivity, double arrivalTime) {
        ScoringFunction scoringFunction = this.scoringFunctions.get(agentId);
        if (scoringFunction == null) {
            return 0.0;
        }

        if (this.finishedAgents.add(agentId)) {
            Activity overnightActivity = PopulationUtils.createActivityFromLinkId(
                    lastActivity.getType(), lastActivity.getLinkId());
            overnightActivity.setFacilityId(lastActivity.getFacilityId());
            if (lastActivity.getCoord() != null) {
                overnightActivity.setCoord(lastActivity.getCoord());
            }
            overnightActivity.setStartTime(arrivalTime);
            // End time deliberately left undefined. -> WHY ???

            scoringFunction.handleActivity(overnightActivity);
            scoringFunction.finish();
        }

        double dayScore = scoringFunction.getScore();
        this.lastReadScore.put(agentId, dayScore);

        return dayScore;
    }

    /**
     * Checks if a specific agent is currently being tracked by the score tracker.
     *
     * @param agentId The ID of the agent to check.
     * @return True if the agent is tracked, false otherwise.
     */
    public boolean isTracked(Id<Person> agentId) {
        return this.scoringFunctions.containsKey(agentId);
    }

    /**
     * Intercepts experienced activity events and feeds them to the agent's scoring function and scratch plan.
     *
     * @param experienced The experienced activity event wrapper.
     */
    private void handleExperiencedActivity(PersonExperiencedActivity experienced) {
        ScoringFunction scoringFunction = getActiveScoringFunction(experienced.getAgentId());
        if (scoringFunction == null) {
            return;
        }

        scoringFunction.handleActivity(experienced.getActivity());
        this.tripRecords.get(experienced.getAgentId()).addActivity(experienced.getActivity());
    }

    /**
     * Intercepts experienced leg events and feeds them to the agent's scoring function and scratch plan.
     *
     * @param experienced The experienced leg event wrapper.
     */
    private void handleExperiencedLeg(PersonExperiencedLeg experienced) {
        ScoringFunction scoringFunction = getActiveScoringFunction(experienced.getAgentId());
        if (scoringFunction == null) {
            return;
        }

        scoringFunction.handleLeg(experienced.getLeg());
        this.tripRecords.get(experienced.getAgentId()).addLeg(experienced.getLeg());
    }

    /**
     * Intercepts activity start events to filter stage activities, reconstruct completed trips, 
     * and pass them to the native scoring function via handleTrip().
     *
     * @param event The activity start event.
     */
    @Override
    public void handleEvent(ActivityStartEvent event) {
        if (StageActivityTypeIdentifier.isStageActivity(event.getActType())) {
            return;
        }

        ScoringFunction scoringFunction = getActiveScoringFunction(event.getPersonId());
        if (scoringFunction == null) {
            return;
        }

        Plan tripRecord = this.tripRecords.get(event.getPersonId());

        Activity destinationActivity = PopulationUtils.createActivityFromLinkId(event.getActType(), event.getLinkId());
        destinationActivity.setStartTime(event.getTime());
        tripRecord.addActivity(destinationActivity);

        List<Trip> trips = TripStructureUtils.getTrips(tripRecord);
        for (Trip trip : trips) {
            if (trip != null) {
                scoringFunction.handleTrip(trip);
            }
        }

        tripRecord.getPlanElements().clear();
    }

    /**
     * Forwards monetary transaction events directly into the agent's active scoring function.
     *
     * @param event The person money event.
     */
    @Override
    public void handleEvent(PersonMoneyEvent event) {
        ScoringFunction scoringFunction = getActiveScoringFunction(event.getPersonId());
        if (scoringFunction != null) {
            scoringFunction.addMoney(event.getAmount());
        }
    }

    /**
     * Forwards custom score adjustment events directly into the agent's active scoring function.
     *
     * @param event The person score event.
     */
    @Override
    public void handleEvent(PersonScoreEvent event) {
        ScoringFunction scoringFunction = getActiveScoringFunction(event.getPersonId());
        if (scoringFunction != null) {
            scoringFunction.addScore(event.getAmount());
        }
    }

    /**
     * Forwards agent stuck events directly into the active scoring function to assess penalties.
     *
     * @param event The person stuck event.
     */
    @Override
    public void handleEvent(PersonStuckEvent event) {
        ScoringFunction scoringFunction = getActiveScoringFunction(event.getPersonId());
        if (scoringFunction != null) {
            scoringFunction.agentStuck(event.getTime());
        }
    }

    @Override
    public void reset(int iteration) {}
}
