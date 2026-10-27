package modechoice.discrete_choice.java.core;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.Scenario;
import org.matsim.api.core.v01.population.Person;
import org.matsim.api.core.v01.population.Plan;
import org.matsim.api.core.v01.population.PlanElement;
import org.matsim.core.mobsim.framework.MobsimAgent;
import org.matsim.core.mobsim.qsim.agents.WithinDayAgentUtils;
import org.matsim.core.router.TripRouter;
import org.matsim.core.router.TripStructureUtils;
import org.matsim.core.router.TripStructureUtils.Trip;
import org.matsim.withinday.core.WithinDayReplanner;
import org.matsim.withinday.environment.WithinDayObserver;
import org.matsim.withinday.networking.CommunicationManager;
import org.matsim.withinday.utils.WithinDayAgentExperience;
import org.matsim.core.utils.timing.TimeInterpretation;

import com.google.inject.Inject;

public class DiscreteModeChoiceReplanner extends WithinDayReplanner {

    @Inject
    public DiscreteModeChoiceReplanner(Scenario scenario, TripRouter router, TimeInterpretation timeInterpretation,
                                      WithinDayObserver customObserver, CommunicationManager communicationManager) {
        super(scenario, router, timeInterpretation, customObserver);
    }

    @Override
    public void initializeExternalModel() {}

    @Override 
    protected Map<String, Object> determineAction(MobsimAgent agent, Trip nextTrip, Map<String, Object> agentDemographics, Map<String, Object> stateObservation) {
        Plan modifiablePlan = WithinDayAgentUtils.getModifiablePlan(agent);
        Person person = modifiablePlan.getPerson();

        // Use modifiablePlan
        List<TripStructureUtils.Trip> originalTrips = TripStructureUtils.getTrips(modifiablePlan);

        int currentTripIndex = -1;
        for (int i = 0; i < originalTrips.size(); i++) {
            if (originalTrips.get(i).equals(nextTrip)) {
                currentTripIndex = i;
                break;
            }
        }

        // Pull the trip structure from the active modifiable plan as well
        List<TripStructureUtils.Trip> currentTripStructure = TripStructureUtils.getTrips(modifiablePlan);

        // Extract the mode for this specific trip index from the modified plan
        String currentTripMode = "";
        if (currentTripIndex >= 0 && currentTripIndex < currentTripStructure.size()) {
            TripStructureUtils.Trip correspondingModifiedTrip = currentTripStructure.get(currentTripIndex);
            currentTripMode = TripStructureUtils.identifyMainMode(correspondingModifiedTrip.getTripElements());
        } else {
            currentTripMode = TripStructureUtils.identifyMainMode(nextTrip.getTripElements());
        }

        // Update agent's executed plan in memory
        List<? extends PlanElement> newNextTrip = editTrips.replanFutureTrip(nextTrip, modifiablePlan, 
            currentTripMode, agent.getActivityEndTime());

        Map<String, Object> result = new HashMap<>();
        result.put("mode", currentTripMode);
        result.put("newNextTrip", newNextTrip);
        return result;
    }

    @Override
    public Map<Id<Person>, WithinDayAgentExperience> getAgentExperiences() {
        return null;
    }
}
