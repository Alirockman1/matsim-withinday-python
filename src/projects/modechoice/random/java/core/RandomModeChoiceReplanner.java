package modechoice.random.java.core;

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
import org.matsim.withinday.environment.StateEngine;
import org.matsim.withinday.environment.WithinDayObserver;
import org.matsim.withinday.networking.CommunicationManager;
import org.matsim.withinday.utils.WithinDayAgentExperience;
import org.matsim.core.utils.timing.TimeInterpretation;

import com.google.inject.Inject;

public class RandomModeChoiceReplanner extends WithinDayReplanner {

    private final Random randomGenerator = new Random(42);

    @Inject
    public RandomModeChoiceReplanner(Scenario scenario, TripRouter router, TimeInterpretation timeInterpretation,
                                      WithinDayObserver customObserver, CommunicationManager communicationManager) {
        super(scenario, router, timeInterpretation, customObserver, communicationManager);
    }

    @Override
    public void initializeExternalModel() {
        // Initialize the random model
        Map<String, Object> jsonMap = new HashMap<>();
        long currentSeed = randomGenerator.nextLong();
        jsonMap.put("randomSeed", currentSeed);
        communicationManager.httpPost(gson.toJson(jsonMap), "/session/configure", 30);
    }

    @Override 
    protected Map<String, Object> determineAction(MobsimAgent agent, Trip nextTrip, Map<String, Object> agentDemographics, Map<String, Object> stateObservation) {
        Plan modifiablePlan = WithinDayAgentUtils.getModifiablePlan(agent);

        // Construct the state observation metrics
        if (agentDemographics != null) {
            stateObservation.put("subpopulation", agentDemographics.getOrDefault("subpopulation", "default"));
        }

        // Request decision from serving model
        //String mode = TripStructureUtils.identifyMainMode(nextTrip.getTripElements()).trim();
        log.info("COMMUNICATION NET: Transmitting environment state for agent {}", agentDemographics.get("agentId").toString());
        String jsonState = gson.toJson(stateObservation);
        String response = this.communicationManager.httpPost(jsonState, communicationManager.getDecisionEndpoint(), 360, agentDemographics.get("agentId").toString()).trim();
        String mode = response.replaceAll("[{}\"]", "").split(":")[1].trim().toLowerCase();
        log.info("Assigned mode '{}' to agent {}", mode.toUpperCase(), agentDemographics.get("agentId").toString());

        // Update agent's executed plan in memory
        List<? extends PlanElement> newNextTrip = editTrips.replanFutureTrip(nextTrip, modifiablePlan, 
            mode, agent.getActivityEndTime());

        Map<String, Object> result = new HashMap<>();
        result.put("mode", mode);
        result.put("newNextTrip", newNextTrip);
        return result;
    }

    @Override
    public Map<Id<Person>, WithinDayAgentExperience> getAgentExperiences() {
        return null;
    }
}
