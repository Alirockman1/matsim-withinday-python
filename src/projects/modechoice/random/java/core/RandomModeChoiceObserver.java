package modechoice.random.java.core;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.matsim.api.core.v01.Id;
import org.matsim.api.core.v01.Scenario;
import org.matsim.api.core.v01.population.Person;
import org.matsim.core.mobsim.framework.MobsimAgent;
import org.matsim.core.mobsim.qsim.QSim;
import org.matsim.core.router.TripStructureUtils.Trip;
import org.matsim.withinday.environment.AgentAssetInventory;
import org.matsim.withinday.environment.StateEngine;
import org.matsim.withinday.environment.WithinDayObserver;
import org.matsim.withinday.environment.WithinDayRealTimeScoringEngine;

import com.google.inject.Inject;
import com.google.inject.Singleton;

@Singleton
public class RandomModeChoiceObserver extends WithinDayObserver {

    private static final Logger log = LogManager.getLogger(RandomModeChoiceObserver.class);

    @Inject
    public RandomModeChoiceObserver(Scenario scenario) {
        super(scenario, log);
    }

    @Override
    public Map<String, Object> observeState(MobsimAgent agent, QSim sim, Trip trip, double currentTime,
            boolean isNextTripContext) {
        Map<String, Object> observation = new HashMap<>();
        double departureTimeSeconds = currentTime;

        if (isNextTripContext) {
            departureTimeSeconds = StateEngine.getPredictedDepartureTime(trip.getDestinationActivity(), currentTime).seconds();
        }

        Map<String, Object> endOfDayResult = endOfDay(agent, trip, isNextTripContext);
        boolean isEndOfDay = (boolean) endOfDayResult.get("flag");

        // Mode set for the agent
        List<String> availableModes = new ArrayList<>(AgentAssetInventory.getSimulationModes());

        // Top-Level Framework Metadata
        observation.put("simulationIteration", StateEngine.currentIteration);
        observation.put("agentID", agent.getId().toString());
        observation.put("possibleModeSet", availableModes);
        observation.put("endOfDayFlag", isEndOfDay);

        return observation;
    }

    @Override
    public WithinDayRealTimeScoringEngine getOrCreateScoringEngine(Id<Person> agentId) {
        return this.agentRewardCalculators.computeIfAbsent(agentId, id -> new WithinDayRealTimeScoringEngine(this.scenario, this,
            this.scoringParametersForPerson.getScoringParameters(this.scenario.getPopulation().getPersons().get(id))));
    }
}
