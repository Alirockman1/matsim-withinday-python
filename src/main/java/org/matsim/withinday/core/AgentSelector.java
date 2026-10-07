package org.matsim.withinday.core;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import org.matsim.api.core.v01.Scenario;
import org.matsim.api.core.v01.population.Person;
import org.matsim.api.core.v01.population.Plan;
import org.matsim.core.mobsim.framework.MobsimAgent;
import org.matsim.core.mobsim.qsim.agents.WithinDayAgentUtils;
import org.matsim.withinday.utils.WithinDayConfigGroup;

import com.google.inject.Inject;

import org.matsim.api.core.v01.Id;

/**
 * <h2>AgentSelector Component Description</h2>
 * <p>
 * The {@code AgentSelector} is a core orchestrator class responsible for managing 
 * agent participation, initialization, and tracking in multi-agent scenario simulations. 
 * It bridges MATSim's large-scale population with external bridge protocols by performing 
 * a structured, two-tier sampling and tracking strategy:
 * </p>
 * 
 * <h3>Key Architectural Responsibilities:</h3>
 * <ol>
 *   <li><strong>Tier-1 Simulation Subspace Filtering:</strong> Establishes a master pool of eligible 
 *       special agents at simulation startup. It prioritizes explicit user-defined lists 
 *       ({@code trainAgentIDList}) or falls back to a deterministic percentage sample of the total 
 *       population (defaulting to 10%).</li>
 *   <li><strong>Tier-2 Iteration-Level Subsampling &amp; Capping:</strong> Subsamples agents from the master 
 *       subspace at the start of each simulation iteration, applying runtime population caps 
 *       ({@code agentsPerIteration}) to maintain scalable control over active scenario agents.</li>
 *   <li><strong>Unique ID/Tag Mapping:</strong> Allocates and manages unique numeric sequence tags (e.g., 4-digit 
 *       runtime tracking slots starting from 1001) for active iteration agents. This provides lightweight 
 *       identifiers that synchronize Java state requests with external communication endpoints.</li>
 *   <li><strong>Real-Time Replanning Gatekeeping:</strong> Implements safety checks ({@code shouldReplan}) 
 *       during the mobsim execution to verify that selected agents possess valid active plans and remaining 
 *       trip elements before triggering within-day updates or interactions.</li>
 * </ol>
 * 
 * @author Multi-Agent Scenario Simulation Architecture
 */
public class AgentSelector { 

    private static final Logger log = LogManager.getLogger(AgentSelector.class);
        
    private final Scenario scenario;
    private final Long randomSeed;
    private final int numAgentsPerIteration;
    private final WithinDayConfigGroup configGroup;

    // Master pool of agents eligible for scenario tracking throughout the entire simulation run (Tier 1)
    private final Set<Id<Person>> simulationAgentSubspace = new HashSet<>();
    
    // Active subset of agents participating in the current iteration (Tier 2)
    private final Set<Id<Person>> selectedAgents = new HashSet<>();
    
    // Set of pre-allocated unique numeric tags mapped to active agents for communication integrity
    private final Set<Integer> uniqueAgentTags = new HashSet<>();
    
    // Direct mapping linking each active agent ID to its unique iteration-specific numeric tag
    private final Map<Id<Person>, Integer> agentIterationTags = new HashMap<>();

    /**
     * Constructs a new AgentSelector instance via dependency injection.
     * 
     * @param scenario The MATSim scenario containing the population and network.
     * @param seed     The random seed used for reproducible shuffling.
     * @param log      The Log4j logger for outputting status and warnings.
     */
    @Inject
    public AgentSelector(Scenario scenario, WithinDayConfigGroup configGroup) {
        this.scenario = scenario;
        this.randomSeed = 42L;
        this.configGroup = configGroup;
        this.numAgentsPerIteration = this.configGroup.getAgentsPerIteration();

        createUniqueTags();
    }

    /**
     * Returns the selected agent set participating in the current simulation iteration.
     * 
     * @return An unmodifiable {@link Set} of agent IDs ({@link Id<Person>}) 
     *         currently active in this iteration's scenario run.
     */
    public Set<Id<Person>> getSelectedAgents() {return Collections.unmodifiableSet(selectedAgents);}

    /**
     * Retrieves the unique numeric tag assigned to an agent ID for the current iteration.
     * 
     * @param agentId The unique identifier of the person/agent whose tag is being requested.
     * @return The 4-digit integer tag mapped to the agent, or -1 if the agent is not selected
     * or missing in the current iteration.
     */
    public int getAgentTag(Id<Person> agentId) {
        if (agentIterationTags.containsKey(agentId)) {
            return agentIterationTags.get(agentId);
        }
        log.warn("AgentSelector: Requested unique tag for unselected or missing agent ID in current iteration: {}. Returning -1.", agentId);
        return -1;
    }

    /**
     * Retrieves the unique numeric tag assigned to an agent using their string-based ID for the current iteration.
     * 
     * @param agentIdStr The string representation of the person/agent identifier.
     * @return The 4-digit integer tag mapped to the agent, or -1 if the agent is not found or unselected.
     */
    public int getAgentTag(String agentIdStr) {return getAgentTag(Id.createPersonId(agentIdStr));}

    /**
     * Returns an unmodifiable set containing all pre-allocated unique tags available for assignment.
     * 
     * @return An unmodifiable Set of Integer representing the available unique agent tags.
     */
    public Set<Integer> getUniqueAgentTags() {return Collections.unmodifiableSet(uniqueAgentTags);}

    /**
     * Sets up the fixed master pool of agents.
     * The agents are selected based on explicit lists, or specified percentage sampling (defaulting to 10%).
     */
    public void sampleAgentsForSimulation() {
        simulationAgentSubspace.clear();

        if (this.configGroup == null) {
            log.error("AgentSelector: WithinDayConfigGroup is null! Cannot initialize simulation subspace.");
            return;
        }

        double samplingPercentage = this.configGroup.getSamplingPercentage();
        Collection<String> explicitTrainAgentIds = new ArrayList<>();

        String filterListString = this.configGroup.getTrainAgentIDList();
        if (this.configGroup!= null && filterListString != null) {
            explicitTrainAgentIds = Arrays.asList(filterListString.split("\\s*,\\s*"));
        }

        List<Person> candidatePool = new ArrayList<>();

        if (explicitTrainAgentIds != null && !explicitTrainAgentIds.isEmpty()) {
            for (String agentIdStr : explicitTrainAgentIds) {
                if (agentIdStr == null || agentIdStr.isBlank()) continue;
                Id<Person> id = Id.createPersonId(agentIdStr.trim());
                Person person = scenario.getPopulation().getPersons().get(id);
                if (person != null && isEligiblePerson(person)) {
                    candidatePool.add(person);
                } else {
                    log.warn("AgentSelector: Explicit train agent ID '{}' not found in population.", agentIdStr);
                }
            }
            log.info("AgentSelector: Initialized simulation subspace with {} explicit training agents.", candidatePool.size());
        } else {
            double effectivePercentage = (samplingPercentage > 0.0) ? samplingPercentage : 0.10;
            log.info("AgentSelector: No explicit training list found. Using baseline simulation sampling percentage: {}%", effectivePercentage * 100.0);

            List<Person> allPersons = new ArrayList<>(scenario.getPopulation().getPersons().values());
            Collections.shuffle(allPersons, new Random(this.randomSeed));
            int numberToSample = (int) Math.max(1, Math.round(allPersons.size() * effectivePercentage));

            for (int i = 0; i < numberToSample && i < allPersons.size(); i++) {
                candidatePool.add(allPersons.get(i));
            }
            log.info("AgentSelector: Initialized simulation subspace with {} agents from percentage sampling.", candidatePool.size());
        }

        for (Person p : candidatePool) {
            simulationAgentSubspace.add(p.getId());
        }
    }

    /**
     * Creates and initializes the unique 4-digit runtime tag slots and saves them 
     * into the uniqueAgentTags set.
     */
    public void createUniqueTags() {
        uniqueAgentTags.clear();

        if (this.numAgentsPerIteration <= 0) {
            log.warn("AgentSelector: agentsPerIteration is set to 0 or negative. No unique tags created.");
            return;
        }

        int tagCounter = 1001;
        for (int i = 0; i < this.numAgentsPerIteration; i++) {
            int currentTag = tagCounter++;
            uniqueAgentTags.add(currentTag);
        }

        log.info("AgentSelector: Successfully initialized {} unique tag slots in uniqueAgentTags (1001 to {}).", 
                uniqueAgentTags.size(), tagCounter - 1);
    }

    /**
     * Samples agents from subspace for the current iteration.
     *
     * @param iteration Current MATSim iteration.
     * @param samplingPercentage Value in [0,1]. Ignored if fixedAgentIds is non-empty.
     * @param fixedAgentIds Explicit agent IDs (overrides percentage sampling).
     */
    public void sampleAgentsForIteration(int iteration) {
        selectedAgents.clear();

        List<Id<Person>> poolList = new ArrayList<>(simulationAgentSubspace);
        long dynamicSeed = System.currentTimeMillis() + (iteration * 31L) + this.randomSeed;
        Random iterationRandom = new Random(dynamicSeed);
        Collections.shuffle(poolList, iterationRandom);

        int limit = poolList.size();
        if (this.numAgentsPerIteration > 0 && this.numAgentsPerIteration < limit) {
            limit = this.numAgentsPerIteration;
        }

        List<Id<Person>> sampledSubset = new ArrayList<>();
        for (int i = 0; i < limit && i < poolList.size(); i++) {
            sampledSubset.add(poolList.get(i));
        }

        // Delegate assignment to private helper method
        assignTagsToAgentIds(sampledSubset);

        log.info("AgentSelector: Iteration {} sampled and assigned {} active agents to unique tags.", 
                iteration, selectedAgents.size());
    }

    /**
     * Maps each selected agents to an unique tag.
     * @param sampledSubset Sampled agent Ids from the sub space donor pool.
     */
    private void assignTagsToAgentIds(List<Id<Person>> sampledSubset) {
        List<Integer> tagList = new ArrayList<>(uniqueAgentTags);
        Collections.sort(tagList);

        int i = 0;
        for (Integer tag : tagList) {
            if (i >= sampledSubset.size()) {
                break;
            }
            Id<Person> agentId = sampledSubset.get(i++);
            selectedAgents.add(agentId);
            agentIterationTags.put(agentId, tag);
        }
    }

    /**
     * Determines whether a MATSim agent should enter the replanning process 
     * at the current simulation step.
     * 
     * An agent is eligible for RL replanning only if:
     *     The agent was sampled for the current iteration.
     *     The agent has a valid current plan element.
     *     The agent has a future trip available that can be modified.
     * 
     * @param agent The MATSim simulation agent being evaluated.
     * @return {@code true} if the agent is selected for replanning in the current iteration; {@code false} otherwise.
     **/
    public boolean shouldReplan(MobsimAgent agent) {

        if (!selectedAgents.contains(agent.getId())) {
            return false;
        }

        Integer currentPlanElementIndex = WithinDayAgentUtils.getCurrentPlanElementIndex(agent);
        Plan plan = WithinDayAgentUtils.getModifiablePlan(agent);
        Integer maxPlanElementIndex =plan.getPlanElements().size();
        
        if (currentPlanElementIndex == null || currentPlanElementIndex >=  maxPlanElementIndex - 1) {
            return false;
        }

        return true;
    }

    /** 
     * Checks whether a specific agent ID is included in the active 
     * tracking list for the current simulation iteration.
     * 
     * @param id The agent ID ({@link Id<Person>}) to check.
     * @return {@code true} if the agent is active in the current iteration; {@code false} otherwise.
     */
    public boolean contains(Id<Person> id) {return selectedAgents.contains(id);}

    /** 
     * Resets and clears the active agent selection lists and their 
     * corresponding runtime tracking tags for the current iteration.
     */
    public void reset() {
        selectedAgents.clear();
        agentIterationTags.clear();
    }

    /**
     * Determines whether a given person from the population is eligible 
     * for inclusion during the initial scenario sampling phase.
     * 
     * @param person The MATSim person object being evaluated.
     * @return {@code true} if the person meets eligibility criteria; {@code false} to exclude them.
     */
    private boolean isEligiblePerson(Person person) {
        return true;
    }    
}
