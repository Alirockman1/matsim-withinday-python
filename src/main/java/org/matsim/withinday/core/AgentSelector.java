package org.matsim.withinday.core;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
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

    private static final double DEFAULT_TRAIN_PERCENTAGE = 0.10;
    private static final double DEFAULT_TEST_TO_TRAIN_RATIO = 0.25;
        
    private final Scenario scenario;
    private final Long randomSeed;
    private final int numAgentsPerIteration;
    private final WithinDayConfigGroup configGroup;

    // Master pool of agents eligible for scenario tracking throughout the entire simulation run
    private final Set<Id<Person>> simulationAgentSubspace = new HashSet<>();
    private final Set<Id<Person>> simulationAgentTestSubspace = new HashSet<>();
    
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
        if (this.configGroup != null && filterListString != null && !filterListString.isBlank()) {
            explicitTrainAgentIds = Arrays.asList(filterListString.split("\\s*,\\s*"));
        } else {
            explicitTrainAgentIds = Collections.emptyList();
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
            double trainPercentage = (samplingPercentage > 0.0) ? samplingPercentage : DEFAULT_TRAIN_PERCENTAGE;
            candidatePool = sampleTrainTestSplit(trainPercentage);
        }

        for (Person p : candidatePool) {
            simulationAgentSubspace.add(p.getId());
        }
    }

    /**
     * Splits the population into a disjoint test and train set and returns the one
     * requested by {@code configGroup.getDataSetType()} ("train" or "test").
     * <p>
     * Procedure:
     * <ol>
     *   <li>Sort all eligible persons by ID, so the result does not depend on the
     *       iteration order of the population map.</li>
     *   <li><strong>Test set first:</strong> shuffle with a fixed seed and take
     *       {@link #DEFAULT_TEST_PERCENTAGE} of the population.</li>
     *   <li><strong>Remove the test persons</strong> from the pool.</li>
     *   <li><strong>Train set:</strong> shuffle the remaining pool with a second fixed seed
     *       and take {@code trainPercentage} of the <em>whole</em> population (capped at the
     *       size of the remaining pool).</li>
     * </ol>
     * Both seeds are fixed, so the same population always yields the same sets. The test
     * set is also identical whether the run is a train run or a test run, because it never
     * depends on the train percentage.
     *
     * @param trainPercentage fraction of the whole population to use for training, in (0, 1].
     * @return the persons of the requested data set.
     */
    private List<Person> sampleTrainTestSplit(double trainPercentage) {
        // Deterministic base order
        List<Person> allPersons = new ArrayList<>();
        for (Person person : scenario.getPopulation().getPersons().values()) {
            if (isEligiblePerson(person)) {
                allPersons.add(person);
            }
        }
        allPersons.sort(Comparator.comparing(p -> p.getId().toString()));
        int total = allPersons.size();
 
        if (total < 2) {
            log.warn("AgentSelector: Population has {} eligible person(s); cannot create a disjoint train/test split.", total);
        }
 
        // Draw the test set first
        long testSeed = this.randomSeed;
        long trainSeed = this.randomSeed + 1L;
        
        // Sample the train dataset
        int numTrain = (int) Math.max(1, Math.round(total * trainPercentage));
        numTrain = Math.min(numTrain, total - 1);

        List<Person> shuffledForTrain = new ArrayList<>(allPersons);
        Collections.shuffle(shuffledForTrain, new Random(trainSeed));
        List<Person> trainSet = new ArrayList<>(shuffledForTrain.subList(0, numTrain));

        // Sample the test dataset from the remaining space
        Set<Id<Person>> trainIds = new HashSet<>();
        for (Person p : trainSet) {
            trainIds.add(p.getId());
        }

        List<Person> remainingPool = new ArrayList<>();
        for (Person p : allPersons) {
            if (!trainIds.contains(p.getId())) {
                remainingPool.add(p);
            }
        }

        Collections.shuffle(remainingPool, new Random(testSeed));

        // Test set size is 20% of the train set size (i.e., a 20:80 ratio)
        int numTest = (int) Math.max(1, Math.round(trainSet.size() * DEFAULT_TEST_TO_TRAIN_RATIO));
        numTest = Math.min(numTest, remainingPool.size());

        List<Person> testSet = new ArrayList<>(remainingPool.subList(0, numTest));

        for (Person p : testSet) {
            simulationAgentTestSubspace.add(p.getId());
        }

        log.info("AgentSelector: Split population of {} into {} test and {} train agents (disjoint; test seed {}, train seed {}).",
        total, testSet.size(), trainSet.size(), testSeed, trainSeed);
 
        // Return the requested data set
        String dataSetType = this.configGroup.getDataSetType();
        boolean useTest = dataSetType != null && dataSetType.trim().equalsIgnoreCase("test");
 
        if (useTest) {
            log.info("AgentSelector: dataSetType = 'test' -> simulation subspace is the test set ({} agents).", testSet.size());
            return testSet;
        }
 
        if (dataSetType == null || !dataSetType.trim().equalsIgnoreCase("train")) {
            log.warn("AgentSelector: Unknown dataSetType '{}'; defaulting to 'train'.", dataSetType);
        }

        log.info("AgentSelector: dataSetType = 'train' -> simulation subspace is the train set ({} agents).", trainSet.size());
        return trainSet;
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
     * Returns the number of persons sampled from the entire population, i.e. the
     * train or test set (or the explicit agent list) chosen in {@link #sampleAgentsForSimulation()}.
     * 
     * @return The number of agents in the simulation subspace; 0 if it has not been sampled yet.
     */
    public int getSimulationSubspaceSize() {return simulationAgentSubspace.size();}

    /**
     * Returns the person id of the agents sampled as train cases
     * 
     * @return {@set<Id<Person>>}} of agents {@Id} in the simulation train subspace
     */
    public Set<Id<Person>> getAgentsFromSimulationSubspace() {return simulationAgentSubspace;}

    /**
     * Returns the number of persons sampled as test cases, i.e. the
     * train or test set (or the explicit agent list) chosen in {@link #sampleAgentsForSimulation()}.
     * 
     * @return The number of agents in the simulation subspace; 0 if it has not been sampled yet.
     */
    public int getSimulationTestSize() {return simulationAgentTestSubspace.size();}

    /**
     * Returns the person id of the agents sampled as test cases
     * 
     * @return {@set<Id<Person>>}} of agents {@Id} in the simulation train subspace
     */
    public Set<Id<Person>> getAgentsFromSimulationTestSubspace() {return simulationAgentTestSubspace;}

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
