package modechoice.rl.java.utils;

import org.matsim.api.core.v01.population.Person;
import org.matsim.core.config.ConfigGroup;
import org.matsim.core.config.ReflectiveConfigGroup;
import org.matsim.api.core.v01.Id;

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;


public class CustomConfigGroup extends ReflectiveConfigGroup {
    public static final String GROUP_NAME = "agentModeChoice";
    private static final String WEIGHTS_SET_TYPE = "modelWeights";

    private String modelType;
    private String autoEncoderModel = "";
    private int saveInterval = 1;
    private int validationInterval = 1;
    private double discontinuityPenalty = 1.0;
    private double retrievalCostPenalty = 1.5;

    private double alpha;
    private double gamma;
    private double epsilon;
    private double epsilonDecay;
    private double epsilonMinimum;
    private int trainingCutoffIteration;

    public CustomConfigGroup() {
        super(GROUP_NAME);
    }

    //--- Custom methods ---//
    
    public Map<String, Double> getModelWeights() { 
        Map<String, Double> weights = new HashMap<>();
        
        weights.put("discontinuityPenalty", this.discontinuityPenalty);
        weights.put("retrievalCostPenalty", this.retrievalCostPenalty);
        
        return weights;
    }

    //--- Getters & Setters ---//

    @StringGetter("modelType")
    public String getModelType() { return modelType; }

    @StringSetter("modelType")
    public void setModelType(String modelType) { this.modelType = modelType; }

    @StringGetter("alpha")
    public double getAlpha() { return alpha; }

    @StringSetter("alpha")
    public void setAlpha(double alpha) { this.alpha = alpha; }

    @StringGetter("gamma")
    public double getGamma() { return gamma; }

    @StringSetter("gamma")
    public void setGamma(double gamma) { this.gamma = gamma; }

    @StringGetter("epsilon")
    public double getEpsilon() { return epsilon; }

    @StringSetter("epsilon")
    public void setEpsilon(double epsilon) { this.epsilon = epsilon; }

    @StringGetter("epsilonDecay")
    public double getEpsilonDecay() { return epsilonDecay; }

    @StringSetter("epsilonDecay")
    public void setEpsilonDecay(double epsilonDecay) { this.epsilonDecay = epsilonDecay; }

    @StringGetter("epsilonMinimum")
    public double getEpsilonMinimum() { return epsilonMinimum; }

    @StringSetter("epsilonMinimum")
    public void setEpsilonMinimum(double epsilonMinimum) { this.epsilonMinimum = epsilonMinimum; }

    @StringGetter("trainingCutoffIteration")
    public int getTrainingCutoffIteration() { return trainingCutoffIteration; }

    @StringSetter("trainingCutoffIteration")
    public void setTrainingCutoffIteration(int trainingCutoffIteration) { this.trainingCutoffIteration = trainingCutoffIteration; }

    @StringGetter("saveInterval")
    public int getSaveInterval() { return saveInterval; }

    @StringSetter("saveInterval")
    public void setSaveInterval(int saveInterval) { this.saveInterval = saveInterval; }

    @StringGetter("validationInterval")
    public int getValidationInterval() { return validationInterval; }

    @StringSetter("validationInterval")
    public void setValidationInterval(int validationInterval) { this.validationInterval = validationInterval; }

    @StringGetter("discontinuityPenalty")
    public double getDiscontinuityPenalty() { return discontinuityPenalty; }

    @StringSetter("discontinuityPenalty")
    public void setDiscontinuityPenalty(double discontinuityPenalty) { this.discontinuityPenalty = discontinuityPenalty; }

    @StringGetter("retrievalCostPenalty")
    public double getRetrievalCostPenalty() { return retrievalCostPenalty; }

    @StringSetter("retrievalCostPenalty")
    public void setRetrievalCostPenalty(double retrievalCostPenalty) { this.retrievalCostPenalty = retrievalCostPenalty; }

    @StringGetter("autoEncoderModel")
    public String getEncoderModel() { return autoEncoderModel; }

    @StringSetter("autoEncoderModel")
    public void setEncoderModel(String autoEncoderModel) { this.autoEncoderModel = autoEncoderModel; }

}