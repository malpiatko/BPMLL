package main;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import weka.core.Instance;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.neural.BPMLL;
import mulan.classifier.neural.BPMLLAlgorithm;
import mulan.classifier.neural.DataPair;
import mulan.data.DataUtils;
import mulan.data.MultiLabelInstances;

public class BPMLLWrapper extends BPMLL {
	
	BPMLLAlgorithm learnAlg;
	List<DataPair> trainData;
	int numInstances;
	double prevError = Double.MAX_VALUE;
	int[] outputStructure = null;
	

	public BPMLLWrapper(long random) {
		super(random);
	}

	public BPMLLWrapper() {
		super();
	}
	
	public void setOutputStructure(int[] outputStructure){
		this.outputStructure = outputStructure;
	}

	@Override
	protected void buildInternal(final MultiLabelInstances instances) throws Exception {
		nominalToBinaryFilter = null;

        MultiLabelInstances trainInstances = instances.clone();
        trainData = prepareData(trainInstances);
        Collections.shuffle(trainData, new Random(1));
        int inputsDim = trainData.get(0).getInput().length;
        model = buildNeuralNetwork(inputsDim);
        learnAlg = new BPMLLAlgorithm(model, weightsDecayCost);
        numInstances = trainData.size();
        epochs = 0;
	}
	
	public void learnEpoch() {
		epochs++;
        double error = 0;
        for (int index = 0; index < numInstances; index++) {
            DataPair trainPair = trainData.get(index);
            double result = learnAlg.learn(trainPair.getInput(), trainPair.getOutput(), learningRate);
            if (!Double.isNaN(result)) {
                error += result;
            }
        }
        double errorDiff = prevError - error;
        if (errorDiff <= ERROR_SMALL_CHANGE * prevError) {
            if (getDebug()) {
                debug("Global training error does not decrease enough. Training should be terminated.");
            }
        }
        thresholdF = buildThresholdFunction(trainData);
    }
	
	@Override
	public MultiLabelOutput makePredictionInternal(Instance instance) throws InvalidDataException {

        Instance inputInstance = null;
        if (nominalToBinaryFilter != null) {
            try {
                nominalToBinaryFilter.input(instance);
                inputInstance = nominalToBinaryFilter.output();
                inputInstance.setDataset(null);
            } catch (Exception ex) {
                throw new InvalidDataException("The input instance for prediction is invalid. " +
                        "Instance is not consistent with the data the model was built for.");
            }
        } else {
            inputInstance = DataUtils.createInstance(instance, instance.weight(), instance.toDoubleArray());
        }

        int numAttributes = inputInstance.numAttributes();
        if (numAttributes < model.getNetInputSize()) {
            throw new InvalidDataException("Input instance do not have enough attributes " +
                    "to be processed by the model. Instance is not consistent with the data the model was built for.");
        }

        // if instance has more attributes than model input, we assume that true outputs
        // are there, so we remove them
        List<Integer> someLabelIndices = new ArrayList<Integer>();
        boolean labelsAreThere = false;
        if (numAttributes > model.getNetInputSize()) {
            for (int index : this.labelIndices) {
                someLabelIndices.add(index);
            }

            labelsAreThere = true;
        }

        if (normalizeAttributes) {
            normalizer.normalize(inputInstance);
        }

        int inputDim = model.getNetInputSize();
        double[] inputPattern = new double[inputDim];
        int indexCounter = 0;
        for (int attrIndex = 0; attrIndex < numAttributes; attrIndex++) {
            if (labelsAreThere && someLabelIndices.contains(attrIndex)) {
                continue;
            }
            inputPattern[indexCounter] = inputInstance.value(attrIndex);
            indexCounter++;
        }

        double[] labelConfidences = model.feedForward(inputPattern);
        double threshold = thresholdF.computeThreshold(labelConfidences);
        boolean[] labelPredictions = new boolean[numLabels];
        Arrays.fill(labelPredictions, false);
        
        if(outputStructure == null){
        	for (int labelIndex = 0; labelIndex < numLabels; labelIndex++) {
                if (labelConfidences[labelIndex] > threshold) {
                    labelPredictions[labelIndex] = true;
                }
                // translate from bipolar output to binary
                labelConfidences[labelIndex] = (labelConfidences[labelIndex] + 1) / 2;
            }
        } else{
        	for(int labelIndex = 0, arrayIndex = 0; labelIndex < outputStructure.length; labelIndex++){
        		int cardinality = outputStructure[labelIndex];
        		if(cardinality == 1){
        			if (labelConfidences[arrayIndex] > threshold) {
                        labelPredictions[arrayIndex] = true;
                    }
                    // translate from bipolar output to binary
                    labelConfidences[labelIndex] = (labelConfidences[labelIndex] + 1) / 2;
        		} else {
        			double maxConfidence = -1;
            		int maxIndex = 0;
            		for(int carIndex = 0; carIndex < cardinality; carIndex++){
            			if(labelConfidences[arrayIndex + carIndex] > maxConfidence){
            				maxIndex = carIndex;
            				maxConfidence = labelConfidences[arrayIndex + carIndex];
            			}
            		}
            		for(int carIndex = 0; carIndex < cardinality; carIndex++){
            			if(carIndex == maxIndex){
            				labelPredictions[arrayIndex + carIndex] = true;
            			} else {
            				labelPredictions[arrayIndex + carIndex] = false;
             			}
            		}
        		}
        		
        		arrayIndex += cardinality;
        	}
        }
        

        MultiLabelOutput mlo = new MultiLabelOutput(labelPredictions, labelConfidences);
        return mlo;
    }

}
