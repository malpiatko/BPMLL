package main;

import java.util.ArrayList;
import java.util.Arrays;

import mulan.core.MulanException;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.Measure;

public class Fold{
	
	MultiLabelInstances train;
	MultiLabelInstances test;
	int mainTarget;
	private int epochs = 0;
	ArrayList<BPMLLWrapper> classifiers = new ArrayList<BPMLLWrapper>();
	int nClassifier = 5;
	private int[] outputStructure;

	
	Fold(MultiLabelInstances train, MultiLabelInstances test, int mainTarget, long seed,
			int[] outputStructure){
		this.mainTarget = mainTarget;
		this.train = train;
		this.test = test;
		this.outputStructure = outputStructure;
		if(seed == 0) {
			seed = 1;
			for(int i = 0; i < nClassifier; i++, seed *= 10) {
				BPMLLWrapper classifier = new BPMLLWrapper(seed);
				classifier.setOutputStructure(outputStructure);
				classifiers.add(classifier);
			}
		} else {
			BPMLLWrapper classifier = new BPMLLWrapper(seed);
			classifier.setOutputStructure(outputStructure);
			classifiers.add(classifier);
			nClassifier = 1;
		}
		
		
	}
	
	public void setLearningRate(double learningRate){
		for(int i = 0; i < nClassifier; i++){
			classifiers.get(i).setLearningRate(learningRate);
		}
	}
	
	public void setHiddenLayers(int[] layers){
		for(int i = 0; i < nClassifier; i++){
			classifiers.get(i).setHiddenLayers(layers);
		}
	}

	public void build() throws Exception {
		for(int i = 0; i < nClassifier; i++){
			classifiers.get(i).build(train);
		}
	}
	
	
	public double learnEpoch() throws Exception{
		epochs++;
		MultipleEvaluation mEval = new MultipleEvaluation(train);
		for(int i = 0; i < nClassifier; i++){
			BPMLLWrapper classifier = classifiers.get(i);
			classifier.learnEpoch();
			Evaluator eval = new Evaluator();
			ArrayList<Measure> measures = new ArrayList<Measure>();
			measures.add(new UARMulan(test.getNumLabels()));
			measures.add(new SingleUARMulan(test.getNumLabels()));
			Evaluation evaluation = eval.evaluate(classifier, test, measures);
			mEval.addEvaluation(evaluation);
		}
		mEval.calculateStatistics();
		
		return getRecall(mEval);
	}

	private double getRecall(MultipleEvaluation eval) throws MulanException{
		int index = getIndexOfLabel(mainTarget);
		if(outputStructure == null || outputStructure[mainTarget] == 1){
			return eval.getMean("SingleUAR", index);
		} else {
			double mean = 0;
			for(int i = 0; i < outputStructure[mainTarget]; i++){
				mean += eval.getMean("Unweighted-avarage recall", index+i);
			}
			return mean/outputStructure[mainTarget];
		}
	}
	
	private int getIndexOfLabel(int target){
		int index = 0;
		for(int i = 0; i < target; i++){
			index += outputStructure[i];
		}
		return index;
	}

	
}