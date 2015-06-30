package main;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.ArrayList;

import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;

public class RunExperiment {
	
	Fold fold;
	
	private int nTarget;
	private int epochs = 0;
	private long seed = 0;
	
	private double maxRecall = 0;
	private double maxEpoch = 0;
	

	private double learningRate;

	private int nHidden;

	private String trainFile;
	private String devFile;
	
	
	
	RunExperiment(String train, String dev, int nTarget, int mainTarget, int[] outputStructure) throws InvalidDataFormatException {
		this.nTarget = nTarget;
		this.fold = initialiseFold(train, dev, nTarget, mainTarget, 0, outputStructure);
		
	}
	
	RunExperiment(String train, String dev, int nTarget, int mainTarget, long seed, int[] outputStructure) throws InvalidDataFormatException{
		this.trainFile = train;
		this.devFile = dev;
		this.nTarget = nTarget;
		this.seed = seed;
		this.fold = initialiseFold(train, dev,nTarget, mainTarget, seed, outputStructure);
	}
	
	private Fold initialiseFold(String train, String dev, int nTarget, int mainTarget, long seed,
			int[] outputStructure) throws InvalidDataFormatException {
		MultiLabelInstances trainSet = new MultiLabelInstances(train, nTarget);
		MultiLabelInstances devSet = new MultiLabelInstances(dev, nTarget);
		return new Fold(trainSet, devSet, mainTarget, seed, outputStructure);
	}
	
	public void setLearningRate(double rate){
		this.learningRate = rate;
		fold.setLearningRate(rate);
	}
	
	public void setHiddenLayers(int[] layers){
		this.nHidden = layers[0];
		fold.setHiddenLayers(layers);
	}
	
	public void build() throws Exception{
		fold.build();
	}
	
	public double learnEpoch() throws Exception{
		epochs++;
		double recall = fold.learnEpoch();
		if (recall > maxRecall) {
			maxRecall = recall;
			maxEpoch = epochs;
		}
		return recall;
	}
	
	@Override
	public String toString(){
		return "Learning rate " +  learningRate
				+ ", hidden " + nHidden;
		
	}
	
	public int getEpochs(){
		return epochs;
	}
	
	
	public boolean isImprovement(double recall){
		return (epochs - maxEpoch) < 200;
	}
	
	public static void runFullExperiment(String train, String test, int nTarget, int mainTarget,
			int maxEpochs, int[] outputStructure) throws Exception{
		System.out.println(train);
		int[] hidden = {4,8,16,32,64};
		double[] rate = {0.1, 0.01,0.001};
		for(int rateIndex = 0; rateIndex < rate.length; rateIndex++){
			for(int hIndex = 0; hIndex < hidden.length; hIndex++){
				RunExperiment run = new RunExperiment(train, test, nTarget, mainTarget, 0, outputStructure);
				run.runSingleTest(rate[rateIndex], hidden[hIndex], false, maxEpochs);
			}
		}
		
	}
	
	public static void runFullExperimentSeparate(String train, String test, int nTarget, int mainTarget,
			int maxEpochs, int[] outputStructure) throws Exception{
		System.out.println(train);
		int[] hidden = {4,8,16,32,64};
		double[] rate = {0.1, 0.01,0.001};
		for(int rateIndex = 0; rateIndex < rate.length; rateIndex++){
			for(int hIndex = 0; hIndex < hidden.length; hIndex++){
				long seed = 1;
				for(int i = 0; i < 5; i++, seed *=10){
					RunExperiment run = new RunExperiment(train, test, nTarget, mainTarget, seed, outputStructure);
					run.runSingleTest(rate[rateIndex], hidden[hIndex], false, maxEpochs);
				}
				System.out.println();
			}
		}
		
	}
	
	
	public void runSingleTest(double rate, int hidden, boolean dump, int epochs) throws Exception{
		setLearningRate(rate);
		setHiddenLayers(new int[]{hidden});
		build();
		for(int i = 1; i < epochs + 1; i++){
			double recall = learnEpoch();
			if(!isImprovement(recall)) {
				break;
			}
		}
		System.out.println(toString());
		System.out.println("max epoch: " + maxEpoch + " " + maxRecall);
	}
	
	public static void chooseSeed(double learning, int hidden, String train, String dev, int nTarget, int mainTarget,
			int epochs, int[] outputStructure) throws Exception{
		System.out.print(train);
		int seed = 1;
		for(int i = 0; i < 5; i++, seed *= 10){
			System.out.println(seed);
			RunExperiment run = new RunExperiment(train, dev, nTarget, mainTarget, seed, outputStructure);
			run.runSingle(learning, hidden, epochs);
		}
	}
	
	public void runSingle(double rate, int hidden,int epochs) throws Exception{
		setLearningRate(rate);
		setHiddenLayers(new int[]{hidden});
		build();
		double recall = 0;
		for(int i = 1; i < epochs + 1; i++){
			recall = learnEpoch();
		}
		System.out.println(toString());
		System.out.println(recall);
	}
	
	
	
	
	
	public long getSeed() {
		return seed;
	}

	public static void main(String[] args) throws Exception {
		int nTarget = 15;
		int mainTarget = 0;
		int[] outputStructure = {12,1,1,1};
		String trainFile = "emotion/train/e_a_c2_train1.arff";
		String devFile = "likeability/dev/a_l_c2_dev1.arff";
		String testFile = "emotion/test/e_a_c2_test1.arff";
		String outputFile = "likeability/separateg_a_l_c2.txt";
		int epochs = 500;
		
		
		//System.setOut(new PrintStream(new File(outputFile)));
		
		/*for(int i = 0; i < 5; i++){
			System.out.println("DOING " + i + "\n\n\n");
			mainTarget = i + 1;
			runFullExperiment(trainFile, devFile, nTarget, mainTarget, epochs, outputStructure);
		}*/
		
		//train + dev pick learning, hidden, epoch
		//runFullExperimentSeparate(trainFile, devFile, nTarget, mainTarget, epochs, outputStructure);
		
		//train + dev pick seed with fixed rest
		//chooseSeed(0.1, 32, trainFile, devFile, nTarget, mainTarget, 217, outputStructure);
		
		//Run best parameters on test
		RunExperiment run = new RunExperiment(trainFile, testFile, nTarget, mainTarget, 1, outputStructure);
		run.runSingle(0.1, 64, 371);
	}

}
