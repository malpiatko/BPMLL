package main;

import main.Utils;
import mulan.evaluation.measure.InformationRetrievalMeasures;
import mulan.evaluation.measure.LabelBasedRecall;
import mulan.evaluation.measure.MacroAverageMeasure;

public class UARMulan extends LabelBasedRecall implements MacroAverageMeasure {

	public UARMulan(int numOfLabels) {
		super(numOfLabels);
	}

	@Override
	public String getName() {
		return "Unweighted-avarage recall";
	}

	@Override
	public double getValue() {
		double sum = 0;
		int count = 0;
		for (int labelIndex = 0; labelIndex < numOfLabels; labelIndex++) {
			sum += getValue(labelIndex);
			count++;
		}
		return sum / count;
	}

	@Override
	public double getValue(int labelIndex) {
		return InformationRetrievalMeasures.recall(truePositives[labelIndex],
				falsePositives[labelIndex],
				falseNegatives[labelIndex]);
	}

}
