package main;

import main.Utils;
import mulan.evaluation.measure.LabelBasedRecall;
import mulan.evaluation.measure.MacroAverageMeasure;

public class SingleUARMulan extends LabelBasedRecall implements MacroAverageMeasure {

	public SingleUARMulan(int numOfLabels) {
		super(numOfLabels);
	}

	@Override
	public String getName() {
		return "SingleUAR";
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
		return Utils.UAR(truePositives[labelIndex],
					falsePositives[labelIndex],
					trueNegatives[labelIndex],
					falseNegatives[labelIndex]);
	}

}

