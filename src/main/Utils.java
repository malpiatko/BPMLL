package main;

import mulan.evaluation.measure.InformationRetrievalMeasures;


public class Utils {
	
	public static double UAR(double tp, double fp, double tn, double fn) {
		double recall = InformationRetrievalMeasures.recall(
				tp, fp, fn);
		double specificity = InformationRetrievalMeasures.specificity(
				tn, fp, fn);
		return (recall + specificity) / 2;
	}

}
