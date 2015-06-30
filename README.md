#Testing on multi-target BPMLL

##Requirements
Requires jars for weka, meka and mulan

##How to run
The main method is in RunExperiment. There are three methods to be used in experiments:

1.	RunFullExperiment which will output the number of hidden neurons and epochs.
2.	ChooseSeed will output the best seed for given number hidden neurons and epochs.
3.	RunSingle is used to run on test when parameters are known, it's not a static method so initialisation is needed.

To run experiments all multi-class targets need to be changed to binary, for example:

age - {(Y,NY),(A,NA),(O,NO)}

Moreover, if an experiment is performed on only 1 or 2 binary targets it should also me seperated in a similar matter,
for example: {gender, likeability}. The reason for this is that BPMLL will discard all instances where all or
no labels are present.

### Variables
*	nTarget - number of binary targets, example {age, gender} -> 4
*	outputStructure - array representing the structure of the targets. ith element is the number of binary targets that the
ith class is made of, example {age, gender} -> {1,3}
*	mainTarget - the index of main target in the outputStructure
