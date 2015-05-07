#ifndef ORFOREST_H_
#define ORFOREST_H_
#include "ODTree.h"
class ORForest
{
public:
	int treeNumber;
	int maxDepth;
	int leastNsample;
    ODTree *tree;

    double *trainData;
    int Ns;
    int Nfp1;
public:
    ORForest();
	~ ORForest();
	void Init(int Ntree,int treeDepth, int leastNsampleForSplit);
	void Train(const double *i_trainData, int i_Ns,int i_Nfp1);
	void Predict(const double *i_testData, int i_Ns, int i_Nf, double *o_predict);
    void ConvertTreeToList(int * io_left, int * io_right, 
        int *io_splitFeature, double *io_splitValue, 
        int maxNodeNumber);
};
#endif
