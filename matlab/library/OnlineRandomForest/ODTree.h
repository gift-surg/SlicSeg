#ifndef ODTREE_H_
#define ODTREE_H_
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <algorithm> 
using namespace std;
class ODTree;
class Node
{
public:
	Node * left;
	Node * right;
	int featureIndex;
	double splitValue;
	int depth;
	vector<int> *sampleIndexList;
	ODTree *tree;
public:
	Node();
	~Node();
	void binSplitDataSet(int feature,  double featureValue,
			vector<int> *o_indexList0,vector<int> *o_indexList1);
	double meanLeaf();
	double impurityLeaf(vector<int> *i_sampleIndexList);
	void chooseBestSplit(int * o_bestFeatureIndex, double * o_bestFeatureValue);
	void CreateTree();
	void UpdateTree();
	double PredictOneSample(const double * i_inData, int Nf);
    void ConvertTreeToList(int * io_left, int * io_right, 
        int *io_splitFeature, double *io_splitValue, 
        int currentListIndex, int * io_globalListIndex);
};

class ODTree
{
public:
	Node *root;
	double *trainData;
	int Ns;
	int Nfp1;

	int depthUpperBound;
	double varThreshold;
	int sampleNumberThreshold;
public:
	ODTree();
	~ODTree();
	void setMaxDepth(int depth);
	void Train(const double *i_trainData,int i_Ns,int i_Nfp1);
	void Predict(const double *i_testData,int i_Ns,int i_Nf,double * io_forecast);
	double GetOOBE();//out of bag error
    void ConvertTreeToList(int * io_left, int * io_right, 
        int *io_splitFeature, double *io_splitValue);
};
#endif
