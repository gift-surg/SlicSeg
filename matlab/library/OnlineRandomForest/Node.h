// Online Random Forest
//
// This is a C++ file that is automatically compiled by the function CompileSlicSeg
//
// Author: Guotai Wang
// Copyright (c) 2014-2016 University College London, United Kingdom. All rights reserved.
// http://cmictig.cs.ucl.ac.uk
//
// Distributed under the BSD-3 licence. Please see the file licence.txt 
// This software is not certified for clinical use.
// 

#ifndef NODE_H_
#define NODE_H_

#include <stdlib.h>
#include "Common.h"

using namespace std;
class ODTree;

class Node
{
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

public:    
	vector<int> *sampleIndexList;
	ODTree *tree;

private:
	Node *left;
	Node *right;
	int featureIndex;
	double splitValue;
	int depth;    
};

#endif
