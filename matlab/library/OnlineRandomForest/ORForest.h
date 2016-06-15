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

#ifndef ORFOREST_H_
#define ORFOREST_H_
#include "ODTree.h"

class ORForest
{
    
public:
    ORForest();
	~ ORForest();
	void Init(int Ntree, int treeDepth, int leastNsampleForSplit);
	void Train(const double *i_trainData, int i_Ns,int i_Nfp1);
	void Predict(const double *i_testData, const int i_Ns, const int i_Nf, double *o_predict);
    void ConvertTreeToList(int * io_left, int * io_right, 
        int *io_splitFeature, double *io_splitValue, 
        int maxNodeNumber);
    const int getTreeNumber() const;
    const int getMaxDepth() const;

private:
	int treeNumber;
	int maxDepth;
	int leastNsample;
    ODTree *tree;
    double *trainData;
    int Ns;
    int Nfp1;
};
#endif
