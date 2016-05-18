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

#ifndef ODTREE_H_
#define ODTREE_H_
#include <stdlib.h>
#include <math.h>

#include "Common.h"
#include "Node.h"

using namespace std;

class ODTree
{
public:
	ODTree();
	~ODTree();
	void setMaxDepth(int depth);
	void Train(const double *i_trainData,int i_Ns,int i_Nfp1);
	void Predict(const double *i_testData,int i_Ns,int i_Nf,double * io_forecast);
	double GetOOBE();//out of bag error
    void ConvertTreeToList(int * io_left, int * io_right, 
        int *io_splitFeature, double *io_splitValue);
    
public:
	Node *root;
	double *trainData;
	int Ns;
	int Nfp1;
	int depthUpperBound;
	int sampleNumberThreshold;    
	double varThreshold;
};
#endif
