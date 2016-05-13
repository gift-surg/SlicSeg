// Online Random Forest common functions
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

#ifndef COMMON_H_
#define COMMON_H_

#include <stdlib.h>
#include <vector>
using namespace std;

double getDataValue(const double *i_dataSet, int Ns,int Nfp1, int Is,int If);

void getFeatureRange(const double * i_dataSet, int Ns,int Nfp1, int featureIndx,
		double *o_min,double *o_max);

void PoissonSequence(int possionLambda, int Ns, double bagFactor, vector<int> *list);

#endif
