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

#include "Common.h"
#include <math.h>

double getDataValue(const double *i_dataSet, int Ns,int Nfp1, int Is,int If)
{
  return *(i_dataSet+Is*Nfp1+If);
}

void getFeatureRange(const double * i_dataSet, int Ns,int Nfp1, int featureIndx,
		double *o_min,double *o_max)
{
	double min=100000;
	double max=-100000;
	for(int i=0;i<Ns;i++)
	{
		double value=getDataValue(i_dataSet,Ns,Nfp1,i,featureIndx);
		if(value>max)max=value;
		if(value<min)min=value;
	}
	*o_min=min;
	*o_max=max;
}

void PoissonSequence(int possionLambda, int Ns, double bagFactor, vector<int> *list)
{

	for(int i=0;i<Ns;i++)
	{
		double randNumber=(double)rand()/RAND_MAX;
		if(randNumber>bagFactor)continue;
		double L=exp(-possionLambda);
		int k=0;
		double p=1;
		do
		{
			k=k+1;
			double u=(double)rand()/RAND_MAX;
			p=p*u;
		}
		while(p>L);
		k=k-1;
		for(int j=0;j<k;j++)
		{
			list->push_back(i);
		}
	}
}