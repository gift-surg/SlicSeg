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

#include "ORForest.h"
#include <iostream>
#include <cstring>
using namespace std;
ORForest::ORForest()
{
	treeNumber=20;
	maxDepth=10;
	leastNsample=10;
	tree=NULL;
    trainData=NULL;
    Ns=0;
    Nfp1=0;
};

ORForest::~ORForest()
{
	if(tree)
	{
		delete [] tree;
	}
	if(trainData)free(trainData);
};

void ORForest::Init(int Ntree,int treeDepth, int leastNsampleForSplit)
{
	treeNumber=Ntree;
	maxDepth=treeDepth;
	leastNsample=leastNsampleForSplit;
}

void ORForest::Train(const double *i_trainData, int i_Ns,int i_Nfp1)
{
	if(tree==NULL)
	{
		Ns=i_Ns;
		Nfp1=i_Nfp1;
		trainData=(double *)malloc(sizeof(double)*Ns*Nfp1);
		memcpy( (void*) trainData, (const void*) i_trainData, sizeof(double)*Ns*Nfp1 );

		tree=new ODTree[treeNumber];
		for(int i=0;i<treeNumber;i++)
		{
			tree[i].setMaxDepth(maxDepth);
			tree[i].Train(trainData,Ns,Nfp1);
		}
	}
	else
	{
		//combine old training data and newly arrived training data
		int oldNs=Ns;
		Ns=Ns+i_Ns;
		double *newTrainData=(double *)malloc(sizeof(double)*Ns*Nfp1);
		memcpy( (void*) newTrainData, (const void*) trainData, sizeof(double)*oldNs*Nfp1 );
		memcpy( (void*) (newTrainData+oldNs*Nfp1), (const void*) i_trainData, sizeof(double)*i_Ns*Nfp1 );
		free(trainData);
		trainData=newTrainData;

		for(int i=0;i<treeNumber;i++)
		{
			tree[i].Train(trainData,Ns,Nfp1);
			double oobe=tree[i].GetOOBE();
			if(oobe<0.6)
			{
				tree[i].root=NULL;
				tree[i].Train(trainData,Ns,Nfp1);
			}
		}
	}
}

void ORForest::Predict(const double *i_testData, const int i_Ns, const int i_Nf, double *o_predict)
{
	double tempPredict[i_Ns];
	for(int i=0;i<treeNumber;i++)
	{
		tree[i].Predict(i_testData, i_Ns, i_Nf,tempPredict);
		for(int i=0;i<i_Ns;i++)
		{
			o_predict[i]+=tempPredict[i];
		}
	}
	for(int i=0;i<i_Ns;i++)
	{
		o_predict[i]=o_predict[i]/treeNumber;
	}
}

void ORForest::ConvertTreeToList(int * io_left, int * io_right, 
        int *io_splitFeature, double *io_splitValue, 
        int maxNodeNumber)
{
	for(int i=0;i<treeNumber;i++)
	{
        tree[i].ConvertTreeToList(io_left+i*maxNodeNumber, io_right+i*maxNodeNumber, 
        	io_splitFeature+i*maxNodeNumber, io_splitValue+i*maxNodeNumber);
	}
}

const int ORForest::getTreeNumber() const
{
    return treeNumber;
}

const int ORForest::getMaxDepth() const
{
    return maxDepth;
}


