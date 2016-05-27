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

#include "ODTree.h"
#include <time.h>
#include <algorithm>


ODTree::ODTree()
{
	root=NULL;
	trainData=NULL;
	depthUpperBound=10;
	varThreshold=0.1;
	sampleNumberThreshold=10;
	srand(time(0));
}

ODTree::~ODTree()
{

}

void ODTree::setMaxDepth(int depth)
{
	depthUpperBound=depth;
}

void ODTree::Train(const double *i_trainData,int i_Ns,int i_Nfp1)
{
	if(root==NULL)//create tree
	{
		trainData=(double *)i_trainData;
		Ns=i_Ns;
		Nfp1=i_Nfp1;

		//online bagging
		vector<int> *sampleIndexList = new vector<int>;
		PoissonSequence(1, i_Ns,0.5, sampleIndexList);
		root=new Node;
		root->tree=this;
		root->sampleIndexList=sampleIndexList;
		root->CreateTree();
	}
	else //update tree, now training data is the expanded data set
	{
		trainData=(double *)i_trainData;
		int oldNs=Ns;
		Ns=i_Ns;

		//update sample list
		vector<int> *addSampleIndexList = new vector<int>;
		PoissonSequence(1, Ns-oldNs,0.5, addSampleIndexList);
		vector<int> *newSampleIndexList = new vector<int>;
		newSampleIndexList->resize(root->sampleIndexList->size()+addSampleIndexList->size());
		for(int i=0;i<newSampleIndexList->size();i++)
		{
			if(i<root->sampleIndexList->size())
			{
				newSampleIndexList->at(i)=root->sampleIndexList->at(i);
			}
			else
			{
				newSampleIndexList->at(i)=addSampleIndexList->at(i-root->sampleIndexList->size())+oldNs;
			}
		}
		root->sampleIndexList=newSampleIndexList;
		root->UpdateTree();
	}
}

void ODTree::Predict(const double *i_testData,int i_Ns,int i_Nf,double * io_forecast)
{
	for(int i=0;i<i_Ns;i++)
	{
		double *feature;
		feature=(double * )i_testData+i_Nf*i;
		*(io_forecast+i)=root->PredictOneSample(feature,i_Nf);
	}
}

double ODTree::GetOOBE()
{
	double correctPrediction=0;
	double totalPrediction=0;
	for(int i=0;i<Ns;i++)
	{
		if(std::find(root->sampleIndexList->begin(), root->sampleIndexList->end(), i)==root->sampleIndexList->end())
		{
		      totalPrediction++;
		      double prediction=root->PredictOneSample(trainData+i*Nfp1,Nfp1-1);
		      double trueLabel=getDataValue(trainData, Ns, Nfp1, i,Nfp1-1);
		      if((prediction-0.5)*(trueLabel-0.5)>0)
		      {
		    	  correctPrediction++;
		      }
		}
	}
	double oobe=-1;
	if(totalPrediction>0)
	{
		oobe=correctPrediction/totalPrediction;
	}
	return oobe;
}

void ODTree::ConvertTreeToList(int * io_left, int * io_right, 
        int *io_splitFeature, double *io_splitValue)
{
    int currentListIndex=0;
    int globalListIndex=0;
    root->ConvertTreeToList(io_left, io_right, 
        io_splitFeature, io_splitValue,
        currentListIndex,&globalListIndex);
}
