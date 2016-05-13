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

#include "Node.h"
#include "ODTree.h"

Node::Node()
{
	left=NULL;
	right=NULL;
	featureIndex=-1;
	splitValue=0;
	depth=0;
	sampleIndexList=NULL;
	tree=NULL;
}

Node::~Node()
{

}

void Node::binSplitDataSet(int feature,  double featureValue,
		vector<int> *o_indexList0,vector<int> *o_indexList1)
{
	if(tree->trainData==NULL)return;
	o_indexList0->clear();
	o_indexList1->clear();
	for(int i=0;i<sampleIndexList->size();i++)
	{
		double sampleIndex=sampleIndexList->at(i);
		double value=getDataValue(tree->trainData,tree->Ns,tree->Nfp1,sampleIndex,feature);
		if(value>featureValue)
		{
			o_indexList0->push_back(sampleIndex);
		}
		else
		{
			o_indexList1->push_back(sampleIndex);
		}
	}
}

double Node::meanLeaf()
{
	if(sampleIndexList==NULL)return -1;
	double sum=0;
	for (int i=0;i<sampleIndexList->size();i++)
	{
		int sampleIndex=sampleIndexList->at(i);
		double label=getDataValue(tree->trainData,tree->Ns,tree->Nfp1,sampleIndex,tree->Nfp1-1);
		sum=sum+label;
	}
	double mean=sum/sampleIndexList->size();
	return mean;
}

double Node::impurityLeaf(vector<int> *i_sampleIndexList)
{

	// Gini index
	if(i_sampleIndexList==NULL)return -1;
	double N0=0;
	double N1=0;
	int sampleNumber=i_sampleIndexList->size();
	for (int i=0;i<sampleNumber;i++)
	{
		int sampleIndex=i_sampleIndexList->at(i);
		double label=getDataValue(tree->trainData,tree->Ns,tree->Nfp1,sampleIndex,tree->Nfp1-1);
		if(label>=1)
		{
			N1++;
		}
		else
		{
			N0++;
		}
	}
	return (1.0-(N0/sampleNumber)*(N0/sampleNumber)-(N1/sampleNumber)*(N1/sampleNumber))*sampleNumber;
}

void Node::chooseBestSplit(int * o_bestFeatureIndex, double * o_bestFeatureValue)
{
	bool singleCls=true;
	double label0=getDataValue(tree->trainData,tree->Ns,tree->Nfp1,sampleIndexList->at(0),tree->Nfp1-1);
	int sampleNumber=sampleIndexList->size();
    for(int i=0;i<sampleNumber;i++)
	{
		int sampleIndex=sampleIndexList->at(i);
		double label=getDataValue(tree->trainData,tree->Ns,tree->Nfp1,sampleIndex,tree->Nfp1-1);
		if(label!=label0)
		{
			singleCls=false;
			break;
		}
	}
	if(singleCls)
	{
		*o_bestFeatureIndex=-1;
		*o_bestFeatureValue=meanLeaf();
		return;
	}
	double S=impurityLeaf(sampleIndexList);
	double bestS=1000000;

	// randomly selected features for split
	int Nfsq=sqrt(tree->Nfp1-1)+1;
	for(int i=0;i<Nfsq;i++)
	{
		double randf=(double)rand()/RAND_MAX;
		int fIndex=(tree->Nfp1-1)*randf;
		double max=0;
		double min=0;
		getFeatureRange(tree->trainData, tree->Ns,tree->Nfp1, fIndex,
				&min,&max);
		for(int j=0;j<10;j++)
		{
			double splitValue=min+(max-min)*j/10;
			vector<int> *indexList0=new vector<int>;
			vector<int> *indexList1=new vector<int>;
			binSplitDataSet(fIndex,  splitValue,indexList0,indexList1);
			if(indexList0->size()<tree->sampleNumberThreshold || indexList1->size()<tree->sampleNumberThreshold) continue;

			double newS=impurityLeaf(indexList0)+impurityLeaf(indexList1);
			if(newS<bestS)
			{
				*o_bestFeatureIndex=fIndex;
				*o_bestFeatureValue=splitValue;
				bestS=newS;
			}
		}
	}

	if((S-bestS)<tree->varThreshold)
	{
		*o_bestFeatureIndex=-1;
		*o_bestFeatureValue=meanLeaf();
	}

}


void Node::CreateTree()
{
	int bestFeatureIndex;
	double bestFeatureValue;
	if(depth==tree->depthUpperBound)
	{
		bestFeatureIndex=-1;
		bestFeatureValue=meanLeaf();
		featureIndex=bestFeatureIndex;
		splitValue=bestFeatureValue;
		return;
	}
	chooseBestSplit(&bestFeatureIndex, &bestFeatureValue);
	featureIndex=bestFeatureIndex;
	splitValue=bestFeatureValue;
	if(bestFeatureIndex==-1)
	{
		return;
	}

	vector<int> * indexList0=new vector<int>;
	vector<int> * indexList1=new vector<int>;
	binSplitDataSet(bestFeatureIndex,  bestFeatureValue,
			indexList0,indexList1);

	Node * leftchild=new Node;
	leftchild->depth=depth+1;
	leftchild->sampleIndexList=indexList0;
	leftchild->tree=tree;
	leftchild->CreateTree();
	left=leftchild;

	Node * rightchild=new Node;
	rightchild->depth=depth+1;
	rightchild->sampleIndexList=indexList1;
	rightchild->tree=tree;
	rightchild->CreateTree();
	right=rightchild;
}

void Node::UpdateTree()
{
	if(featureIndex==-1)
	{
		CreateTree();
		return;
	}
	vector<int> * indexList0=new vector<int>;
	vector<int> * indexList1=new vector<int>;
	binSplitDataSet(featureIndex,  splitValue,
			indexList0,indexList1);
	left->sampleIndexList=indexList0;
	right->sampleIndexList=indexList1;
	left->UpdateTree();
	right->UpdateTree();
}

double Node::PredictOneSample(const double * i_inData, int Nf)
{
	if(featureIndex==-1)
	{
		return splitValue;
	}
	if(i_inData[featureIndex]>splitValue)
	{
		return left->PredictOneSample(i_inData, Nf);
	}
	else
	{
		return right->PredictOneSample(i_inData, Nf);
	}
}

void Node::ConvertTreeToList(int * io_left, int * io_right, 
        int *io_splitFeature, double *io_splitValue, 
        int currentListIndex, int * io_globalListIndex)
{
    io_splitFeature[currentListIndex]=featureIndex;
    io_splitValue[currentListIndex]=splitValue;
    if(featureIndex!=-1)
    {
        *io_globalListIndex=*io_globalListIndex+1;
        int leftListIndex=*io_globalListIndex;
        io_left[currentListIndex]=leftListIndex;
        left->ConvertTreeToList(io_left, io_right, 
        	io_splitFeature, io_splitValue, 
        	leftListIndex, io_globalListIndex);
        
        *io_globalListIndex=*io_globalListIndex+1;
        int rightListIndex=*io_globalListIndex;
        io_right[currentListIndex]=rightListIndex;
        right->ConvertTreeToList(io_left, io_right, 
        	io_splitFeature, io_splitValue, 
        	rightListIndex, io_globalListIndex);
    }
}
