#include "ODTree.h"

/////////////////////////////
//common functions
////////////////////////////
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

/////////////////////////////
//Node
////////////////////////////
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
	//cout<<"binSplitDataSet started"<<endl;
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
//	double sum=0;
//	double sq_sum=0;
//	for (int i=0;i<sampleNumber;i++)
//	{
//		int sampleIndex=i_sampleIndexList[i];
//		double label=getDataValue(i_dataSet,Ns,Nfp1,sampleIndex,Nfp1-1);
//		sum=sum+label;
//		sq_sum=sq_sum+label*label;
//	}
//	double mean=sum/sampleNumber;
//	double var=sqrt(sq_sum/sampleNumber-mean*mean);
//	return var*sampleNumber;

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
