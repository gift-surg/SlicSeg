
__device__ double DTreePredictOneSample(const int * i_leftList, const int *i_rightList,
		const int * i_splitFeatureList, const double *i_splitValueList,
		const double * i_inData, int Nf,
		int currentNodeIndex)
{
	int currentFeature=i_splitFeatureList[currentNodeIndex];
	double currentSplitValue=i_splitValueList[currentNodeIndex];
	if(currentFeature==-1)
	{
		return currentSplitValue;
	}
	int childIndex;
	if(i_inData[currentFeature]>currentSplitValue)
	{
		childIndex=i_leftList[currentNodeIndex];
	}
	else
	{
		childIndex=i_rightList[currentNodeIndex];
	}
	return DTreePredictOneSample(i_leftList, i_rightList,
			i_splitFeatureList, i_splitValueList,
			i_inData, Nf,
			childIndex);
}
__global__ void ForestPredict(const int * leftList,const int * rightList,
        const int * splitFeatureList, const double * splitValueList, int Ntree, int maxNodeOnTree,
        const double *testData, int Ns, int Nf, double *prediction)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x>=Ns) return;
    //int y = blockIdx.y * blockDim.y + threadIdx.y;
    double * currentSample=(double *)testData+x*Nf;
    double sum=0;
    for(int i=0;i<Ntree;i++)
    {
        int * tempLeftList=(int *)leftList+i*maxNodeOnTree;
        int * tempRightList=(int *)rightList+i*maxNodeOnTree;
        int * tempSplitFeatureList=(int *)splitFeatureList+i*maxNodeOnTree;
        double * tempSplitValueList=(double *)splitValueList+i*maxNodeOnTree;
        sum=sum+DTreePredictOneSample(tempLeftList,tempRightList,tempSplitFeatureList,
                tempSplitValueList,currentSample,Nf,0);
    }
    *(prediction+x)=sum/Ntree;
}