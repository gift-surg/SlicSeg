__device__ double getPixel(const double *array,int H,int W,int i,int j)
{
    if(i<0 || i>=H || j<0 ||j>=W)
    {
        return 0;
    }
    else
    {
        return *(array+H*j+i);
    }
}
__device__ void setPixel(double *array,int H,int W,int i,int j,double value)
{
    if(i<0 || i>=H || j<0 ||j>=W)
    {
        return;
    }
    *(array+H*j+i)=value;
}

//feval(k,gpuLL2,gpuLH2,gpuHL2,gpuHH2,gpuLH1,gpuHL1,gpuHH1,gpuDWT,H,W);
// size of FeatureMatrix: (H*W)*featureLength
__global__ void wgtDWTFeature(const double * pDataLL2, const double * pDataLH2,
	const double * pDataHL2, const double * pDataHH2,
    const double * pDataLH1, const double * pDataHL1,
    const double * pDataHH1, double * FeatureMatrix, const int height, const int width)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i>=height || j>=width) return;

    int regionSize=8;
    int halfSize=8/2;
    int quaterSize=8/4;
    int featureLength=regionSize*regionSize;
    

    int starti=i-halfSize-1;
    int startj=j-halfSize-1;
    double tempValue=0;
    for(int x=0;x<regionSize;x++)
    {
        for(int y=0;y<regionSize;y++)
        {
            if(x<quaterSize && y<quaterSize)
            {
                tempValue=getPixel(pDataLL2, height, width, starti+4*x, startj+4*y);
            }
            else if(x>=quaterSize && x<halfSize && y<quaterSize)
            {
                tempValue=getPixel(pDataLH2, height, width, starti+4*(x-quaterSize), startj+4*y);
            }
            else if(x<quaterSize && y>=quaterSize && y<halfSize)
            {
                tempValue=getPixel(pDataHL2, height, width, starti+4*x, startj+4*(y-quaterSize));
            }
            else if(x>=quaterSize && x<halfSize && y>=quaterSize && y<halfSize)
            {
                tempValue=getPixel(pDataHH2, height, width, starti+4*(x-quaterSize), startj+4*(y-quaterSize));
            }
            else if(x>=halfSize && y<halfSize)
            {
                tempValue=getPixel(pDataLH1, height, width, starti+2*(x-halfSize), startj+2*y);
            }
            else if(x<halfSize && y>=halfSize)
            {
                tempValue=getPixel(pDataHL1, height, width, starti+2*x, startj+2*(y-halfSize));
            }
            else{
                tempValue=getPixel(pDataHH1, height, width, starti+2*(x-halfSize), startj+2*(y-halfSize));
            }
            setPixel(FeatureMatrix,height*width,featureLength,i+j*height,x+y*regionSize,tempValue);
        }
    }
}