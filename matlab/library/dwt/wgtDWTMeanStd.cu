// getMeanStdInRegion
//
// This is a CUDA C++ file that is automatically compiled by the function CompileSlicSeg
//
// Author: Guotai Wang
// Copyright (c) 2015-2016 University College London, United Kingdom. All rights reserved.
// http://cmictig.cs.ucl.ac.uk
//
// Distributed under the BSD-3 licence. Please see the file licence.txt 
// This software is not certified for clinical use.
// 

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

__device__ void getMeanStdInRegion(const double *array,int H,int W,int x,int roisize,int h0,int w0,int dh,int dw,double *mean,double *std)
{
    double sum=0;
    double sqSum=0;
    for(int i=h0;i<h0+dh;i++)
    {
        for(int j=w0;j<w0+dw;j++)
        {
            int y=i*roisize+j;
            double tempValue=getPixel(array,H,W,x,y);
            sum+=tempValue;
            sqSum+=tempValue*tempValue;
        }
    }
    int pixelNumber=dh*dw;
    *mean=sum/pixelNumber;
    double var=sqSum/pixelNumber-(*mean)*(*mean);
    *std=sqrt(var);
}

__global__ void wgtDWTMeanStd(const double * dwt,double *outputMean,double *outputStd,const int H,const int W)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x>=H) return;
    int size=8;
    int halfSize=size/2;
    int quaterSize=size/4;
    double tempMean[7];
    double tempStd[7];

    getMeanStdInRegion(dwt,H,W,x,size,0,0,quaterSize,quaterSize,&tempMean[0],&tempStd[0]);//LL2
    getMeanStdInRegion(dwt,H,W,x,size,quaterSize,0,quaterSize,quaterSize,&tempMean[1],&tempStd[1]);//LH2
    getMeanStdInRegion(dwt,H,W,x,size,0,quaterSize,quaterSize,quaterSize,&tempMean[2],&tempStd[2]);//HL2
    getMeanStdInRegion(dwt,H,W,x,size,quaterSize,quaterSize,quaterSize,quaterSize,&tempMean[3],&tempStd[3]);//HH2
    getMeanStdInRegion(dwt,H,W,x,size,halfSize,0,halfSize,halfSize,&tempMean[4],&tempStd[4]);//HL1
    getMeanStdInRegion(dwt,H,W,x,size,0,halfSize,halfSize,halfSize,&tempMean[5],&tempStd[5]);//LH1
    getMeanStdInRegion(dwt,H,W,x,size,halfSize,halfSize,halfSize,halfSize,&tempMean[6],&tempStd[6]);//HH1

    for(int i=0;i<7;i++)
    {
        setPixel(outputMean,H,7,x,i,tempMean[i]);
        setPixel(outputStd,H,7,x,i,tempStd[i]);
    }
}