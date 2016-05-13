// wgtDWTConvolution
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

// lp_hp----0,low pass or 1, high pass
// dir_h----true, horizonal, false,vertical
__global__ void wgtDWTConvolution(const double * pDataIn, double * pDataOut, const int H, const int W, 
    const int lp_hp, const int scale, const bool dir_h)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i>=H || j>=W) return;

    double kernelL[2]={0.7071067811865476, 0.7071067811865476}; //low pass filter
    double kernelH[2]={-0.7071067811865476,0.7071067811865476}; //high pass filter
    double *kernel=&kernelL[0];
    if(lp_hp>0) kernel=&kernelH[0];
    double sum=0;
    if(dir_h)
    {
        for (int k=0;k<2;k++)
        {
            sum+=(*(kernel+k))*(getPixel(pDataIn, H,  W, i, j-scale*k));
        }
    }
    else
    {
        for (int k=0;k<2;k++)
        {
            sum+=(*(kernel+k))*(getPixel(pDataIn, H,  W, i-scale*k, j));
        }
    }
    setPixel(pDataOut,H,W,i,j,sum);
}