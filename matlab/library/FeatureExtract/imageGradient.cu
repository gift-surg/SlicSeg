// imageGradient
//
// This is a CUDA C++ file that is automatically compiled by the function CompileSlicSeg
//
// Author: Guotai Wang
// Copyright (c) 2014-2016 University College London, United Kingdom. All rights reserved.
// http://cmictig.cs.ucl.ac.uk
//
// Distributed under the BSD-3 licence. Please see the file licence.txt 
// This software is not certified for clinical use.
// 

__device__ double getPixel(const double *array, int H, int W, int i, int j)
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

__device__ void setPixel(double *array, int H, int W, int i, int j, double value)
{
    if(i<0 || i>=H || j<0 ||j>=W)
    {
        return;
    }
    *(array+H*j+i) = value;
}

__global__ void imageGradient(const double * inputData, double *g_mag, double *g_orient, const int H, const int W)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    double dx = getPixel(inputData, H, W, x+1, y)-getPixel(inputData, H, W, x-1, y);
    double dy = getPixel(inputData, H, W, x, y-1)-getPixel(inputData, H, W, x, y+1);
    double mag = sqrt(dx*dx+dy*dy);
    double orient = atan2(dy, dx);
    setPixel(g_mag, H, W, x, y, mag);
    setPixel(g_orient, H, W, x, y, orient);
}