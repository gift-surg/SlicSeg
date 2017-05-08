// intensityFeature
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
    *(array+H*j+i) = value;
}

__global__ void intensityFeature(const double *inputData, double *outputMean, double *outputvar, const int H, const int W, int r)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int kernel_radius = r;
    int kernel_size = 2*kernel_radius+1;
    if(x<kernel_radius || x>=H-kernel_radius ||
       y<kernel_radius || y>=W-kernel_radius)
    {
        double temp_value = *(inputData+x+y*H);
        setPixel(outputMean, H, W, x, y, temp_value);
        setPixel(outputvar, H, W, x, y, 0);
    }
    else
    {
        double sum = 0;
        double square_sum = 0;
        for(int i=-kernel_radius; i<=kernel_radius; i++)
        {
            for(int j=-kernel_radius; j<=kernel_radius; j++)
            {
                double tempValue = *(inputData + x + i + (y+j)*H);
                sum += tempValue;
                square_sum += tempValue*tempValue;
            }
        }
        double mean = sum/(kernel_size*kernel_size);
        double var = square_sum/(kernel_size*kernel_size) - mean*mean;
        var = sqrt(var);
        setPixel(outputMean, H, W, x, y, mean);
        setPixel(outputvar, H, W, x, y, var);
    }
}