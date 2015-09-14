//
//  convolution.cpp
//  wgtWavelet
//
//  Created by Guotai Wang on 10/01/2015.
//  Copyright (c) 2015 Guotai Wang. All rights reserved.
//

#include "convolution.h"
double getPixel(const double *array,int H,int W,int i,int j)
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
void setPixel(double *array,int H,int W,int i,int j,double value)
{
    if(i<0 || i>=H || j<0 ||j>=W)
    {
        return;
    }
    *(array+H*j+i)=value;
}

void convolution(const double *arrayIn,double *arayOut,int H,int W,const double *kernel,int L,int scale,char dir)
{
    for(int i=0;i<H;i++)
    {
        for(int j=0;j<W;j++)
        {
            double sum=0;
            if(dir=='h' || dir=='H')
            {
                for (int k=0;k<L;k++)
                {
                    sum+=(*(kernel+k))*(getPixel(arrayIn, H,  W, i, j-scale*k));
                }
            }
            else
            {
                for (int k=0;k<L;k++)
                {
                    sum+=(*(kernel+k))*(getPixel(arrayIn, H,  W, i-scale*k, j));
                }
            }
            setPixel(arayOut,H,W,i,j,sum);
        }
    }
}