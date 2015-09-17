//
//  convolution.h
//  wgtWavelet
//
//  Created by Guotai Wang on 10/01/2015.
//  Copyright (c) 2015 Guotai Wang. All rights reserved.
//

#ifndef __wgtWavelet__convolution__
#define __wgtWavelet__convolution__

#include <stdio.h>
double getPixel(const double *array,int H,int W,int i,int j);
void setPixel(double *array,int H,int W,int i,int j,double value);
void convolution(const double *arrayIn,double *arayOut,int H,int W,const double *kernel,int L,int scale,char dir);
#endif /* defined(__wgtWavelet__convolution__) */
