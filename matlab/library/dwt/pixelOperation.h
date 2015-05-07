//
//  pixelOperation.h
//  wgtWavelet
//
//  Created by Guotai Wang on 13/01/2015.
//  Copyright (c) 2015 Guotai Wang. All rights reserved.
//

#ifndef __wgtWavelet__pixelOperation__
#define __wgtWavelet__pixelOperation__

#include <stdio.h>
double getPixel(const double *array,int H,int W,int i,int j);
void setPixel(double *array,int H,int W,int i,int j,double value);
unsigned char getPixel(const unsigned char *array,int H,int W,int i,int j);
void setPixel(unsigned char *array,int H,int W,int i,int j,unsigned char value);
#endif /* defined(__wgtWavelet__pixelOperation__) */
