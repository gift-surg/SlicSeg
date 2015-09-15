//
//  wgtCooccurance.cpp
//  wgtWavelet
//
//  Created by Guotai Wang on 13/01/2015.
//  Copyright (c) 2015 Guotai Wang. All rights reserved.
//

#include "wgtCooccurance.h"
#include "pixelOperation.h"

void cooccurance(const unsigned char *arrayIn,unsigned char *arayOut,int H,int W,int offsetH,int offsetW,int bin)
{
    float binLength=256/bin;
    for(int i=0;i<H;i++)
    {
        for(int j=0;j<W;j++)
        {
            int value0=getPixel(arrayIn, H, W, i, j)/binLength;
            int value1=getPixel(arrayIn, H, W, i+offsetH, j+offsetW)/binLength;
            setPixel(arayOut, H, W, i, j, value0*bin+value1);
        }
    }
}