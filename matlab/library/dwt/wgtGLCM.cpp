//
//  wgtDWT.cpp
//  wgtWavelet
//
//  Created by Guotai Wang on 12/01/2015.
//  Copyright (c) 2015 Guotai Wang. All rights reserved.
//

//------------------------------------------------------------------------
// guotai wang
// guotai.wang.14@ucl.ac.uk
// 9 Dec, 2014
//------------------------------------------------------------------------

#include "mex.h"
#include "wgtCooccurence.h"
#include "pixelOperation.h"
#include <iostream>
#include <cmath>

using namespace std;

// glcmfeatures=wgtGLCM(I)
void mexFunction(int			nlhs, 		/* number of expected outputs */
                 mxArray		*plhs[],	/* mxArray output pointer array */
                 int			nrhs, 		/* number of inputs */
                 const mxArray	*prhs[]		/* mxArray input pointer array */)
{
    // input checks
    if (nrhs != 1 )
    {
        //mexErrMsgTxt ("USAGE: glcmfeatures=wgtGLCM(I);");
        mexErrMsgTxt ("USAGE: glcmfeatures=wgtGLCM(I);");
    }
    const mxArray *I = prhs[0];
    unsigned char * IPr=(unsigned char *)mxGetPr(I);
    mwSize height = mxGetM(I);//height
    mwSize width = mxGetN(I);//width
    long pixelNumber=height*width;
    int binNum=4;
    unsigned char * pCohorizon=new unsigned char[width*height];
    cooccurance(IPr,pCohorizon,height,width,0,2,binNum);
    unsigned char * pCovertical=new unsigned char[width*height];
    cooccurance(IPr,pCovertical,height,width,2,0,binNum);
    
    int regionRadius=3;
    int regionSize=2*regionRadius+1;
    int regionArea=regionSize*regionSize;
    int featureLength=binNum*binNum*2;
    int halfFeatureLength=binNum*binNum;
    
//    plhs[0] = mxCreateDoubleMatrix(pixelNumber,featureLength,mxREAL);
    plhs[0] =mxCreateNumericMatrix(pixelNumber, featureLength, mxUINT8_CLASS, mxREAL);
    unsigned char *featureSets=(unsigned char *)mxGetData(plhs[0]);
    for(int i=0;i<height;i++)
    {
        for(int j=0;j<width;j++)
        {
            unsigned char *feature=new unsigned char[featureLength];
            for(int index=0;index<featureLength;index++)
            {
                feature[index]=0;
            }
            for(int x=-regionRadius;x<=regionRadius;x++)
            {
                for(int y=-regionRadius;y<=regionRadius;y++)
                {
                    int horizonValue=getPixel(pCohorizon,height,width,i+x,j+y);
                    int verticalValue=getPixel(pCovertical,height,width,i+x,j+y);
                    feature[horizonValue]++;
                    feature[verticalValue+halfFeatureLength]++;
                }
            }
            for(int l=0;l<featureLength;l++)
            {
                setPixel(featureSets,pixelNumber,featureLength,i+j*height,l,*(feature+l));
            }
        }
    }
}
