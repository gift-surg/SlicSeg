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
#include "convolution.h"
#include <iostream>
#include <cmath>

using namespace std;

// dwtfeatures=wgtDWT(I)
void mexFunction(int			nlhs, 		/* number of expected outputs */
                 mxArray		*plhs[],	/* mxArray output pointer array */
                 int			nrhs, 		/* number of inputs */
                 const mxArray	*prhs[]		/* mxArray input pointer array */)
{
    // input checks
    if (nrhs != 1 )
    {
        //mexErrMsgTxt ("USAGE: dwtfeatures=wgtDWT(I);");
        mexErrMsgTxt ("USAGE: dwtfeatures=wgtDWT(I);");
    }
    const mxArray *I = prhs[0];
    unsigned char * IPr=(unsigned char *)mxGetPr(I);
    mwSize height = mxGetM(I);//height
    mwSize width = mxGetN(I);//width
    long pixelNumber=height*width;
    double * pDataIn=new double[width*height];
    for(int i=0;i<width*height;i++)
    {
        pDataIn[i]=IPr[i];
    }
    cout<<"image load success"<<endl;
    double kernelL[2]={0.7071067811865476,
        0.7071067811865476};
    double kernelH[2]={-0.7071067811865476,
        0.7071067811865476};
    //level 1
    double * pDatalow_horizon1=new double[width*height];
    double * pDatahigh_horizon1=new double[width*height];
    double * pDataLL1=new double[width*height];
    double * pDataLH1=new double[width*height];
    double * pDataHL1=new double[width*height];
    double * pDataHH1=new double[width*height];
    convolution(pDataIn, pDatalow_horizon1, height, width, &(kernelL[0]), 2, 1, 'h');
    convolution(pDataIn, pDatahigh_horizon1, height, width, &(kernelH[0]), 2, 1, 'h');
    
    convolution(pDatalow_horizon1, pDataLL1, height, width, &(kernelL[0]), 2, 1, 'v');
    convolution(pDatalow_horizon1, pDataLH1, height, width, &(kernelH[0]), 2, 1, 'v');
    convolution(pDatahigh_horizon1, pDataHL1, height, width, &(kernelL[0]), 2, 1, 'v');
    convolution(pDatahigh_horizon1, pDataHH1, height, width, &(kernelH[0]), 2, 1, 'v');
    cout<<"level 1 converlute success"<<endl;
    //level 2
    double * pDatalow_horizon2=new double[width*height];
    double * pDatahigh_horizon2=new double[width*height];
    double * pDataLL2=new double[width*height];
    double * pDataLH2=new double[width*height];
    double * pDataHL2=new double[width*height];
    double * pDataHH2=new double[width*height];
    
    convolution(pDataIn, pDatalow_horizon2, height, width, &(kernelL[0]), 2, 2, 'h');
    convolution(pDataIn, pDatahigh_horizon2, height, width, &(kernelH[0]), 2, 2, 'h');
    
    convolution(pDatalow_horizon2, pDataLL2, height, width, &(kernelL[0]), 2, 2, 'v');
    convolution(pDatalow_horizon2, pDataLH2, height, width, &(kernelH[0]), 2, 2, 'v');
    convolution(pDatahigh_horizon2, pDataHL2, height, width, &(kernelL[0]), 2, 2, 'v');
    convolution(pDatahigh_horizon2, pDataHH2, height, width, &(kernelH[0]), 2, 2, 'v');
    
    cout<<"level 2 converlute success"<<endl;
    int regionSize=8;
    int halfSize=8/2;
    int quaterSize=8/4;
    int featureLength=regionSize*regionSize;
    
    plhs[0] = mxCreateDoubleMatrix(pixelNumber,featureLength,mxREAL);
    double *featureSets=(double *)mxGetData(plhs[0]);
    for(int i=0;i<height;i++)
    {
        for(int j=0;j<width;j++)
        {
            int starti=i-halfSize-1;
            int startj=j-halfSize-1;
            double *feature=new double[featureLength];
            for(int x=0;x<regionSize;x++)
            {
                for(int y=0;y<regionSize;y++)
                {
                    if(x<quaterSize && y<quaterSize)
                    {
                        *(feature+y*regionSize+x)=getPixel(pDataLL2, height, width, starti+4*x, startj+4*y);
                    }
                    else if(x>=quaterSize && x<halfSize && y<quaterSize)
                    {
                        *(feature+y*regionSize+x)=getPixel(pDataLH2, height, width, starti+4*(x-quaterSize), startj+4*y);
                    }
                    else if(x<quaterSize && y>=quaterSize && y<halfSize)
                    {
                        *(feature+y*regionSize+x)=getPixel(pDataHL2, height, width, starti+4*x, startj+4*(y-quaterSize));
                    }
                    else if(x>=quaterSize && x<halfSize && y>=quaterSize && y<halfSize)
                    {
                        *(feature+y*regionSize+x)=getPixel(pDataHH2, height, width, starti+4*(x-quaterSize), startj+4*(y-quaterSize));
                    }
                    else if(x>=halfSize && y<halfSize)
                    {
                        *(feature+y*regionSize+x)=getPixel(pDataLH1, height, width, starti+2*(x-halfSize), startj+2*y);
                    }
                    else if(x<halfSize && y>=halfSize)
                    {
                        *(feature+y*regionSize+x)=getPixel(pDataHL1, height, width, starti+2*x, startj+2*(y-halfSize));
                    }
                    else{
                        *(feature+y*regionSize+x)=getPixel(pDataHH1, height, width, starti+2*(x-halfSize), startj+2*(y-halfSize));
                    }
                    
                }
            }
            for(int l=0;l<featureLength;l++)
            {
                setPixel(featureSets,pixelNumber,featureLength,i+j*height,l,*(feature+l));
            }
        }
    }
}
