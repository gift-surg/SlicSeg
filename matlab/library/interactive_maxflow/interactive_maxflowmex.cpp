// interactive_maxflowmex: max flow with possibility for 2d slice (based on random forest)
//
// Syntax:
//   [flow, label] = interactive_maxflowmex(I, Seeds, Prob, lambda, sigma);
//
// This is a C++ mex file that is automatically compiled by the function CompileSlicSeg
//
// Partly derived from maxflow by Michael Rubinstein, WDI R&D and IDC. See copyright and licensing information in the library/maxflow folder
//
// Author: Guotai Wang
// Copyright (c) 2014-2016 University College London, United Kingdom. All rights reserved.
// http://cmictig.cs.ucl.ac.uk
//
// Distributed under the BSD-3 licence. Please see the file licence.txt 
// This software is not certified for clinical use.
// 

#include "mex.h"
#include "../maxflow/maxflow-v3.0/graph.h"
#include <iostream>
#include <cmath>
#define   FOREGROUND_LABEL 127
#define   BACKGROUND_LABEL 255
using namespace std;

void mexFunction(int			nlhs, 		/* number of expected outputs */
				 mxArray		*plhs[],	/* mxArray output pointer array */
				 int			nrhs, 		/* number of inputs */
				 const mxArray	*prhs[]		/* mxArray input pointer array */)
{
	// input checks
	if (nrhs != 5 )
	{
		mexErrMsgTxt ("USAGE: [flow label]=interactive_maxflowmex(I,Seeds,Prob,lambda,sigma); \n");
	}
	const mxArray *I = prhs[0];
	const mxArray *Seed = prhs[1];
    const mxArray *Prob = prhs[2];
    double lamda=* mxGetPr(prhs[3]);
    double sigma= * mxGetPr(prhs[4]);
    
    double * IPr=(double *)mxGetPr(I);
    unsigned char * SeedPr=(unsigned char *)mxGetPr(Seed);
    double * ProbPr=mxGetPr(Prob);
    
	// size of image
	mwSize m = mxGetM(I);//height
	mwSize n = mxGetN(I);//width

    //construct graph
    typedef Graph<float,float,float> GraphType;
    GraphType *g = new GraphType(/*estimated # of nodes*/ m*n, /*estimated # of edges*/ 2*m*n);
    g->add_node(m*n);
    float maxWeight=-10000;
    for(int x=0;x<n;x++)
    {
        for(int y=0;y<m;y++)
        {
            //n-link
            float pValue=(float)*(IPr+x*m+y);
            //int label=seed.at<uchar>(y,x);
            int uperPointx=x;
            int uperPointy=y-1;
            int LeftPointx=x-1;
            int LeftPointy=y;
            float n_weight=0;
            if(uperPointy>=0 && uperPointy<m)
            {
                float qValue=(float)*(IPr+uperPointx*m+uperPointy);
                n_weight=lamda*exp(-(pValue-qValue)*(pValue-qValue)/(2*sigma*sigma));
                int pIndex=x*m+y;
                int qIndex=uperPointx*m+uperPointy;

                g->add_edge(qIndex,pIndex,n_weight,n_weight);
            }
            if(LeftPointx>=0 && LeftPointx<n)
            {
                float qValue=(float)*(IPr+LeftPointx*m+LeftPointy);
                n_weight=lamda*exp(-(pValue-qValue)*(pValue-qValue)/(2*sigma*sigma));
                int pIndex=x*m+y;
                int qIndex=LeftPointx*m+LeftPointy;
                g->add_edge(qIndex,pIndex,n_weight,n_weight);
            }
            if(n_weight>maxWeight)
            {
                maxWeight=n_weight;
            }
        }
    }
    maxWeight=10*maxWeight;

    for(int x=0;x<n;x++)
    {
        for(int y=0;y<m;y++)
        {
            //t-link
            unsigned char label=*(SeedPr+x*m+y);
            float s_weight=0;
            float t_weight=0;
            if(label==FOREGROUND_LABEL)
            {
                s_weight=maxWeight;
            }
            else if(label==BACKGROUND_LABEL )
            {
                t_weight=maxWeight;
            }
            else
            {
                float forePosibility=*(ProbPr+x*m+y);
                float backPosibility=1.0-forePosibility;
                if(forePosibility>0.6)
                {
                    s_weight=-log(0.5+backPosibility*0.5);
                    t_weight=-log(0.5+forePosibility*0.5);  
                }
                else if(forePosibility<0.1)
                {
                    s_weight=-log(0.87+backPosibility*0.13);
                    t_weight=-log(0.87+forePosibility*0.13); 
                }
                else
                {
                    s_weight=-log(0.9+backPosibility*0.1);
                    t_weight=-log(0.9+forePosibility*0.1);
                }
            }
            int pIndex=x*m+y;
            g->add_tweights(pIndex,s_weight,t_weight);
        }
    }
    // return the results
	plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
	double* flow = mxGetPr(plhs[0]);
	*flow = g->maxflow();
    
    // printf("max flow: %f\n",*flow);
	// figure out segmentation
	plhs[1] = mxCreateNumericMatrix(m, n, mxUINT8_CLASS, mxREAL);
	unsigned char * labels = (unsigned char*)mxGetData(plhs[1]);
	for (int x = 0; x < n; x++)
	{
        for (int y=0;y<m;y++)
        {
            int Index=x*m+y;
            labels[Index] = g->what_segment(Index);
        }
	}
	// cleanup
	delete g;
}

