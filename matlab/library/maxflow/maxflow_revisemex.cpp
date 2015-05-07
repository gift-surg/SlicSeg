//------------------------------------------------------------------------
// MAXFLOWMEX	A wrapper library for maximum flow calculation.
//	Input:
//	A - an NxN sparse matrix of real entries.
//	T - as Nx2 sparse matrix of real entries.
//  Output:
//  flow - maximum flow value
//  labels - a vector of size Nx1 containing the label of each node 
//           respectively
//
//  Note that it is not guaranteed that A will be checked for correct
//  construction (e.g. self loops, etc). That is, garbage in - garbage out.
// 
//	(c) 2008 Michael Rubinstein, WDI R&D and IDC
//	$Revision: 140 $
//	$Date: 2008-09-15 15:35:01 -0700 (Mon, 15 Sep 2008) $

// calculating possibility from seeds and then constructing the graph
//------------------------------------------------------------------------

#include "mex.h"
#include "maxflow-v3.0/graph.h"
#include <iostream>
#include <cmath>
#define   FOREGROUND_LABEL 127
#define   BACKGROUND_LABEL 255
using namespace std;

// [flow label]=maxflow_reivsemex(I,Seeds,lambda,sigma);
void mexFunction(int			nlhs, 		/* number of expected outputs */
                 mxArray		*plhs[],	/* mxArray output pointer array */
                 int			nrhs, 		/* number of inputs */
                 const mxArray	*prhs[]		/* mxArray input pointer array */)
{
    // input checks
    if (nrhs != 4 )
    {
        mexErrMsgTxt ("USAGE: [flow label]=maxflow_revise(I,Seeds,lambda,sigma);");
    }
    const mxArray *I = prhs[0];
    const mxArray *Seed = prhs[1];
    double lamda=* mxGetPr(prhs[2]);
    double sigma= * mxGetPr(prhs[3]);
    
    unsigned char * IPr=(unsigned char *)mxGetPr(I);
    unsigned char * SeedPr=(unsigned char *)mxGetPr(Seed);
    // size of image
    mwSize m = mxGetM(I);//height
    mwSize n = mxGetN(I);//width
    
    //get distribution parameter of forground
    float f_sum=0;
    float f_mean;
    float f_squaresum=0;
    float f_var;
    int   f_count;

    float b_sum=0;
    float b_mean;
    float b_squaresum=0;
    float b_var;
    int   b_count;
    
    for(int x=0;x<n;x++)
    {
        for(int y=0;y<m;y++)
        {
            float pValue=(float)*(IPr+x*m+y);
            unsigned char label=*(SeedPr+x*m+y);
            if(label==FOREGROUND_LABEL)
            {
                f_sum+=pValue;
                f_squaresum+=pValue*pValue;
                f_count++;
//                 if(f_min>pValue)
//                 {
//                     f_min=pValue;
//                 }
//                 if(f_max<pValue)
//                 {
//                     f_max=pValue;
//                 }
            }
            else if(label==BACKGROUND_LABEL)
            {
                b_sum+=pValue;
                b_squaresum+=pValue*pValue;
                b_count++;
//                 if(b_min>pValue)
//                 {
//                     b_min=pValue;
//                 }
//                 if(b_max<pValue)
//                 {
//                     b_max=pValue;
//                 }
            }
        }
    }
    f_mean=f_sum/f_count;
    f_var=f_squaresum/f_count-f_mean*f_mean;
//     b_mean1=(b_min+f_min)/2;
//     b_var1=f_min>b_min?(f_min-b_min)/2:(b_min-f_min)/2;
//     b_mean2=(b_max+f_max)/2;
//     b_var2=f_max>b_max?(f_max-b_max)/2:(b_max-f_max)/2;
    
//     f_mean=(f_min+f_max)/2;
//     f_var=(f_mean-f_min)/2;
//     b_mean=(b_min+b_max)/2;
//     b_var=(b_mean-b_min)/2;
    b_mean=b_sum/b_count;
    b_var=b_squaresum/b_count-b_mean*b_mean;

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
               float v=(float)*(IPr+x*m+y);
               float forePosibility=exp(-(v-f_mean)*(v-f_mean)/(2*f_var))/(sqrt(2*3.14*f_var));
               float backPosibility=exp(-(v-b_mean)*(v-b_mean)/(2*b_var))/(sqrt(2*3.14*b_var));
               s_weight=-log(backPosibility);
               t_weight=-log(forePosibility);
            }
            int pIndex=x*m+y;
            g->add_tweights(pIndex,s_weight,t_weight);
        }
    }
    // return the results
    plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
    double* flow = mxGetPr(plhs[0]);
    *flow = g->maxflow();
    printf("max flow: %f\n",*flow);
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

