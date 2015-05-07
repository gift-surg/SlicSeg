#include "mex.h"
#include <float.h>
#include <memory.h>
#include <math.h>
#include "matrix.h"
#include <iostream>
#include <vector>

typedef unsigned int uint;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  /* Input parameters
     C         [ h x w x D] (double)
     roffset   [ K x 1] (int32)
     coffset   [ K x 1] (int32)
     Output:
     lEdges, rEdges, colorWts, spWeights
  */

  if (nrhs != 3)
    mexErrMsgTxt("3 input arguments expected.");
  if (nlhs != 4)
    mexErrMsgTxt("4 output arguments expected.");

  if(mxGetClassID(prhs[0])!=mxDOUBLE_CLASS) mexErrMsgTxt("prhs[0] (image) shd be of type double\n");
  if(mxGetClassID(prhs[1])!=mxINT32_CLASS) mexErrMsgTxt("prhs[1] (roffset) shd be of type int32\n");
  if(mxGetClassID(prhs[2])!=mxINT32_CLASS) mexErrMsgTxt("prhs[2] (coffset) shd be of type int32\n");

  int numDims=mxGetNumberOfDimensions(prhs[0]);
  
  int h=mxGetDimensions(prhs[0])[0];
  int w=mxGetDimensions(prhs[0])[1];
  int N=h*w;
  int D;
  if(numDims==3)
    D=mxGetDimensions(prhs[0])[2];
  else
    D=1;

  if(mxGetN(prhs[1])!=1 || mxGetN(prhs[2])!=1)
      mexErrMsgTxt("Invalid size for roffset or coffset\n");

  int K=mxGetM(prhs[1]);
  if(mxGetM(prhs[2])!=K)
    mexErrMsgTxt("Invalid size for roffset or coffset\n");

  int *roffsets=(int*)mxGetData(prhs[1]);
  int *coffsets=(int*)mxGetData(prhs[2]);

  double *C=mxGetPr(prhs[0]);

  int *indexOffsets=(int*)mxMalloc(K*sizeof(int));
  double *spOffsetWeights=(double*)mxMalloc(K*sizeof(double));

  int i;
  for(i=0;i<K;i++) {
    indexOffsets[i]=coffsets[i]*h+roffsets[i];
    spOffsetWeights[i]=(double)(coffsets[i]*coffsets[i]+roffsets[i]*roffsets[i]);
  }

  std::vector<int> lEdges;
  std::vector<int> rEdges;
  std::vector<uint> offsetNum;

  int x,y;
  for(x=0;x<w;x++){
    for(y=0;y<h;y++){
      int lEdge=y+x*h;
      for(i=0;i<K;i++){
        int r=y+roffsets[i];
        if(r>=0 && r<h){
          int c=x+coffsets[i];
          if(c>=0 && c<w){
            int rEdge=lEdge+indexOffsets[i];
            lEdges.push_back(lEdge);
            rEdges.push_back(rEdge);
            offsetNum.push_back(i);
          }
        }
      }
    }
  }

  int numEdges=lEdges.size();

  plhs[0]=mxCreateDoubleMatrix(numEdges,1,mxREAL);
  plhs[1]=mxCreateDoubleMatrix(numEdges,1,mxREAL);
  plhs[2]=mxCreateDoubleMatrix(numEdges,1,mxREAL);
  plhs[3]=mxCreateDoubleMatrix(numEdges,1,mxREAL);

  double *lEdges_out=mxGetPr(plhs[0]);
  double *rEdges_out=mxGetPr(plhs[1]);
  double *colorWeights=mxGetPr(plhs[2]);
  double *spWeights=mxGetPr(plhs[3]);

  for(i=0;i<numEdges;i++){
    int lEdge=lEdges[i];
    int rEdge=rEdges[i];
    lEdges_out[i]=(double)(lEdge+1);
    rEdges_out[i]=(double)(rEdge+1);

    int d;
    double colorW=0;
    for(d=0;d<D;d++){
      double diff=(C[lEdge+d*N]-C[rEdge+d*N]);
      colorW+=diff*diff;
    }

    colorWeights[i]=colorW;
    spWeights[i]=spOffsetWeights[offsetNum[i]];
    
  }

  // --- To free ---
  mxFree(indexOffsets);
  mxFree(spOffsetWeights);

}
