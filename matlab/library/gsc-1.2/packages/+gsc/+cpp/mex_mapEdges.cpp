#include "mex.h"

void mexFunction(	int nlhs, mxArray *plhs[], 
				 int nrhs, const mxArray*prhs[] ) 
{ 
	/* retrive arguments */
	if( nrhs<4 ) 
		mexErrMsgTxt("4 input arguments are required."); 
	if( nlhs<1 ) 
		mexErrMsgTxt("1 output arguments are required."); 
	
	unsigned int *r=(unsigned int*)mxGetData(prhs[0]);
	unsigned int *c=(unsigned int*)mxGetData(prhs[1]);
	unsigned int *rw=(unsigned int*)mxGetData(prhs[2]);
	unsigned int *cw=(unsigned int*)mxGetData(prhs[3]);

	unsigned int numRows=mxGetM(prhs[2]);
	unsigned int numRows_orig=mxGetM(prhs[0]);

	int dims[2];dims[0]=numRows;dims[1]=1;
	plhs[0]=mxCreateNumericArray(2,dims,mxUINT32_CLASS,mxREAL);
	unsigned int *map=(unsigned int*)mxGetData(plhs[0]);

	unsigned int i=0;
	unsigned int curIdx=1;
	for(i=0;i<numRows;i++){
	  while( (*r)!=(*rw) || (*c)!=(*cw)){
	    r++;c++;curIdx++;
	    if(curIdx>numRows_orig)
	      mexErrMsgTxt("Unexpected input combination\n");
	  }
	  *map=curIdx;map++;
	  rw++;cw++;
	}

	return;
}
