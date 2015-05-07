#include "mex.h"
#include <float.h>
#include <memory.h>
#include <math.h>
#include "matrix.h"
#include <iostream>
#include <vector>

void displayDims( const char *name, const int *dims, int ndims ) {
  std::cout << name << " is [ " << dims[ 0 ];
  for( int i = 1; i < ndims; ++i )
    std::cout << " x " << dims[ i ];
  std::cout << " ]" << std::endl;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  /* Input parameters
     P         [ 1 x N ]
     X         [ D x N ]
     MU        [ D x 1 ]
     INVSIGMA  [ D x D ]
     K         [ 1 x 1 ]

     N = number of points
     D = number of dimensions

  */

  if (nrhs != 4)
    mexErrMsgTxt("4 input arguments expected.");
  if (nlhs != 1)
    mexErrMsgTxt("1 output argument expected.");

  {
    int ndims[] = {2, 2, 2, 2, 2};
    
    for( int i = 0; i < nrhs; ++i ) {
      if( !mxIsDouble( prhs[ i ] ) || mxIsComplex( prhs[ i ] ) ||
	  mxGetNumberOfDimensions( prhs[ i ] ) != ndims[ i ] ) {
	std::cout << "??? Input " << i << 
	  " is not a real double matrix with " <<
	  ndims[ i ] << " dimension(s): {double==" << 
	  mxIsDouble( prhs[ i ] ) << ",complex==" << 
	  mxIsComplex( prhs[ i ] ) << ",ndims==" << 
	  mxGetNumberOfDimensions( prhs[ i ] ) << "}" <<
	  std::endl << std::endl;
	mexErrMsgTxt("Error as above");
      }
    }
  }

  int N = mxGetN( prhs[ 0 ] );
  int D = mxGetM( prhs[ 0 ] );

  // check input dimensions
  const int *nX = mxGetDimensions( prhs[ 0 ] );
  const int *nMU = mxGetDimensions( prhs[ 1 ] );
  const int *nIS = mxGetDimensions( prhs[ 2 ] );
  const int *nK = mxGetDimensions( prhs[ 3 ] );

  if ( nX[ 0 ] != D || nX[ 1 ] != N ||
       nMU[ 0 ] != D || nMU[ 1 ] != 1 ||
       nIS[ 0 ] != D || nIS[ 1 ] != D ||
       nK[ 0 ] != 1 || nK[ 1 ] != 1 ) {
    std::cout << "D: " << D << std::endl <<
      "N: " << N << std::endl;
    displayDims( "X", nX, 2 );
    displayDims( "MU", nMU, 2 );
    displayDims( "IS", nIS, 2 );
    displayDims( "K", nK, 2 );
    mexErrMsgTxt("Input dimensions are not consistent.");
  }

  const double *in_X, *in_MU, *in_IS, *in_K;
  double *in_P;

  plhs[0]=mxCreateDoubleMatrix(1,N,mxREAL);
  in_P = (double *)mxGetPr( plhs[ 0 ] );
  in_X = (double *)mxGetPr( prhs[ 0 ] );
  in_MU = (double *)mxGetPr( prhs[ 1 ] );
  in_IS = (double *)mxGetPr( prhs[ 2 ] );
  in_K = (double *)mxGetPr( prhs[ 3 ] );

  std::vector< double > dX( D );
  
  const double *ptr_IS;
  double t1;
  double K = *in_K;
  // compute dX' * INVSIGMA * dX
  for( int p = 0; p < N; ++p, in_P++, in_X = &in_X[ D ] ) {
    *in_P = 0;
    // calc dX
    for( int i = 0; i < D; ++i )
      dX[ i ] = in_X[ i ] - in_MU[ i ];
    
    ptr_IS = in_IS;
    for( int i = 0; i < D; ++i, ptr_IS = &ptr_IS[D] ) {
      t1 = 0;
      for( int j = 0; j < D; ++j )
	t1 += ptr_IS[ j ] * dX[ j ];
      *in_P += t1 * dX[ i ];
    }
    *in_P = K * exp(*in_P * -0.5);
  }

}
