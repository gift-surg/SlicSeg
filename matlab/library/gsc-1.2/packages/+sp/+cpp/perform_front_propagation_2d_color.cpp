/*=================================================================
% perform_front_propagation_2d - perform a Fast Marching front propagation.
%
%   [D,S,Q,stPoints] = perform_front_propagation_2d(W,start_points,end_points,nb_iter_max,H);
%
%   'D' is a 2D array containing the value of the distance function to seed.
%	'S' is a 2D array containing the state of each point : 
%		-1 : dead, distance have been computed.
%		 0 : open, distance is being computed but not set.
%		 1 : far, distance not already computed.
%  'Q' contains the backlink to the node it came from (0 indexed)
%  stPoints -> index of the root node to which the point is nearest to
%	'W' is the weight matrix (inverse of the speed).
%	'start_points' is a 2 x num_start_points matrix where k is the number of starting points.
%	'H' is an heuristic (distance that remains to goal). This is a 2D matrix.
%   
%   Copyright (c) 2004 Gabriel Peyré
*=================================================================*/

//#include "perform_front_propagation_2d_color.h"
//#include "fheap/fib.h"
//#include "fheap/fibpriv.h"
#include "heaps/f_heap.hpp"
#include "mex.h"
#include <math.h>
#include "config.h"
#include <stdio.h>
#include <string.h>
#include <vector>
#include <algorithm>
#define kDead -1
#define kOpen 0
#define kFar 1

/* Global variables */
int n;			// height
int p;			// width
int nCh; // The number of channels in the image
int nPixels; // The number of pixels = n*p
double* D = NULL;
double* S = NULL;
double* stPoints=NULL;
double* W = NULL;
double* Q = NULL;
double* start_points = NULL;
double* end_points = NULL;
double* H = NULL;
double* L = NULL;
int nb_iter_max = 100000;
int nb_start_points = 0;
int nb_end_points = 0;
//fibheap_el** heap_pool = NULL;

#define ACCESS_ARRAY(a,i,j) a[(i)+n*(j)]
#define ACCESS_ARRAY_3D(a,i,j,k) a[(i)+n*(j)+k*nPixels]
#define D_(i,j) ACCESS_ARRAY(D,i,j)
#define S_(i,j) ACCESS_ARRAY(S,i,j)
#define W_(i,j,k) ACCESS_ARRAY_3D(W,i,j,k)
#define H_(i,j) ACCESS_ARRAY(H,i,j)
#define Q_(i,j) ACCESS_ARRAY(Q,i,j)
#define stPoints_(i,j) ACCESS_ARRAY(stPoints,i,j)
#define L_(i,j) ACCESS_ARRAY(L,i,j)
#define heap_pool_(i,j) ACCESS_ARRAY(heap_pool,i,j)

#define start_points_(i,k) start_points[(i)+2*(k)]
#define end_points_(i,k) end_points[(i)+2*(k)]

class point
{
public:
	int i,j;
	point( int ii, int jj )
	{ i = ii; j = jj; }

  friend int operator<(const point& rhs,const point& lhs){
	if( H==NULL )
		return D_(lhs.i,lhs.j)< D_(rhs.i,rhs.j);
	else
		return D_(lhs.i,lhs.j)+H_(lhs.i,lhs.j)<D_(rhs.i,rhs.j)+H_(rhs.i,rhs.j);
  }
  // Note: The meaning of the < operator is reversed, because using a max heap here

};

using namespace boost;
fibonacci_heap<point>::pointer* heap_pool = NULL;

typedef std::vector<point*> point_list;

inline 
bool end_points_reached(const int i, const int j )
{
	for( int k=0; k<nb_end_points; ++k )
	{
		if( i==((int)end_points_(0,k)) && j==((int)end_points_(1,k)) )
			return true;
	}
	return false;
}

// test the heap validity
void check_heap( int i, int j )
{
	for( int x=0; x<n; ++x )
		for( int y=0; y<p; ++y )
		{
			if( heap_pool_(x,y)!=NULL )
			{
				const point& pt = heap_pool_(x,y)->data();
				if( H==NULL )
				{
					if( D_(i,j)>D_(pt.i,pt.j) )
						mexErrMsgTxt("Problem with heap.\n");
				}
				else
				{
					if( D_(i,j)+H_(i,j)>D_(pt.i,pt.j)+H_(pt.i,pt.j) )
						mexErrMsgTxt("Problem with heap(H).\n");
				}
			}
		}
	return;
}

// select to test or not to test (debug purpose)
//#define CHECK_HEAP check_heap(i,j);
#define CHECK_HEAP


void perform_front_propagation_2d()
{
	// create the Fibonacci heap
  fibonacci_heap<point> open_heap;

	double h = 1.0/n;
	
	// initialize points
	for( int i=0; i<n; ++i )
	for( int j=0; j<p; ++j )
	{
		D_(i,j) = GW_INFINITE;
		S_(i,j) = kFar;
		Q_(i,j) = -1;
    stPoints_(i,j)=-1;
	}

	// record all the points
	heap_pool = new fibonacci_heap<point>::pointer[n*p]; 
	memset( heap_pool, NULL, n*p*sizeof(fibonacci_heap<point>::pointer) );

	// inialize open list
	point_list existing_points;
	for( int k=0; k<nb_start_points; ++k )
	{
		int i = (int) start_points_(0,k);
		int j = (int) start_points_(1,k);

		if( D_( i,j )==0 )
			mexErrMsgTxt("start_points should not contain duplicates.");

		point* pt = new point( i,j );
		existing_points.push_back( pt );			// for deleting at the end
		heap_pool_(i,j) = open_heap.push(*pt);			// add to heap
		D_( i,j ) = 0;
		S_( i,j ) = kOpen;
		//Q_(i,j) = k;
		stPoints_(i,j) = k;
		Q_(i,j) = i+n*j;
	}

	// perform the front propagation
	int num_iter = 0;
	bool stop_iteration = GW_False;
	while( !open_heap.empty() && num_iter<nb_iter_max && !stop_iteration )
	{
		num_iter++;

		// current point
		const point& cur_point = open_heap.top();
    open_heap.pop();
		int i = cur_point.i;
		int j = cur_point.j;
		heap_pool_(i,j) = NULL;
		S_(i,j) = kDead;
		stop_iteration = end_points_reached(i,j);

		CHECK_HEAP;

		// recurse on each neighbor
		int nei_i[4] = {i+1,i,i-1,i};
		int nei_j[4] = {j,j+1,j,j-1};
		for( int k=0; k<4; ++k )
		{
			int ii = nei_i[k];
			int jj = nei_j[k];
			bool bInsert = true;
			// check that the contraint distance map is ok
			if( ii>=0 && jj>=0 && ii<n && jj<p && bInsert )
			{
        double l1_gradient=0;
        int iCh;
        for(iCh=0;iCh<nCh;iCh++){
          l1_gradient+=GW_ABS(W_(ii,jj,iCh)-W_(i,j,iCh));
        }
				double A1 = D_(i,j) + l1_gradient;
				if( ((int) S_(ii,jj)) == kDead )
				{}
				else if( ((int) S_(ii,jj)) == kOpen )
				{
					// check if action has change.
					if( A1<D_(ii,jj) )
					{

						D_(ii,jj) = A1;
						// update the value of the closest starting point
						//Q_(ii,jj) = Q_(i,j);
						stPoints_(ii,jj) = stPoints_(i,j);
						Q_(ii,jj) = i+n*j;
						// Modify the value in the heap
            fibonacci_heap<point>::pointer cur_el = heap_pool_(ii,jj);
						if( cur_el!=NULL ){
							open_heap.increase(cur_el, cur_el->data() );	// use same data for update
              // Its increase instead of decrease because using a max_heap now!!
            }
						else
							mexErrMsgTxt("Error in heap pool allocation."); 
					}
				}
				else if( ((int) S_(ii,jj)) == kFar )
				{
					if( D_(ii,jj)!=GW_INFINITE )
						mexErrMsgTxt("Distance must be initialized to Inf");
					if( L==NULL || A1<=L_(ii,jj) )
					{
						S_(ii,jj) = kOpen;
						// distance must have change.
						D_(ii,jj) = A1;
						// update the value of the closest starting point
						//Q_(ii,jj) = Q_(i,j);
						stPoints_(ii,jj) = stPoints_(i,j);
						Q_(ii,jj) = i+n*j;
						// add to open list
						point* pt = new point(ii,jj);
						existing_points.push_back( pt );
						heap_pool_(ii,jj) = open_heap.push(*pt );			// add to heap	
					}
				}
				else 
					mexErrMsgTxt("Unkwnown state."); 
					
			}	// end switch
		}		// end for
	}			// end while

	// free point pool
	for( point_list::iterator it = existing_points.begin(); it!=existing_points.end(); ++it )
		GW_DELETE( *it );
	// free fibheap pool
	GW_DELETEARRAY(heap_pool);
	return;
}

void mexFunction(	int nlhs, mxArray *plhs[], 
				 int nrhs, const mxArray*prhs[] ) 
{ 
	/* retrive arguments */
	if( nrhs<4 ) 
		mexErrMsgTxt("4 or 5 input arguments are required."); 
	if( nlhs<4 ) 
		mexErrMsgTxt("4 output arguments are required."); 

	// first argument : weight list
  int numDims=mxGetNumberOfDimensions(prhs[0]);
  if(numDims < 3){
	  n = mxGetM(prhs[0]); 
	  p = mxGetN(prhs[0]);
    nCh = 1;
  }
  else if(numDims==3){
    const mwSize *dims=mxGetDimensions(prhs[0]);
    n = dims[0];
    p = dims[1];
    nCh = dims[2];
  }
  else{
    mexErrMsgTxt("Invalid size for the weight matrix\n");
  }
  nPixels=n*p;

	W = mxGetPr(prhs[0]);
	// second argument : start_points
	start_points = mxGetPr(prhs[1]);
	int tmp = mxGetM(prhs[1]); 
	nb_start_points = mxGetN(prhs[1]);
	if( tmp!=2 )
		mexErrMsgTxt("start_points must be of size 2 x nb_start_poins."); 
	// third argument : end_points
	end_points = mxGetPr(prhs[2]);
	tmp = mxGetM(prhs[2]); 
	nb_end_points = mxGetN(prhs[2]);
	if( nb_end_points!=0 && tmp!=2 )
		mexErrMsgTxt("end_points must be of size 2 x nb_end_poins."); 
	// third argument : nb_iter_max
	nb_iter_max = (int) *mxGetPr(prhs[3]);
	// second argument : heuristic
	if( nrhs>=5 )
	{
		H = mxGetPr(prhs[4]);
		if( mxGetM(prhs[4])==0 && mxGetN(prhs[4])==0 )
			H=NULL;
		if( H!=NULL && (mxGetM(prhs[4])!=n || mxGetN(prhs[4])!=p) )
			mexErrMsgTxt("H must be of size n x p."); 
	}
	else
		H = NULL;
	if( nrhs>=6 )
	{
		L = mxGetPr(prhs[5]);
		if( mxGetM(prhs[5])==0 && mxGetN(prhs[5])==0 )
			L=NULL;
		if( L!=NULL && (mxGetM(prhs[5])!=n || mxGetN(prhs[5])!=p) )
			mexErrMsgTxt("L must be of size n x p."); 
	}
	else
		L = NULL;
		
		
	// first ouput : distance
	plhs[0] = mxCreateDoubleMatrix(n, p, mxREAL); 
	D = mxGetPr(plhs[0]);
	// second output : state
	plhs[1] = mxCreateDoubleMatrix(n, p, mxREAL); 
	S = mxGetPr(plhs[1]);
	// third output : backlinks
	plhs[2] = mxCreateDoubleMatrix(n, p, mxREAL); 
	Q = mxGetPr(plhs[2]);
  
	// fourth output : startingpoints
	plhs[3] = mxCreateDoubleMatrix(n, p, mxREAL); 
	stPoints = mxGetPr(plhs[3]);

	// launch the propagation
	perform_front_propagation_2d();

	return;
}

