/*=================================================================
% Do shortest path computation, given existing shortest paths
%
%   [D,S,Q,stPoints] = shortestPaths_constrained(W,start_points,end_points,nb_iter_max,geoGamma,nbrHoodSize,rescale_geo);
%
%   'D' is a 2D array containing the value of the distance function to seed.
%	'S' is a 2D array containing the state of each point : 
%		-1 : dead, distance have been computed.
%		 0 : open, distance is being computed but not set.
%		 1 : far, distance not already computed.
% 'Q' contains the backlink to the node it came from (0 indexed)
%  stPoints -> index of the root node to which the point is nearest to
%	'W' is the weight matrix (inverse of the speed).
%	'start_points' is a 2 x num_start_points matrix where k is the number of starting points.

%  Distance is computed as \sum (sqrt((1-geoGamma)eucledian_ij+geoGamma*|grad(I)_ij|^2) )
%  grad(I) is divided by the number of channels in the image
%   
%   Copyright (c) 2004 Gabriel Peyré
*=================================================================*/

#include "perform_front_propagation_2d_color.h"
//#include "fheap/fib.h"
//#include "fheap/fibpriv.h"
#include "heaps/f_heap.hpp"

#define kDead -1
#define kOpen 0
#define kFar 1

/* Global variables */
int n;			// height
int p;			// width
int nCh; // The number of channels in the image
int nPixels; // The number of pixels = n*p
int nbrHoodSize;
double Gamma=0;
double rescale_geo=1;
double* D = NULL;
double* D_in = NULL;
double* S = NULL;
double* W = NULL;
double* Q = NULL;
double* Q_in = NULL;
bool* prevSeg = NULL;
double* start_points = NULL;
double* stPoints=NULL;
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
#define D_in_(i,j) ACCESS_ARRAY(D_in,i,j)
#define prevSeg_(i,j) ACCESS_ARRAY(prevSeg,i,j)
#define S_(i,j) ACCESS_ARRAY(S,i,j)
#define W_(i,j,k) ACCESS_ARRAY_3D(W,i,j,k)
#define H_(i,j) ACCESS_ARRAY(H,i,j)
#define Q_(i,j) ACCESS_ARRAY(Q,i,j)
#define Q_in_(i,j) ACCESS_ARRAY(Q_in,i,j)
#define L_(i,j) ACCESS_ARRAY(L,i,j)
#define stPoints_(i,j) ACCESS_ARRAY(stPoints,i,j)
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


void perform_front_propagation_2d(T_callback_intert_node callback_insert_node)
{
	// create the Fibonacci heap
  fibonacci_heap<point> open_heap;

	double h = 1.0/n;
	
	// initialize points
	for( int i=0; i<n; ++i )
	for( int j=0; j<p; ++j )
	{
		//D_(i,j) = GW_INFINITE;
		D_(i,j) = D_in_(i,j);
		S_(i,j) = kFar;
		//Q_(i,j) = -1;
		Q_(i,j) = Q_in_(i,j);
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

		//if( D_( i,j )==0 )
			//mexErrMsgTxt("start_points should not contain duplicates.");

		point* pt = new point( i,j );
		existing_points.push_back( pt );			// for deleting at the end
		heap_pool_(i,j) = open_heap.push(*pt);			// add to heap
		D_( i,j ) = 0;
		S_( i,j ) = kOpen;
		//Q_(i,j) = k;
		stPoints_(i,j) = k;
		Q_(i,j) = i+n*j;
	}

  if(nbrHoodSize==4){
	  // perform the front propagation
	  int num_iter = 0;
	  bool stop_iteration = GW_False;
    double eucledianConstant=(1-Gamma);
    double geoScale=rescale_geo*Gamma;
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
      bool inForbidden=prevSeg_(i,j);

      CHECK_HEAP;

      // recurse on each neighbor
      int nei_i[4] = {i+1,i,i-1,i};
      int nei_j[4] = {j,j+1,j,j-1};
      for( int k=0; k<4; ++k )
      {
        int ii = nei_i[k];
        int jj = nei_j[k];
        // check that it is not going into the forbidden region
        bool entryOk=true;

        if( ii>=0 && jj>=0 && ii<n && jj<p)
        {
          if(!inForbidden && prevSeg_(ii,jj))
            entryOk=false;

          if(entryOk)
          {

            double l2_gradient=0;
            int iCh;
            for(iCh=0;iCh<nCh;iCh++){
              l2_gradient+=SQR(W_(ii,jj,iCh)-W_(i,j,iCh));
            }
            double A1 = D_(i,j) + sqrt(eucledianConstant+geoScale*l2_gradient);
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
              //if( D_(ii,jj)!=GW_INFINITE )
              //mexErrMsgTxt("Distance must be initialized to Inf");

              if(A1<D_(ii,jj)){

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
        }
      }		// end for
    }			// end while
  }
  else if(nbrHoodSize==8){
	  int num_iter = 0;
	  bool stop_iteration = GW_False;
    int iOffsets[8] = {1,1,0,-1,-1,-1,0,1};
    int jOffsets[8] = {0,1,1,1,0,-1,-1,-1};
    double geoScale=rescale_geo*Gamma;
    double eucledianConstants[8];
    for(int k=0;k<8;k++){
      eucledianConstants[k]=(1-Gamma)*(SQR(iOffsets[k])+SQR(jOffsets[k]));
    }
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
      bool inForbidden=prevSeg_(i,j);

      CHECK_HEAP;

      for( int k=0; k<8; ++k )
      {
        int ii = i+iOffsets[k];
        int jj = j+jOffsets[k];

        bool entryOk=true;

        if( ii>=0 && jj>=0 && ii<n && jj<p)
        {
          if(!inForbidden && prevSeg_(ii,jj))
            entryOk=false;
          
          if(entryOk)
          {
            double l2_gradient=0;
            int iCh;
            for(iCh=0;iCh<nCh;iCh++){
              l2_gradient+=SQR(W_(ii,jj,iCh)-W_(i,j,iCh));
            }
            double A1 = D_(i,j) + sqrt(eucledianConstants[k]+geoScale*l2_gradient);
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
              //if( D_(ii,jj)!=GW_INFINITE )
              //mexErrMsgTxt("Distance must be initialized to Inf");

              if(A1<D_(ii,jj)){

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
        }
      }		// end for
    }			// end while
  }
  else{mexErrMsgTxt("Invalid neighbourhood size\n");}

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
	if( nrhs<9 ) 
		mexErrMsgTxt("9 input arguments are required."); 
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
	H=NULL;
	L = NULL;
	Gamma=*mxGetPr(prhs[4]);	
  nbrHoodSize=(int)*mxGetPr(prhs[5]);
	rescale_geo=*mxGetPr(prhs[6]);	
  D_in=mxGetPr(prhs[7]);
  Q_in=mxGetPr(prhs[8]);
  if(mxGetClassID(prhs[9])!=mxLOGICAL_CLASS)
    mexErrMsgTxt("prevSeg has to be of type logical\n");

  prevSeg=(bool*)mxGetData(prhs[9]);

  if(Gamma<0 || Gamma>1)
    mexErrMsgTxt("Invalid Gamma for geodesic, has to be in [0,1]\n");
		
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
