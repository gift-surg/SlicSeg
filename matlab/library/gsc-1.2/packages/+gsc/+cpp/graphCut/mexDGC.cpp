#include "mex.h"
#include "graph.h"
#include <iostream>

#define MAX_HANDLES 100

static int numInstances=0;
static Graph<int,int,int>* gHandles[MAX_HANDLES];

void displayDims( const char *name, const int *dims, int ndims ) {
  std::cout << name << " is [ " << dims[ 0 ];
  for( int i = 1; i < ndims; ++i )
    std::cout << " x " << dims[ i ];
  std::cout << " ]" << std::endl;
}

void exitFunction(){
  myPrintf("Exit function called for mexDGC.cpp, clearing %d graphs\n",numInstances);
  for(int i=0;i<numInstances;i++){
    Graph<int,int,int>* g=gHandles[i];
    deleteGraph<int,int,int>(&g);
  }
  numInstances=0;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  /* Input parameters combinations
     rhs[0] = command -> A string that tells which command to do, depending on this command,
     here are the other parameter combinations: 

     Parameter combinations:
     command='initialize'
     rhs[1] = number of nodes(N) (1x1 UINT32 array)
     rhs[2] = edges (2xM UINT32 array of interacting node labels) M = number of interacting pairs
                    Note: node indices are 1-indexed for this array,

     lhs[0] -> returns the handle of the graph that should be passed to subsequent calls

     command='minimizeOnce'
     rhs[1] = number of nodes (1x1 UINT32 array)
     rhs[2] = a 2 x N array of unary energies, currently only INT32 is supported
     rhs[3] = edges (2xM or 3xM UINT32 array) M = number of interacting pairs
              if it is 2xM array, then interaction energies are given sequentially in rhs[4]
              if it is 3xM array, then interaction energies (in rhs[4]) are indexed by the 3rd row of this array
              This feature is provided, so that if you have a same prior for all, you can just have one
              interaction energy, and index it from the 3rd row. (again, 1 based indexing should be passed)
              If you dont understand this, then just do this:
              keep it a 2xM array, and in rhs[4], just put the corresponding energies

     rhs[4] = edge energies (4xL array of interaction energies, currently only INT32 supported)
              when rhs[3] is a 2xM array, then L should be = M,
              when rhs[3] is 3xM, then L can be smaller
              

     lhs[0] -> returns the binary labels of all nodes (array of type 1xN BOOL)
     lhs[1] -> returns the flow
     lhs[2] -> handle to graph (OPTIONAL)

     command='minimizeDynamic'
     rhs[1] -> handle to previous graph
     rhs[2] -> 1 x L UINT32 array, of nodes whose unary energies have been modified
     rhs[3] -> 2 x L array of the new updated unary energies (only INT32 supported right now)
     rhs[4] -> 2 x K UINT32 array of the new updated edges which have been modified
     rhs[5] -> 4 x K array of new updated interaction energies (only INT32 supported right now)
               [the 4 energies in order are q00,q01,q10,q11]
               Edit: rhs[5] can also be 2xL (in which the 00 and 11 energies are assumed 0)

     lhs[0] -> returns the binary labels of all nodes (array of type 1xN BOOL)
     lhs[1] -> returns the flow

     command='cleanup'
     rhs[1] -> handle to graph

     no output for this option

     // TO ADD, options for adding and deleting nodes and edges

     IMP: if in any combination, handle to graph is outputted, that means memory has been allocated inside,
          and you must call 'cleanup' with this handle to clear this memory, else it will become a memory leak
          Though on doing a clear all, i have set up the mexAtExit function properly, to clear all unallocated graphs

  */

  if (nrhs < 1)
    mexErrMsgTxt("Atleast one input required");

  if(mxGetClassID(prhs[0])!=mxCHAR_CLASS) mexErrMsgTxt("prhs[0] (command) should be of type char\n");
  char *command=mxArrayToString(prhs[0]); 

  typedef Graph<int,int,int> GraphInt;

  //myPrintf("DGC called with command %s\n",command);

  if(strcmp(command,"initialize")==0){
    mexAtExit(exitFunction);
    if(nrhs!=3) mexErrMsgTxt("Need 3 arguments for initialization\n");
    // --- Verify data types and size of the arguments -------
    if(mxGetClassID(prhs[1])!=mxUINT32_CLASS) mexErrMsgTxt("prhs[1] (number of nodes) shd be of type uint32\n");
    if(mxGetN(prhs[1])!=1 || mxGetM(prhs[1])!=1) mexErrMsgTxt("prhs[1] (number of nodes) shd be 1x1\n");

    if(mxGetClassID(prhs[2])!=mxUINT32_CLASS) mexErrMsgTxt("prhs[2] (edges) shd be of type uint32\n");
    if(mxGetM(prhs[2])!=2) mexErrMsgTxt("prhs[2] (number of nodes) shd be 2 x something\n");

    if(numInstances >= MAX_HANDLES){
      myPrintf("Max graph handles = %d\n",MAX_HANDLES);
      mexErrMsgTxt("Max number of instances of graph reached, please change limit in cpp file for more\n");
    }

    // ---- Data verified ok, now initialize the graph cut code
    int n,m;
    unsigned int* edgeTraveller;
    n=*((unsigned int*)mxGetData(prhs[1]));
    m=mxGetN(prhs[2]);
    edgeTraveller=(unsigned int*)mxGetData(prhs[2]);

    GraphInt *g=newGraph<int,int,int>(n,m,NULL); // put the new way of allocation here
    gHandles[numInstances]=g;
    numInstances++;
    
    g->add_node(n);
    for(int i=0;i<n;i++){g->add_tweights(i,0,0);}
    for(int j=0;j<m;j++){
      int src=*edgeTraveller-1;edgeTraveller++;
      int dest=*edgeTraveller-1;edgeTraveller++;
      g->add_edge(src,dest,0,0);
    }

    // IMP: might want to call g->maxflow(), if the dgc code does not initialize the search trees properly
    g->maxflow();
    // --- Now return back this handle --
    int numBytes=sizeof(GraphInt*);
    int dims[2];dims[0]=numBytes;dims[1]=1;
    plhs[0]=mxCreateNumericArray(2,dims,mxINT8_CLASS,mxREAL);
    memcpy(mxGetData(plhs[0]),&g,numBytes);
    //myPrintf("Sent graph = %x\n",*((unsigned int*)mxGetData(plhs[0])));

  }
  else if(strcmp(command,"minimizeOnce")==0){mexErrMsgTxt("Minimize Once not yet implemented\n");}
  else if(strcmp(command,"minimizeDynamic")==0){
    // -- Check data types ---
    if(mxGetClassID(prhs[1])!=mxINT8_CLASS) {mexErrMsgTxt("Invalid graph handle prhs[1]\n");}
    if(mxGetClassID(prhs[2])!=mxUINT32_CLASS) {mexErrMsgTxt("prhs[2]: Node indexes should be uint32\n");}
    if(mxGetClassID(prhs[3])!=mxINT32_CLASS) {mexErrMsgTxt("prhs[3]: unary energies shd be int32\n");}
    if(mxGetClassID(prhs[4])!=mxUINT32_CLASS) {mexErrMsgTxt("prhs[4]: edges, shd be uint32\n");}
    if(mxGetClassID(prhs[5])!=mxINT32_CLASS) {mexErrMsgTxt("prhs[5]: interaction energies shd be int32\n");}
    if(mxGetM(prhs[2])!=1 && mxGetM(prhs[2])!=0) {mexErrMsgTxt("prhs[2], incorrect dimensions\n");}
    if(mxGetM(prhs[3])!=2 && mxGetM(prhs[3])!=0) {mexErrMsgTxt("prhs[3], incorrect dimensions\n");}
    if(mxGetM(prhs[4])!=2 && mxGetM(prhs[4])!=0) {mexErrMsgTxt("prhs[4], incorrect dimensions\n");}
    if(mxGetM(prhs[5])!=4 && mxGetM(prhs[5])!=0 && mxGetM(prhs[5])!=2) {mexErrMsgTxt("prhs[5], incorrect dimensions\n");}
    if(nlhs < 1) {mexErrMsgTxt("Atleast one output required\n");}
    // --- Data verified ok ------------

    GraphInt *g;
    int numBytes=sizeof(GraphInt*);
    if(mxGetM(prhs[1])!=numBytes){mexErrMsgTxt("Something wrong with handle, incorrect size\n");}
    memcpy(&g,mxGetData(prhs[1]),numBytes);

    //myPrintf("Got graph = %x\n",(unsigned int)g);
    
    int sizeEdgeEnergies=mxGetM(prhs[5]);
    int nChanges=mxGetN(prhs[2]);
    int mChanges=mxGetN(prhs[4]);
    if(mxGetN(prhs[3])!=nChanges) {mexErrMsgTxt("Num of unary energies shd be equal to num of nodes supplied\n");}
    if(mxGetN(prhs[5])!=mChanges) {mexErrMsgTxt("Num of interaction energies, shd = num of edges supplied\n");}

    unsigned int *nodeIter = (unsigned int*)mxGetData(prhs[2]);
    int *unaryEnergyIter = (int*)mxGetData(prhs[3]);
    // --- Update the unary energies ---
    int i;
    for(i=0;i<nChanges;i++,nodeIter++){
      int si,it;
      si=*unaryEnergyIter;unaryEnergyIter++;
      it=*unaryEnergyIter;unaryEnergyIter++;
      g->edit_tweights(*nodeIter-1,si,it);// IMP: we are assuming 00 and 11 energies to be 0 right now, see note below
      g->mark_node(*nodeIter-1); 
      //g->edit_tweights_wt(*nodeIter-1,si,it); // IMP: we are assuming 00 and 11 energies to be 0 right now, see note below
    }

    // ---- Update the interaction energies ---
    // IMP: We have not yet implemented 00,and 11 energies, i.e we assume them to be 0
    // To implement them, we will need to save the unary energies seperately
    unsigned int *edgeIter = (unsigned int*)mxGetData(prhs[4]);
    int *interEnergyIter = (int*)mxGetData(prhs[5]);
    if(sizeEdgeEnergies==2){
      for(i=0;i<mChanges;i++){
        int e10,e01; 
        e01=*interEnergyIter;interEnergyIter++;
        e10=*interEnergyIter;interEnergyIter++;
        int src,dest;
        src=*edgeIter-1;edgeIter++;
        dest=*edgeIter-1;edgeIter++;

        g->edit_edge(src,dest,e10,e01); g->mark_node(src); g->mark_node(dest);
        //g->edit_edge_wt(src,dest,e01,e10);

      }
    }
    else{
      for(i=0;i<mChanges;i++){
        int e10,e01; //e00 and e11 are assumed 0, yet to be implemented
        interEnergyIter++;
        e01=*interEnergyIter;interEnergyIter++;
        e10=*interEnergyIter;interEnergyIter++;
        interEnergyIter++;
        int src,dest;
        src=*edgeIter-1;edgeIter++;
        dest=*edgeIter-1;edgeIter++;

        g->edit_edge(src,dest,e10,e01); g->mark_node(src); g->mark_node(dest);
        //g->edit_edge_wt(src,dest,e01,e10);

      }
    }
    
    //mexPrintf("Warning: Currently 00, and 11 energies are not supported, so will be ignoring those terms\n");
    int flow=g->maxflow(true);
    //int flow=g->maxflow(false);

    // --- Now prepare the output ------------
    int dims[2];dims[0]=1;dims[1]=g->get_node_num();
    plhs[0]=mxCreateNumericArray(2,dims,mxUINT8_CLASS,mxREAL);
    plhs[1]=mxCreateDoubleScalar((double)flow);
    unsigned char *segmentIter=(unsigned char*)mxGetData(plhs[0]);
    for(i=0;i<dims[1];i++,segmentIter++){
      *segmentIter=(g->what_segment(i)==GraphInt::SOURCE?0:1); // 0 for source, 1 for sink
    }

    //if(nlhs > 1) {mexErrMsgTxt("To add support for the flow and energy yet\n");}
    
  }
  else if(strcmp(command,"cleanup")==0){
    GraphInt *g;
    int numBytes=sizeof(GraphInt*);
    if(mxGetM(prhs[1])!=numBytes){mexErrMsgTxt("Something wrong with handle, incorrect size\n");}
    memcpy(&g,mxGetData(prhs[1]),numBytes);

    // Update the gHandles table to remove this one
    int i;
    for(i=0;i<numInstances;i++)
      if(gHandles[i]==g) break;

    if(i==numInstances){mexErrMsgTxt("Invalid graph handle passed to cleanup, graph doesnt exist\n");}

    numInstances--;
    for(int j=i;j<numInstances;j++){
     gHandles[j]=gHandles[j+1];
    }
    
    deleteGraph<int,int,int>(&g);
  }
  else{mexErrMsgTxt("Invalid command\n");}

  // --- Deallocating memories
  mxFree(command);
}
