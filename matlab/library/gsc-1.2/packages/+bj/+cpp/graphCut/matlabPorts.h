#ifndef MATLAB_PORTS_H
#define MATLAB_PORTS_H

//#define myPrintf printf
#define myPrintf mexPrintf
#include "mex.h"

//inline void myFree(void* ptr){
  //free(ptr);
//}

inline void myFree(void* ptr){
  mxFree(ptr);
}

//inline void* myMalloc(size_t numBytes){
  //return malloc(numBytes); 
//}

inline void* myMalloc(size_t numBytes){
  void *m=mxMalloc(numBytes);
  mexMakeMemoryPersistent(m);
  return m;
}

#endif
