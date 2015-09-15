#include "mex.h"
#include "class_handle.hpp"
#include "ORForest.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{	
    // Get the command string
    char cmd[64];
	if (nrhs < 1 || mxGetString(prhs[0], cmd, sizeof(cmd)))
		mexErrMsgTxt("First input should be a command string less than 64 characters long.");
        
    // New
    if (!strcmp("new", cmd)) {
        // Check parameters
        if (nlhs != 1)
            mexErrMsgTxt("New: One output expected.");
        // Return a handle to a new C++ instance
        plhs[0] = convertPtr2Mat<ORForest>(new ORForest());
//         plhs[0] = convertPtr2Mat<Forest>(&f);
        return;
    }
    
    // Check there is a second input, which should be the class instance handle
    if (nrhs < 2)
		mexErrMsgTxt("Second input should be a class instance handle.");
    
    // Delete
    if (!strcmp("delete", cmd)) {
        // Destroy the C++ object
        destroyObject<ORForest>(prhs[1]);
        // Warn if other commands were ignored
        if (nlhs != 0 || nrhs != 2)
            mexWarnMsgTxt("Delete: Unexpected arguments ignored.");
        return;
    }
    
    // Get the class instance pointer from the second input
    ORForest *rf_instance = convertMat2Ptr<ORForest>(prhs[1]);
    
    // Call the various class methods
    // Train(a,I);//I is a matrix for training    
    if (!strcmp("Init", cmd)) {
        // Check parameters
        int treeN=* mxGetPr(prhs[2]);
        int treeDepth=* mxGetPr(prhs[3]); 
        int leastNsample=* mxGetPr(prhs[4]);
     
        rf_instance->Init(treeN,treeDepth,leastNsample);
        if (nlhs < 0 || nrhs < 2)
            mexErrMsgTxt("Train: Unexpected arguments.");
        // Call the method
        //dummy_instance->train();
        return;
    }
    if (!strcmp("Train", cmd)) {
        // Check parameters
        const mxArray *TraingSet=prhs[2];
        mwSize m=mxGetM(TraingSet);// row number=featureNumber+1
        mwSize n=mxGetN(TraingSet);//column number=traning data number
        double* TrainingSetPr=(double *)mxGetPr(TraingSet);
        rf_instance->Train(TrainingSetPr,n,m);
        if (nlhs < 0 || nrhs < 2)
            mexErrMsgTxt("Train: Unexpected arguments.");
        // Call the method
        //dummy_instance->train();
        return;
    }
    // Test    
    if (!strcmp("Predict", cmd)) {
        // Check parameters
        if (nlhs != 1)
            mexErrMsgTxt("New: One output expected.");
        // Call the method
        const mxArray *TestSet=prhs[2];
        mwSize m=mxGetM(TestSet);// row number=featureNumber
        mwSize n=mxGetN(TestSet);//column number=testing data number
        double* TestSetPr=(double *)mxGetPr(TestSet);
        
        plhs[0] = mxCreateDoubleMatrix(n,1,mxREAL);
        double * prediction = (double *)mxGetPr(plhs[0]);
        rf_instance->Predict(TestSetPr,n,m,prediction);
        return;
    }
    
    if (!strcmp("ConvertTreeToList", cmd)) {
        // Check parameters
        if (nlhs != 4)
            mexErrMsgTxt("New: four output expected.");
        // Call the method
//         const mxArray *TestSet=prhs[2];
//         mwSize m=mxGetM(TestSet);// row number=featureNumber
//         mwSize n=mxGetN(TestSet);//column number=testing data number
//         double* TestSetPr=(double *)mxGetPr(TestSet);
        int treeNumber=rf_instance->treeNumber;
        int depth=rf_instance->maxDepth+1;
        int NodeNumber=pow(2,depth);
        plhs[0]=mxCreateNumericMatrix(NodeNumber, treeNumber,mxINT32_CLASS,mxREAL);
        plhs[1]=mxCreateNumericMatrix(NodeNumber, treeNumber,mxINT32_CLASS,mxREAL);
        plhs[2]=mxCreateNumericMatrix(NodeNumber, treeNumber,mxINT32_CLASS,mxREAL);
        plhs[3]=mxCreateDoubleMatrix(NodeNumber, treeNumber,mxREAL);
        int * left=(int *)mxGetPr(plhs[0]);
        int * right=(int *)mxGetPr(plhs[1]);
        int * splitFeature=(int *)mxGetPr(plhs[2]);
        double * splitValue = (double *)mxGetPr(plhs[3]);
        rf_instance->ConvertTreeToList(left,right,splitFeature,splitValue,NodeNumber);
//         
//         
//         plhs[0] = mxCreateDoubleMatrix(n,1,mxREAL);
//         double * prediction = (double *)mxGetPr(plhs[0]);
//         rf_instance->Predict(TestSetPr,n,m,prediction);
        return;
    }
//     if (!strcmp("GetMaxNodeOnTree", cmd)) {
//         plhs[0]=mxCreateNumericMatrix(1,1,mxINT32_CLASS,mxREAL);
//         int* tempPr=(int *)mxGetPr(plhs[0]);
//         *tempPr=rf_instance->max_node_on_tree;
//         return;
//     }    
//     if (!strcmp("GetLeftList", cmd)) {
//         int treeNumber=rf_instance->treeNumber;
//         int max_node_on_tree=rf_instance->max_node_on_tree;
//         int arrayL=treeNumber*max_node_on_tree;
//         plhs[0]=mxCreateNumericMatrix(arrayL,1,mxINT32_CLASS,mxREAL);
//         int* leftListPr=(int *)mxGetPr(plhs[0]);
//         for(int i=0;i<arrayL;i++)
//         {
//             leftListPr[i]=rf_instance->leftList[i];
//         }
//         return;
//     } 
//     if (!strcmp("GetRightList", cmd)) {
//         int treeNumber=rf_instance->treeNumber;
//         int max_node_on_tree=rf_instance->max_node_on_tree;
//         int arrayL=treeNumber*max_node_on_tree;
//         plhs[0]=mxCreateNumericMatrix(arrayL,1,mxINT32_CLASS,mxREAL);
//         int* RightListPr=(int *)mxGetPr(plhs[0]);
//         for(int i=0;i<arrayL;i++)
//         {
//             RightListPr[i]=rf_instance->rightList[i];
//         }
//         return;
//     } 
//     if (!strcmp("GetSplitFeatureList", cmd)) {
//         int treeNumber=rf_instance->treeNumber;
//         int max_node_on_tree=rf_instance->max_node_on_tree;
//         int arrayL=treeNumber*max_node_on_tree;
//         plhs[0]=mxCreateNumericMatrix(arrayL,1,mxINT32_CLASS,mxREAL);
//         int* splitFeatureListPr=(int *)mxGetPr(plhs[0]);
//         for(int i=0;i<arrayL;i++)
//         {
//             splitFeatureListPr[i]=rf_instance->splitFeatureList[i];
//         }
//         return;
//     }
//     if (!strcmp("GetSplitValueList", cmd)) {
//         int treeNumber=rf_instance->treeNumber;
//         int max_node_on_tree=rf_instance->max_node_on_tree;
//         int arrayL=treeNumber*max_node_on_tree;
//         plhs[0]=mxCreateDoubleMatrix(arrayL,1,mxREAL);
//         double* splitValueListPr=(double *)mxGetPr(plhs[0]);
//         for(int i=0;i<arrayL;i++)
//         {
//             splitValueListPr[i]=rf_instance->splitValueList[i];
//         }
//         return;
//     }
    // Got here, so command not recognized
    mexErrMsgTxt("Command not recognized.");
}
