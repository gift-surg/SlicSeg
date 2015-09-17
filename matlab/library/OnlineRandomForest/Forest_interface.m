%CLASS_INTERFACE Example MATLAB class wrapper to an underlying C++ class
classdef Forest_interface < handle
    properties (SetAccess = private, Hidden = true)
        objectHandle; % Handle to the underlying C++ class instance
        treeNum;% treeNumber
    end
    methods
        %% Constructor - Create a new C++ class instance 
        function this = Forest_interface(varargin)
            this.objectHandle =Forest_interface_mex('new', varargin{:});
        end
        
        %% Destructor - Destroy the C++ class instance
        function delete(this)
            Forest_interface_mex('delete', this.objectHandle);
        end
        function varargout = Init(this, varargin)
            this.treeNum=varargin{1};
            [varargout{1:nargout}] = Forest_interface_mex('Init', this.objectHandle, varargin{:});
        end
        %% Train - an example class method call
        function varargout = Train(this, varargin)
            [varargout{1:nargout}] = Forest_interface_mex('Train', this.objectHandle, varargin{:});
        end
        %% Test - another example class method call
        function varargout = Predict(this, varargin)
            [varargout{1:nargout}] = Forest_interface_mex('Predict', this.objectHandle, varargin{:});
        end
        
        function varargout = GPUPredict(this,varargin)
            testData=varargin{1};
            [Nf,Nte]=size(testData);

            [left,right,splitFeature,splitValue]=this.ConvertTreeToList();
            maxnode=size(left,1);
%             right=this.GetRightList();
%             maxnode=this.GetMaxNodeOnTree();
%             splitFeature=this.GetSplitFeatureList();
%             splitValue=this.GetSplitValueList();
            gpuLeft=gpuArray(left);
            gpuRight=gpuArray(right);
            gpuSplitF=gpuArray(splitFeature);
            gpuSplitV=gpuArray(splitValue);
            gpuTestData=gpuArray(testData);
            gpuPredict=gpuArray(zeros(Nte,1));
            
            k = parallel.gpu.CUDAKernel('ForestPredict.ptx','ForestPredict.cu','ForestPredict');
            k.GridSize=[ceil(Nte/32),1,1];
            k.ThreadBlockSize = [32,1,1];
            gpuPredict=feval(k,gpuLeft,gpuRight,gpuSplitF,gpuSplitV,this.treeNum,maxnode,...
                gpuTestData,Nte,Nf,gpuPredict);
            [varargout{1:nargout}]=gather(gpuPredict);
        end
%         function varargout = GetMaxNodeOnTree(this)
%             [varargout{1:nargout}] = Forest_interface_mex('GetMaxNodeOnTree', this.objectHandle);
%         end
%         function varargout = GetLeftList(this)
%             [varargout{1:nargout}] = Forest_interface_mex('GetLeftList', this.objectHandle);
%         end
%         function varargout = GetRightList(this)
%             [varargout{1:nargout}] = Forest_interface_mex('GetRightList', this.objectHandle);
%         end
%         function varargout = GetSplitFeatureList(this)
%             [varargout{1:nargout}] = Forest_interface_mex('GetSplitFeatureList', this.objectHandle);
%         end
%         function varargout = GetSplitValueList(this)
%             [varargout{1:nargout}] = Forest_interface_mex('GetSplitValueList', this.objectHandle);
%         end
        function varargout = ConvertTreeToList(this)
            [varargout{1:nargout}] = Forest_interface_mex('ConvertTreeToList', this.objectHandle);
        end
    end
end