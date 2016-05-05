classdef Forest_interface < handle
    % Forest_interface A wrapper around a C++ implementation of a random forest
    %
    % For more information see Slic-Seg (Wang et al, 2016)
    %
    % Author: Guotai Wang
    % Copyright (c) 2015-2016 University College London, United Kingdom. All rights reserved.
    % Distributed under the BSD-3 licence. Please see the file licence.txt 
    %
    % This software is not certified for clinical use.
    %
    
    properties (SetAccess = private, Hidden = true)
        forestHandle  % Handle to the underlying C++ class instance
        treeNum       % treeNumber
    end
    
    methods
        function this = Forest_interface(varargin)
            % Create a new instance of the C++ class
            this.forestHandle = Forest_interface_mex('new', varargin{:});
        end
        
        function delete(this)
            % Destructor - destroy the C++ object instance when the
            % Matlab object is destroyed
            Forest_interface_mex('delete', this.forest);
        end

        function varargout = Init(this, varargin)
            % Initialise the random forest
            this.treeNum=varargin{1};
            [varargout{1:nargout}] = Forest_interface_mex('Init', this.forest, varargin{:});
        end

        function varargout = Train(this, varargin)
            % Train the random forest
            [varargout{1:nargout}] = Forest_interface_mex('Train', this.forestHandle, varargin{:});
        end

        function varargout = Predict(this, varargin)
            % Predict values from the random forest
            [varargout{1:nargout}] = Forest_interface_mex('Predict', this.forestHandle, varargin{:});
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
%             [varargout{1:nargout}] = Forest_interface_mex('GetMaxNodeOnTree', this.forestHandle);
%         end
%         function varargout = GetLeftList(this)
%             [varargout{1:nargout}] = Forest_interface_mex('GetLeftList', this.forestHandle);
%         end
%         function varargout = GetRightList(this)
%             [varargout{1:nargout}] = Forest_interface_mex('GetRightList', this.forestHandle);
%         end
%         function varargout = GetSplitFeatureList(this)
%             [varargout{1:nargout}] = Forest_interface_mex('GetSplitFeatureList', this.forestHandle);
%         end
%         function varargout = GetSplitValueList(this)
%             [varargout{1:nargout}] = Forest_interface_mex('GetSplitValueList', this.forestHandle);
%         end
        function varargout = ConvertTreeToList(this)
            [varargout{1:nargout}] = Forest_interface_mex('ConvertTreeToList', this.forestHandle);
        end
    end
end