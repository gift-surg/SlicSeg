classdef ForestWrapper < handle
    % ForestWrapper A wrapper around a C++ implementation of a random forest
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
        
        function delete(obj)
            % Destructor - destroy the C++ object instance when the
            % Matlab object is destroyed
            Forest_interface_mex('delete', obj.forestHandle);
        end

        function varargout = Init(obj, varargin)
            % Initialise the random forest
            obj.treeNum=varargin{1};
            [varargout{1:nargout}] = Forest_interface_mex('Init', obj.forest, varargin{:});
        end

        function varargout = Train(obj, varargin)
            % Train the random forest
            [varargout{1:nargout}] = Forest_interface_mex('Train', obj.forestHandle, varargin{:});
        end

        function varargout = Predict(obj, varargin)
            % Predict values from the random forest
            [varargout{1:nargout}] = Forest_interface_mex('Predict', obj.forestHandle, varargin{:});
        end
        
        function varargout = GPUPredict(obj, varargin)
            testData=varargin{1};
            [Nf,Nte]=size(testData);

            [left,right,splitFeature,splitValue]=obj.ConvertTreeToList();
            maxnode=size(left,1);
            gpuLeft=gpuArray(left);
            gpuRight=gpuArray(right);
            gpuSplitF=gpuArray(splitFeature);
            gpuSplitV=gpuArray(splitValue);
            gpuTestData=gpuArray(testData);
            gpuPredict=gpuArray(zeros(Nte,1));
            
            k = parallel.gpu.CUDAKernel('ForestPredict.ptx','ForestPredict.cu','ForestPredict');
            k.GridSize=[ceil(Nte/32),1,1];
            k.ThreadBlockSize = [32,1,1];
            gpuPredict=feval(k,gpuLeft,gpuRight,gpuSplitF,gpuSplitV,obj.treeNum,maxnode,...
                gpuTestData,Nte,Nf,gpuPredict);
            [varargout{1:nargout}]=gather(gpuPredict);
        end

        function varargout = ConvertTreeToList(obj)
            [varargout{1:nargout}] = Forest_interface_mex('ConvertTreeToList', obj.forestHandle);
        end
    end
end