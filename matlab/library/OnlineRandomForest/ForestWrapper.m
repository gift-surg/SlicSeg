classdef ForestWrapper < handle
    % ForestWrapper A wrapper around a C++ implementation of a random forest
    %
    % For more information see Slic-Seg (Wang et al, 2016)
    %
    % This mex interface is partly inspired by the Example MATLAB class wrapper by Oliver Woodford (see library/C___class_interface)
    %
    % Author: Guotai Wang
    % Copyright (c) 2014-2016 University College London, United Kingdom. All rights reserved.
    % http://cmictig.cs.ucl.ac.uk
    %
    % Distributed under the BSD-3 licence. Please see the file licence.txt 
    % This software is not certified for clinical use.
    %
    
    properties (SetAccess = private, Hidden = true)
        forestHandle  % Handle to the underlying C++ class instance
    end
    
    methods
        function obj = ForestWrapper(varargin)
            % Create a new instance of the C++ class
            obj.forestHandle = Forest_interface_mex('new', varargin{:});
        end
        
        function delete(obj)
            % Destructor - destroy the C++ object instance when the
            % Matlab object is destroyed
            Forest_interface_mex('delete', obj.forestHandle);
        end

        function varargout = Init(obj, varargin)
            % Initialise the random forest
            [varargout{1:nargout}] = Forest_interface_mex('Init', obj.forestHandle, varargin{:});
        end

        function varargout = Train(obj, varargin)
            % Train the random forest
            [varargout{1:nargout}] = Forest_interface_mex('Train', obj.forestHandle, varargin{:});
        end

        function varargout = Predict(obj, varargin)
            % Predict values from the random forest
            [varargout{1:nargout}] = Forest_interface_mex('Predict', obj.forestHandle, varargin{:});
        end
        
        function varargout = ConvertTreeToList(obj)
            [varargout{1:nargout}] = Forest_interface_mex('ConvertTreeToList', obj.forestHandle);
        end
    end
end