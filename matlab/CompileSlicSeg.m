function CompileSlicSeg(varargin)
    % CompileSlicSeg Compiles Slic-Seg mex and cuda files
    %
    % Author: Tom Doel
    % Copyright (c) 2014-2016 University College London, United Kingdom. All rights reserved.
    % http://cmictig.cs.ucl.ac.uk
    %
    % Distributed under the BSD-3 licence. Please see the file licence.txt 
    % This software is not certified for clinical use.
    % Creating the CoreMex object will trigger compilation of mex and cuda files
    
    coreMexHandle = CoreMex(SlicSegGetMexFilesToCompile, CoreReporting(CoreProgressDialog, false, fullfile(CoreDiskUtilities.GetUserDirectory, 'slicseg.log')));
    
    if nargin > 0 && strcmp(varargin{1}, 'recompile')
        coreMexHandle.Recompile;
    end
end