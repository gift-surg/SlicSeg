function CompileSlicSeg(varargin)

    % Creating the CoreMex object will trigger compilation of mex and cuda files
    coreMexHandle = CoreMex(SlicSegGetMexFilesToCompile, CoreReporting(CoreProgressDialog, false, fullfile(CoreDiskUtilities.GetUserDirectory, 'slicseg.log')));
    
    if nargin > 0 && strcmp(varargin{1}, 'recompile')
        coreMexHandle.Recompile;
    end
end