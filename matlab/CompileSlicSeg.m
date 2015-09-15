function CompileSlicSeg

    % Creating the CoreMex object will trigger compilation of mex and cuda files
    CoreMex(SlicSegGetMexFilesToCompile, CoreReporting(CoreProgressDialog, false, fullfile(CoreDiskUtilities.GetUserDirectory, 'slicseg.log')));
end