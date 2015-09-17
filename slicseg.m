% Script to run Slic-Seg

% Sets up the Matlab path
SlicSegAddPaths;

% Compiles the necessary mex and cuda files
CompileSlicSeg;

% Runs the user interface
ImageSegUI;