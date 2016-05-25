% CIBuildAndTest Script to be run from a continuous integration (CI) server
%
% Author: Tom Doel
% Distributed under the BSD-3 licence. Please see the file licence.txt 
% This software is not certified for clinical use.
%

SlicSegAddPaths;
CompileSlicSeg recompile
test;
