function mex_files_to_compile_map = SlicSegGetMexFilesToCompile(~)
    % SlicSegGetMexFilesToCompile Returns a list of mex files used by Slic-Seg
    %
    % Author: Tom Doel
    % Copyright (c) 2014-2016 University College London, United Kingdom. All rights reserved.
    % http://cmictig.cs.ucl.ac.uk
    %
    % Distributed under the BSD-3 licence. Please see the file licence.txt 
    % This software is not certified for clinical use.
    %
    
    % Populate list with known files
    files_to_compile = {};
    [mex_dir, ~, ~] = fileparts(mfilename('fullpath'));
    
    % wgtmaxflow    
    files_to_compile{end + 1} = CoreMexInfo(2, 'wgtmaxflowmex', 'cpp', fullfile(mex_dir, 'library', 'wgtmaxflow'), [], ...
        {fullfile(mex_dir, 'library', 'maxflow', 'maxflow-v3.0', 'graph.cpp'), fullfile(mex_dir, 'library', 'maxflow', 'maxflow-v3.0', 'maxflow.cpp')} ...
    );

    % Online random forest
    files_to_compile{end + 1} = CoreMexInfo(1, 'Forest_interface_mex', 'cpp', fullfile(mex_dir, 'library', 'OnlineRandomForest'), [], ...
        {fullfile(mex_dir, 'library', 'OnlineRandomForest', 'ORForest.cpp'), fullfile(mex_dir, 'library', 'OnlineRandomForest', 'ODTree.cpp')} ...
    );

    % Cuda files - dwt
    files_to_compile{end + 1} = CoreCudaInfo('nvcc', 1, 'wgtDWTConvolution', 'cu', fullfile(mex_dir, 'library', 'dwt'), '-ptx', []);
    files_to_compile{end + 1} = CoreCudaInfo('nvcc', 1, 'wgtDWTMeanStd', 'cu', fullfile(mex_dir, 'library', 'dwt'), '-ptx', []);
    files_to_compile{end + 1} = CoreCudaInfo('nvcc', 1, 'wgtDWTFeature', 'cu', fullfile(mex_dir, 'library', 'dwt'), '-ptx', []);

    % Cuda files - FeatureExtract
    files_to_compile{end + 1} = CoreCudaInfo('nvcc', 1, 'imageGradient', 'cu', fullfile(mex_dir, 'library', 'FeatureExtract'), '-ptx', []);
    files_to_compile{end + 1} = CoreCudaInfo('nvcc', 1, 'imageHoG', 'cu', fullfile(mex_dir, 'library', 'FeatureExtract'), '-ptx', []);
    files_to_compile{end + 1} = CoreCudaInfo('nvcc', 1, 'intensityFeature', 'cu', fullfile(mex_dir, 'library', 'FeatureExtract'), '-ptx', []);
    
    % Transfer to a map
    mex_files_to_compile_map = containers.Map;
    for mex_file = files_to_compile
        mex_files_to_compile_map(mex_file{1}.Name) = mex_file{1};
    end
    
end