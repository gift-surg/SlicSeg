function featureSet = image2DWTfeature(I)
    % image2DWTfeature Constructs features based on an input image
    %
    % Author: Guotai Wang
    % Copyright (c) 2014-2016 University College London, United Kingdom. All rights reserved.
    % http://cmictig.cs.ucl.ac.uk
    %
    % Distributed under the BSD-3 licence. Please see the file licence.txt 
    % This software is not certified for clinical use.

    dwt = wgtDWTGPU(I);
    [H, W] = size(dwt);
    k = parallel.gpu.CUDAKernel('wgtDWTMeanStd.ptx', 'wgtDWTMeanStd.cu', 'wgtDWTMeanStd');
    k.GridSize = [ceil(H/32), 1, 1];
    k.ThreadBlockSize = [32, 1, 1];
    gpuDWT = gpuArray(dwt);
    gpuInitMean = gpuArray(zeros(H, 7));
    gpuInitStd = gpuArray(zeros(H, 7));
    [gpuMean, gpuStd] = feval(k, gpuDWT, gpuInitMean, gpuInitStd, H, W);
    dwtMean = gather(gpuMean);
    dwtStd = gather(gpuStd);
    featureSet = [dwtMean dwtStd];
    MeanValues = repmat(mean(featureSet), H, 1);
    StdValues = repmat(sqrt(var(featureSet)), H, 1);
    featureSet = (featureSet-MeanValues)./StdValues;

    clear gpuDWT gpuInitMean gpuInitStd gpuMean gpuStd
end