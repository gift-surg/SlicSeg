function intensityFeature = image2IntensityFeature(I)
    % image2IntensityFeature Constructs features based on an input image
    %
    % Author: Guotai Wang
    % Copyright (c) 2014-2016 University College London, United Kingdom. All rights reserved.
    % http://cmictig.cs.ucl.ac.uk
    %
    % Distributed under the BSD-3 licence. Please see the file licence.txt 
    % This software is not certified for clinical use.
    
    [H, W] = size(I);
    gpuIin = gpuArray(I);
    gpuIoutMean = gpuArray(zeros(H, W));
    gpuIoutVar = gpuArray(zeros(H, W));
    k = parallel.gpu.CUDAKernel('intensityFeature.ptx', 'intensityFeature.cu', 'intensityFeature');
    k.GridSize = [ceil(H/32), ceil(W/32), 1];
    k.ThreadBlockSize = [32, 32, 1];
    r = 2;
    [outMean, outVar] = feval(k, gpuIin, gpuIoutMean, gpuIoutVar, H, W, r);
    MeanI = gather(outMean);
    VarI = gather(outVar);
    meanVector = reshape(MeanI, H*W, 1);
    varVector = reshape(VarI, H*W, 1);
    meanVector2 = meanVector.*meanVector;
    meanVector3 = meanVector2.*meanVector;
    meanVector4 = exp(meanVector);
    intensityFeature = [meanVector meanVector2 meanVector3 meanVector4 varVector];
    meanInten = repmat(mean(intensityFeature), H*W, 1);
    stdInten = repmat(sqrt(var(intensityFeature)), H*W, 1);
    intensityFeature = (intensityFeature-meanInten)./stdInten;
    
    clear gpuIin gpuIoutMean gpuIoutVar outMean outVar
end