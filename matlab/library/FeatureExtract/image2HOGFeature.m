function HOGFeature = image2HOGFeature(I)
    % image2HOGFeature Constructs a HOG feature based on an input image
    %
    % Author: Guotai Wang
    % Copyright (c) 2015-2016 University College London, United Kingdom. All rights reserved.
    % Distributed under the BSD-3 licence. Please see the file licence.txt 
    %
    
    [H, W] = size(I);
    gpuIin = gpuArray(I);
    gpuInitGmag = gpuArray(zeros(H, W));
    gpuInitGori = gpuArray(zeros(H, W));
    k1 = parallel.gpu.CUDAKernel('imageGradient.ptx', 'imageGradient.cu', 'imageGradient');
    k1.GridSize = [ceil(H/32), ceil(W/32), 1];
    k1.ThreadBlockSize = [32, 32, 1];
    [gpuGmag, gpuGori] = feval(k1, gpuIin, gpuInitGmag, gpuInitGori, H, W);
    Gmag = gather(gpuGmag);

    k2 = parallel.gpu.CUDAKernel('imageHoG.ptx', 'imageHoG.cu', 'imageHoG');
    k2.GridSize = [ceil(H/32), ceil(W/32), 1];
    k2.ThreadBlockSize = [32, 32, 1];
    bins = 8;
    r = 4;
    HoGInit = zeros(H*W, bins);
    gpuInitHOG = gpuArray(HoGInit);
    [gpuHOG] = feval(k2, gpuGmag, gpuGori, gpuInitHOG, H, W, bins, r);
    HOGI = gather(gpuHOG);

    GmagVector = reshape(Gmag, H*W, 1);
    GmagVector1 = sqrt(GmagVector);
    GmagVector2 = GmagVector.*GmagVector;
    GmagVector3 = GmagVector2.*GmagVector;

    HOGFeature = [GmagVector GmagVector1 GmagVector2 GmagVector3 HOGI];
    meanInten = repmat(mean(HOGFeature), H*W, 1);
    stdInten = repmat(sqrt(var(HOGFeature)), H*W, 1);
    HOGFeature = (HOGFeature-meanInten)./stdInten;

    clear gpuIin gpuInitGmag gpuInitGori gpuGmag gpuGori gpuGmag gpuInitHOG gpuHOG
end