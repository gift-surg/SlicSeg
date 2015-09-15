
clear;
r=2;
I=imread('rice.png');
[H, W]=size(I);
gpuIin=gpuArray(I);
gpuIoutMean=gpuArray(zeros(H,W));
gpuIoutVar=gpuArray(zeros(H,W));
k = parallel.gpu.CUDAKernel('intensityFeature.ptx','intensityFeature.cu','intensityFeature');
k.GridSize=[ceil(H/32),ceil(W/32),1];
k.ThreadBlockSize = [32,32,1];
tic;
[outMean,outVar]=feval(k,gpuIin,gpuIoutMean,gpuIoutVar,H,W,r);
toc;
MeanI=gather(outMean);
VarI=gather(outVar);
imshow(uint8(MeanI));
figure;
imshow(uint8(outVar));
