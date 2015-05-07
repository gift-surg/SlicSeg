
clear;
r=2;
I=imread('37.png');
[H, W]=size(I);
gpuIin=gpuArray(I);
gpuIoutMean=gpuArray(zeros(H,W));
gpuIoutVar=gpuArray(zeros(H,W));
k = parallel.gpu.CUDAKernel('imageGradient.ptx','imageGradient.cu','imageGradient');
k.GridSize=[ceil(H/32),ceil(W/32),1];
k.ThreadBlockSize = [32,32,1];
tic;
[outMean,outVar]=feval(k,gpuIin,gpuIoutMean,gpuIoutVar,H,W);
toc;
MeanI=gather(outMean);
VarI=gather(outVar);
imshow(uint8(MeanI));
figure;
imshow(uint8((outVar+3.15/2)*100));

k2 = parallel.gpu.CUDAKernel('imageHoG.ptx','imageHoG.cu','imageHoG');
k2.GridSize=[ceil(H/32),ceil(W/32),1];
k2.ThreadBlockSize = [32,32,1];
bins=10;
HoGinit=zeros(H*W,bins);
gpuHOG=gpuArray(HoGinit);
[outHoG]=feval(k2,outMean,outVar,gpuHOG,H,W,bins,4);
HOGI=gather(outHoG);