function dwt=wgtDWTGPU(I)
[H,W]=size(I);
k = parallel.gpu.CUDAKernel('wgtDWTConvolution.ptx','wgtDWTConvolution.cu','wgtDWTConvolution');
k.GridSize=[ceil(H/32),ceil(W/32),1];
k.ThreadBlockSize = [32,32,1];

%level 1
gpuI=gpuArray(double(I));
gpuLow_horizon1=gpuArray(zeros(H,W));
gpuHigh_horizon1=gpuArray(zeros(H,W));
gpuLL1=gpuArray(zeros(H,W));
gpuLH1=gpuArray(zeros(H,W));
gpuHL1=gpuArray(zeros(H,W));
gpuHH1=gpuArray(zeros(H,W));

gpuLow_horizon1=feval(k,gpuI,gpuLow_horizon1,H,W,0,1,true);
gpuHigh_horizon1=feval(k,gpuI,gpuHigh_horizon1,H,W,1,1,true);
gpuLL1=feval(k,gpuLow_horizon1,gpuLL1,H,W,0,1,false);
gpuLH1=feval(k,gpuLow_horizon1,gpuLH1,H,W,1,1,false);
gpuHL1=feval(k,gpuHigh_horizon1,gpuHL1,H,W,0,1,false);
gpuHH1=feval(k,gpuHigh_horizon1,gpuHH1,H,W,1,1,false);

%     double * pDatahigh_horizon1=new double[width*height];
%     double * pDataLL1=new double[width*height];
%     double * pDataLH1=new double[width*height];
%     double * pDataHL1=new double[width*height];
%     double * pDataHH1=new double[width*height];
%     convolution(pDataIn, pDatalow_horizon1, height, width, &(kernelL[0]), 2, 1, 'h');
%     convolution(pDataIn, pDatahigh_horizon1, height, width, &(kernelH[0]), 2, 1, 'h');
%     
%     convolution(pDatalow_horizon1, pDataLL1, height, width, &(kernelL[0]), 2, 1, 'v');
%     convolution(pDatalow_horizon1, pDataLH1, height, width, &(kernelH[0]), 2, 1, 'v');
%     convolution(pDatahigh_horizon1, pDataHL1, height, width, &(kernelL[0]), 2, 1, 'v');
%     convolution(pDatahigh_horizon1, pDataHH1, height, width, &(kernelH[0]), 2, 1, 'v');
%     cout<<"level 1 converlute success"<<endl;
%level 2
gpuLow_horizon2=gpuArray(zeros(H,W));
gpuHigh_horizon2=gpuArray(zeros(H,W));
gpuLL2=gpuArray(zeros(H,W));
gpuLH2=gpuArray(zeros(H,W));
gpuHL2=gpuArray(zeros(H,W));
gpuHH2=gpuArray(zeros(H,W));

gpuLow_horizon2=feval(k,gpuI,gpuLow_horizon2,H,W,0,2,true);
gpuHigh_horizon2=feval(k,gpuI,gpuHigh_horizon2,H,W,1,2,true);
gpuLL2=feval(k,gpuLow_horizon2,gpuLL2,H,W,0,2,false);
gpuLH2=feval(k,gpuLow_horizon2,gpuLH2,H,W,1,2,false);
gpuHL2=feval(k,gpuHigh_horizon2,gpuHL2,H,W,0,2,false);
gpuHH2=feval(k,gpuHigh_horizon2,gpuHH2,H,W,1,2,false);

k = parallel.gpu.CUDAKernel('wgtDWTFeature.ptx','wgtDWTFeature.cu','wgtDWTFeature');
k.GridSize=[ceil(H/32),ceil(W/32),1];
k.ThreadBlockSize = [32,32,1];
gpuDWT=gpuArray(zeros(H*W,64));
gpuDWT=feval(k,gpuLL2,gpuLH2,gpuHL2,gpuHH2,gpuLH1,gpuHL1,gpuHH1,gpuDWT,H,W);
dwt=gather(gpuDWT);