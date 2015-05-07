clear;
k = parallel.gpu.CUDAKernel('addvector.ptx','addvector.cu','add2');
N = 128;
k.ThreadBlockSize = N;
gpuIn1 = ones(N,1,'gpuArray');
gpuIn2 = ones(N,1,'gpuArray');
for i=1:N
    gpuIn1(i)=i;
    gpuIn1(i)=i;
end
gpuResult = feval(k,gpuIn1,gpuIn2);
Result=gather(gpuResult);