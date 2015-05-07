clear;
r=3;
I=imread('rice.png');
[H, W]=size(I);
gpuIin=gpuArray(I);
gpuIout=gpuArray(uint8(zeros(H,W)));
k = parallel.gpu.CUDAKernel('meanFilter.ptx','meanFilter.cu','meanFilterKernel');
k.GridSize=[ceil(H/32),ceil(W/32),1];
k.ThreadBlockSize = [32,32,1];
tic;
gpuResult = feval(k,gpuIin,gpuIout,H,W,r);
t0=toc;
Result=gather(gpuResult);

Result1=uint8(zeros(H,W));
tic;
regionArea=(2*r+1)^2;
for i=1:H
    for j=1:W
        if(i<=r || i>H-r || j<=r || j>W-r)
            Result1(i,j)=I(i,j);
        else
            sum=double(0);
            for di=-r:1:r
                for dj=-r:1:r
                    sum=sum+double(I(i+di,j+dj));
                end
            end
            aver=uint8(sum/regionArea);
            Result1(i,j)=aver;
        end
    end
end
t1=toc;
subplot(1,3,1);
imshow(I);
subplot(1,3,2);
imshow(Result);
subplot(1,3,3);
imshow(Result1);
speedup=t1/t0
