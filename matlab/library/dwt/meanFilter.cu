__global__ void meanFilterKernel(const unsigned char * inputData,unsigned char *outputData,const int H,const int W,int r)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int kernel_radius=r;
    int kernel_size=2*kernel_radius+1;
    if(x<kernel_radius || x>=H-kernel_radius ||
       y<kernel_radius || y>=W-kernel_radius)
    {
        char temp_value=*(inputData+x+y*H);
        *(outputData+x+y*H)=temp_value;
    }
    else
    {
        double sum=0;
        for(int i=-kernel_radius;i<=kernel_radius;i++)
        {
            for(int j=-kernel_radius;j<=kernel_radius;j++)
            {
                sum += *(inputData+x+i+(y+j)*H);
            }
        }
        double mean=sum/(kernel_size*kernel_size);
        *(outputData+x+y*H)=mean;
    }
}