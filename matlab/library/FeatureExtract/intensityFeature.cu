__global__ void intensityFeature(const unsigned char * inputData,double *outputMean,double *outputvar,const int H,const int W,int r)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int kernel_radius=r;
    int kernel_size=2*kernel_radius+1;
    if(x<kernel_radius || x>=H-kernel_radius ||
       y<kernel_radius || y>=W-kernel_radius)
    {
        char temp_value=*(inputData+x+y*H);
        *(outputMean+x+y*H)=temp_value;
        *(outputvar+x+y*H)=0;
    }
    else
    {
        double sum=0;
        double square_sum=0;
        for(int i=-kernel_radius;i<=kernel_radius;i++)
        {
            for(int j=-kernel_radius;j<=kernel_radius;j++)
            {
                double tempValue=*(inputData+x+i+(y+j)*H);
                sum += tempValue;
                square_sum+=tempValue*tempValue;
            }
        }
        double mean=sum/(kernel_size*kernel_size);
        double var=square_sum/(kernel_size*kernel_size)-mean*mean;
        var=sqrt(var);
        *(outputMean+x+y*H)=mean;
        *(outputvar+x+y*H)=var;
    }
}