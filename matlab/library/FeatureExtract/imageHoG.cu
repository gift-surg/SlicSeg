__device__ double getPixel(const double *array,int H,int W,int i,int j)
{
    if(i<0 || i>=H || j<0 ||j>=W)
    {
        return 0;
    }
    else
    {
        return *(array+H*j+i);
    }
}
__device__ void setPixel(double *array,int H,int W,int i,int j,double value)
{
    if(i<0 || i>=H || j<0 ||j>=W)
    {
        return;
    }
    *(array+H*j+i)=value;
}
__global__ void imageHoG(const double * g_mag,const double * g_ori,double *hog,const int H,const int W,const int bins,const int r)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    const double pi=3.1416;
    int imageLen=H*W;
    double bin_len=2*pi/bins;
    for(int i=-r;i<=r;i++)
    {
        for(int j=-r;j<=r;j++)
        {
            double mag=getPixel(g_mag,H,W,x+i,y+j);
            double ori=getPixel(g_ori,H,W,x+i,y+j);
            int bin_n=floor((ori+pi)/bin_len);
            double dis=(double)(i*i+j*j)/(r*r);
            double dvalue=mag*(exp(-dis));
            double newValue=dvalue+getPixel(hog,imageLen,bins,H*y+x,bin_n);
            setPixel(hog,imageLen,bins,H*y+x,bin_n,newValue);
        }
    }
    double sum=0;
    for(int i=0;i<bins;i++)
    {
        sum=sum+getPixel(hog,imageLen,bins,H*y+x,i);
    }
    if(sum!=0)
    {
        for(int i=0;i<bins;i++)
        {
            double newValue=getPixel(hog,imageLen,bins,H*y+x,i)/sum;
            setPixel(hog,imageLen,bins,H*y+x,i,newValue);
        }
    }
}