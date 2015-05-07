__global__ void add2( double * v1, const double * v2 )
{
 int idx = threadIdx.x;
 v1[idx] -= v2[idx-1];
}