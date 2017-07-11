#include <stdio.h>-	

#define RADIUS        3
#define BLOCK_SIZE    256
#define NUM_ELEMENTS  (4096*2)

// CUDA API error checking macro
#define cudaCheck(error) \
  if (error != cudaSuccess) { \
    printf("Fatal error: %s at %s:%d\n", \
      cudaGetErrorString(error), \
      __FILE__, __LINE__); \
    exit(1); \
  }


__global__ void stencil_1d(int *in, int *out) 
{
   //Add your code for the kernel
  __shared__ int temp[BLOCK_SIZE + 2* RADIUS]; //shared memory
  int gindex = threadIdx.x + blockIdx.x * blockDim.x+RADIUS;
  int lindex = threadIdx.x +RADIUS;

  //READ input elements into shared mempory:mapping
  temp[lindex] = in[gindex];

  
  //printf("temp[%d]=%d\n",lindex,temp[lindex]);
  //由每个block的前面RADIUS个thread来读取halo数据
  if(threadIdx.x < RADIUS){
    temp[lindex - RADIUS] = in[gindex - RADIUS]; //读取前面的halo
    temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];//后面的
  }
  
  //Synchronize(ensure all data is available)
    __syncthreads();

  //Apply the stencil
  int result = 0;
  for(int offset = -RADIUS; offset <= RADIUS;offset++)
    result += temp[lindex+offset];
  //printf("result:%d",result);
    
  
  //Store the result
  out[gindex-RADIUS] = result;
  

}

int main()
{
  unsigned int i;
  int h_in[NUM_ELEMENTS + 2 * RADIUS], h_out[NUM_ELEMENTS]; //cpu memory
  int size = NUM_ELEMENTS + 2 * RADIUS; //cpu memory数组长度
  int *d_in, *d_out;//global memory

  // Initialize host data as all 1. Add your code
  //int *h_in =(int*)malloc(sizeof(int)*(NUM_ELEMENTS+RADIUS));
  for (unsigned int i=0; i < size; i++) { h_in[i] = 1;}

  

  // Allocate space on the device for d_in and d_out. Add your code
  cudaMalloc((void**)&d_in,size*sizeof(int));
  cudaMalloc((void**)&d_out,NUM_ELEMENTS*sizeof(int)); //???last edit here




  // Copy host input data to device. Add your code
  cudaMemcpy(d_in, h_in, size * sizeof(int),cudaMemcpyHostToDevice); 

  // Invoke the stencil_id kernel by defining the right grid and block dimension. You may need a boundary block. Add your code
  
  //NUM_ELEMENT = 4096*2; BLOCK_SIZE = 256; -> NUM_ELEMENT/BOLCK_SIZE =32;(正好整除) but we have halo, so we should have 33 BLOCKS
  dim3 DimGrid(ceil((NUM_ELEMENTS+2*RADIUS)/BLOCK_SIZE),1 ,1);
  dim3 DimBlock(BLOCK_SIZE, 1, 1);  //此处也应当+2*RADIUS吧。否，blocksize是block内线程的数量，而不是block内存储空间的大小
  stencil_1d<<<DimGrid, DimBlock>>>(d_in,d_out);
  

  


  // Copy device output data to host. Add your code
  cudaMemcpy(h_out, d_out,NUM_ELEMENTS*sizeof(int),cudaMemcpyDeviceToHost);

 

  // Verify every out value is 7. Already given
  for( i = 0; i < NUM_ELEMENTS; ++i )
    if (h_out[i] != 7)
    {
      printf("Element h_out[%d] == %d != 7\n", i, h_out[i]);
      //break;
    }


  if (i == NUM_ELEMENTS)
    printf("SUCCESS!\n");

  // Free out memory. Add your code. 
  //free(h_in);
  //free(h_out);
  cudaFree(d_in);
  cudaFree(d_out);



  return 0;
}

