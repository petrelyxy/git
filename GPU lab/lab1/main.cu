/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include "support.h"
#include "kernel.cu"

__host__
void vecAdd(float *A_h, float *B_h, float *C_h, int n){
      //d_h, d_B, d_C allocations and copies omitted
      //Run ceil(n/256.0) blocks of 256 threads each
      dim3 DimGrid(ceil(n/256.0), 1, 1); //定义grid维度和大小, 1表示不使用
      dim3 DimBlock(256, 1, 1); //同上，一维block，每一个大小256
      //vecAddKernel<<<DimGrid, DimBlock>>>(d_A, d_B, d_C, n); //d_A,d_B,d_C is not defined
}
    

int main(int argc, char**argv) {

    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    unsigned int n;
    if(argc == 1) {
        n = 10000;
    } else if(argc == 2) {
        n = atoi(argv[1]);
    } else {
        printf("\n    Invalid input parameters!"
           "\n    Usage: ./vecadd               # Vector of size 10,000 is used"
           "\n    Usage: ./vecadd <m>           # Vector of size m is used"
           "\n");
        exit(0);
    }
    
    //初始化host中A，B,C.A，B赋予随机值
    float* A_h = (float*) malloc( sizeof(float)*n );
    for (unsigned int i=0; i < n; i++) { A_h[i] = (rand()%100)/100.00; }

    float* B_h = (float*) malloc( sizeof(float)*n );
    for (unsigned int i=0; i < n; i++) { B_h[i] = (rand()%100)/100.00; }

    float* C_h = (float*) malloc( sizeof(float)*n );

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    Vector size = %u\n", n);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE ALLOCATE DEVICE MEMORY A B AND C
    

    //@param *h_A,*h_B:加数 *h_C:result, n:n维向量
    //void vecAdd(float *A_h, float *B_h, float *C_h, int n){
        int size = n * sizeof(float);
        float *d_A, *d_B, *d_C;

        //在gpu上分配内存,单位为byte,利用copy赋值
        cudaMalloc((void **)&d_A, size);
        //cudaMemcpy(d_A, A_h, size,cudaMemcpyHostToDevice);//利用copy赋值
        cudaMalloc((void **)&d_B, size);
        //cudaMemcpy(d_B,B_h,size,cudaMemcpyHostToDevice);
        cudaMalloc((void **)&d_C, size);

        //kernel invocation code- to be shown later
        //cudaMemcpy(C_h, d_C,size, cudaMemcpyDeviceToHost);//完成计算后将值拷回;计算过程在哪？在kernel,cu
       // cudaFree(d_A);cudaFree(d_B);cudaFree(d_C);
    //}
    






    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device FOR A AND B ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE
    cudaMemcpy(d_A, A_h, size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B_h,size,cudaMemcpyHostToDevice);
   



    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel ----------------------------------------------------------

    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE CONFIGURE THE GRID AND LAUNCH IT ,vecAddKernel is defined in kernel.cu
    //vecAdd(A_h, B_h, C_h, n);
     dim3 DimGrid(ceil(n/256.0), 1, 1); //定义grid维度和大小, 1表示不使用
     dim3 DimBlock(256, 1, 1); //同上，一维block，每一个大小256
     vecAddKernel<<<DimGrid, DimBlock>>>(d_A, d_B, d_C, n); 






    cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host FOR C ----------------------------------------

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE
    cudaMemcpy(C_h, d_C,size, cudaMemcpyDeviceToHost);




    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    verify(A_h, B_h, C_h, n);

    // Free memory ------------------------------------------------------------

    free(A_h);
    free(B_h);
    free(C_h);

    //INSERT CODE HERE
    cudaFree(d_A);cudaFree(d_B);cudaFree(d_C);



    return 0;

}


