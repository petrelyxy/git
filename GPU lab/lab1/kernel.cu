/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

//compute vector sum c = A + B;
// Each thread performs one pair-wise addtion.

__global__ void vecAddKernel(float* A, float* B, float* C, int n) {

    // Calculate global thread index based on the block and thread indices ----

    //INSERT KERNEL CODE HERE
    int i = threadIdx.x + blockDim.x * blockIdx.x;
      
    

    // Use global index to determine which elements to read, add, and write ---

    //INSERT KERNEL CODE HERE
    if(i < n) C[i] = A[i] + B[i];
   

}


