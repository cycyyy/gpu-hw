/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"
#include "support.cu"



inline
cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
  }
  return result;
}

int main (int argc, char *argv[])
{
    //set standard seed
    srand(217);

    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    float *A_h, *B_h, *C_h;
    float *A_d, *B_d, *C_d;
    size_t A_sz, B_sz, C_sz;
    unsigned VecSize;
   
    dim3 dim_grid, dim_block;

    if (argc == 1) {
        VecSize = 100 * 10000 * 10;

      } else if (argc == 2) {
      VecSize = atoi(argv[1]);   
      
      
      }
  
      else {
        printf("\nOh no!\nUsage: ./vecAdd <Size>");
        exit(0);
    }

    A_sz = VecSize;
    B_sz = VecSize;
    C_sz = VecSize;
    A_h = (float*) malloc( sizeof(float)*A_sz );
    for (unsigned int i=0; i < A_sz; i++) { A_h[i] = (rand()%100)/100.00; }

    B_h = (float*) malloc( sizeof(float)*B_sz );
    for (unsigned int i=0; i < B_sz; i++) { B_h[i] = (rand()%100)/100.00; }

    C_h = (float*) malloc( sizeof(float)*C_sz );

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    size Of vector: %u x %u\n  ", VecSize);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE
    int size = sizeof(float) * VecSize;
    cudaMalloc((void **)&A_d, size);
    cudaMalloc((void **)&B_d, size);
    cudaMalloc((void **)&C_d, size);

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    int s_num = 2;
    if (!VecSize % s_num) {
    	printf("s_num error! v_s: %d, s_num: %d\n", VecSize, s_num);
        FATAL("Error input size!");
    }
    int s_size = VecSize / s_num;
    int s_byte = sizeof(float) * s_size;
    int block_size = 256;
    int grid_size = (int)ceil((float)s_size/block_size);
    cudaStream_t streams[s_num];
    for (int i=0;i<s_num;i++) {
	checkCuda(cudaStreamCreate(&streams[i]));
    }
    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE
    //cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    //cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
    //cudaDeviceSynchronize();
    for (int i=0;i<s_num;i++) {
        checkCuda(cudaMemcpyAsync(A_d + i*s_size, A_h + i*s_size, s_byte, cudaMemcpyHostToDevice, streams[i]));
        checkCuda(cudaMemcpyAsync(B_d + i*s_size, B_h + i*s_size, s_byte, cudaMemcpyHostToDevice, streams[i]));
    }
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel  ---------------------------
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);
  //  basicVecAdd(A_d, B_d, C_d, VecSize); //In kernel.cu
    for (int i=0;i<s_num;i++) {
        VecAdd<<<grid_size, block_size, 0, streams[i]>>>(s_size, A_d+i*s_size, B_d+i*s_size, C_d+i*s_size);
    }
  //  cuda_ret = cudaDeviceSynchronize();
  //	if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE
    //cudaDeviceSynchronize();
    //cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    for (int i=0;i<s_num;i++) {
        checkCuda(cudaMemcpyAsync(C_h+i*s_size, C_d+i*s_size, s_byte, cudaMemcpyDeviceToHost, streams[i]));
    }
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    verify(A_h, B_h, C_h, VecSize);


    // Free memory ------------------------------------------------------------

    free(A_h);
    free(B_h);
    free(C_h);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
     
    for (int i=0;i<s_num;i++) {
	cudaStreamDestroy(streams[i]);
    }
    //INSERT CODE HERE
    return 0;

}
