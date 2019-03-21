/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
    

__global__ void VecAdd(int n, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A + B
     *   where A is a (1 * n) vector
     *   where B is a (1 * n) vector
     *   where C is a (1 * n) vector
     *
     ********************************************************************/

     int id = blockIdx.x*blockDim.x+threadIdx.x;
     if (id < n)
	C[id] = A[id] + B[id];
}


void basicVecAdd( float *A,  float *B, float *C, int n)
{

    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = 256;
    int gridSize = (int)ceil((float)n/BLOCK_SIZE);
    VecAdd<<<gridSize, BLOCK_SIZE>>>(n, A, B, C); 

    //INSERT CODE HERE
    

}

