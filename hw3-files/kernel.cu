/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

#define TILE_SIZE 10

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/
     int bx = blockIdx.x;
     int by = blockIdx.y;

     int tx = threadIdx.x;
     int ty = threadIdx.y;

     int right_boundary = k*TILE_SIZE*by + k;
     float Sum = 0;
     for (int a=k*TILE_SIZE*by, b=bx*TILE_SIZE; a<right_boundary; a+=TILE_SIZE,b+=(TILE_SIZE*n))
     {
	__shared__ float Acache[TILE_SIZE][TILE_SIZE];
	__shared__ float Bcache[TILE_SIZE][TILE_SIZE];

	Acache[ty][tx] = A[a + k * ty + tx];
	Bcache[ty][tx] = B[b + n * ty + tx];
	__syncthreads();

	for (int i=0; i<TILE_SIZE; i++) {
	    Sum += Acache[ty][i] * Bcache[i][tx];
	}
	__syncthreads();
     }

    // INSERT KERNEL CODE HERE
    int c = n * TILE_SIZE * by + TILE_SIZE * bx;
    C[c + n * ty + tx] = Sum;
}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'N') && (transb != 'n')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;
    //INSERT CODE HERE
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(n / dimBlock.x, m / dimBlock.y); 
    mysgemm<<<dimGrid, dimBlock>>>(m, n, k, A, B, C);

    // Invoke CUDA kernel -----------------------------------------------------

}


