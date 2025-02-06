/* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define MAX_DEPTH  256
#define MAX_EDGE_WEIGHT 1000000
#define SELECTION_SORT 32
#define NUM_OF_EDGES 100000000
#define NUM_OF_NODES 100000

// Based on NVIDIA official implementation of qsort 
// Best execution time ~ 42s for qsort
// Recursive algorithms have big overhead

__device__ void selection_sort(unsigned int data[][3], unsigned int left, unsigned int right) {
  for (int i = left; i <= right; ++i) {
    unsigned min_val = data[i][2];
    int min_idx = i;

    for (int j = i + 1; j <= right; ++j) {
      unsigned val_j = data[j][2];

      if (val_j < min_val) {
        min_idx = j;
        min_val = val_j;
      }
    }

    if (i != min_idx) {       
        unsigned int temp[3];
        temp[0] = data[min_idx][0];
        temp[1] = data[min_idx][1];
        temp[2] = data[min_idx][2];

        data[min_idx][0] = data[i][0];
        data[min_idx][1] = data[i][1];
        data[min_idx][2] = data[i][2];

        data[i][0] = temp[0];
        data[i][1] = temp[1];
        data[i][2] = temp[2];
    }
  }
}


__global__
void cdp_simple_quicksort(unsigned int data[][3], unsigned int left, unsigned int right, int depth){

    if( depth >= MAX_DEPTH || right-left <= SELECTION_SORT ){
        selection_sort( data, left, right );
        return;
    }

    cudaStream_t s,s1;
    unsigned int (*lptr)[3] = data+left;
    unsigned int (*rptr)[3] = data+right;
    unsigned int pivot = data[(left+right)/2][2];

    unsigned int lval;
    unsigned int rval;

    unsigned int nright, nleft;

    // Do the partitioning.
    while (lptr <= rptr){
        // Find the next left and right values to swap
        lval = (*lptr)[2];
        rval = (*rptr)[2];

        // Move the left pointer as long as the pointed element is smaller than the pivot.
        while (lval < pivot && lptr < data+right){
            lptr++;
            lval = (*lptr)[2];
        }

        // Move the right pointer as long as the pointed element is larger than the pivot.
        while (rval > pivot && rptr > data+left){
            rptr--;
            rval = (*rptr)[2];
        }

        // If the swap points are valid, do the swap
        if (lptr <= rptr){
            unsigned int temp[3];

            // Swap valuesS
            temp[0] = (*lptr)[0];
            temp[1] = (*lptr)[1];
            temp[2] = (*lptr)[2];

            (*lptr)[0] = (*rptr)[0];
            (*lptr)[1] = (*rptr)[1];
            (*lptr)[2] = (*rptr)[2];

            (*rptr)[0] = temp[0];
            (*rptr)[1] = temp[1];
            (*rptr)[2] = temp[2];
            lptr++;
            rptr--;
        }
    }

    // Recursive part
    nright = rptr - data;
    nleft  = lptr - data;

    // Kernel specs: << gridDim, blockDim, sharedMem, stream >> 
    // Streams are created to launch recursive quicksort calls asynchronously ~ OMP tasks

    // Launch a new block to sort the left part.
    if (left < (rptr-data)){
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking); 
        cdp_simple_quicksort<<< 1, 1, 0, s >>>(data, left, nright, depth+1);
        cudaStreamDestroy(s);
    }

    // Launch a new block to sort the right part.
    if ((lptr-data) < right){
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        cdp_simple_quicksort<<< 1, 1, 0, s1 >>>(data, nleft, right, depth+1);
        cudaStreamDestroy(s1);
    }
}


// CPU mem => GPU mem => kernal, a GPU function, does the job => CPU mem
void gpu_qsort(unsigned int data[][3], int n){
    unsigned int (*gpuData)[3];
    unsigned int left = 0;
    unsigned int right = n-1;

    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH);

    cudaMalloc((void**)&gpuData,n* 3 * sizeof(unsigned int));
    cudaMemcpy(gpuData,data, n* 3 * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Launch kernel
    cdp_simple_quicksort<<< 1, 1 >>>(gpuData, left, right, 0);
    cudaDeviceSynchronize();

    cudaMemcpy(data, gpuData, n* 3 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(gpuData);

    cudaDeviceReset();
}

void create_edges(int num_of_edges, int num_of_nodes, unsigned int edges[NUM_OF_EDGES][3]) {
    srand(time(NULL));
    for (int i = 0; i < num_of_nodes - 1; i++) {
        edges[i][0] = i;
        edges[i][1] = i + 1;
        edges[i][2] = rand() % MAX_EDGE_WEIGHT + 1;
    }
    for (int i = num_of_nodes - 1; i < num_of_edges; i++) {
        edges[i][0] = rand() % num_of_nodes;
        edges[i][1] = rand() % num_of_nodes;
        while (edges[i][1] == edges[i][0]) {
            edges[i][1] = rand() % num_of_nodes;
        }
        edges[i][2] = rand() % MAX_EDGE_WEIGHT + 1;
    }
}

int main() {

    int num_of_edges = NUM_OF_EDGES;
    int num_of_nodes = NUM_OF_NODES;
    unsigned int (*edges)[3] = (unsigned int (*)[3]) malloc(num_of_edges * 3 * sizeof(unsigned int));
    create_edges(num_of_edges, num_of_nodes, edges);

    //We print 20 elements for validation
    printf("Before sorting: ");
    printf("\n");
    for (int i = 0; i < 20; ++i)
        printf("%d %d %d\n", edges[i][0], edges[i][1], edges[i][2]);
    printf("\n");

    double end, start = omp_get_wtime();
    gpu_qsort(edges, num_of_edges);
    end = omp_get_wtime();
    printf("Execution time: %lf.\n", end - start);

    printf("After sorting: ");
    printf("\n");
    for (int i = 0; i < 20; ++i)
        printf("%d %d %d\n", edges[i][0], edges[i][1], edges[i][2]);
    printf("\n");

    free(edges);
    return 0;
}