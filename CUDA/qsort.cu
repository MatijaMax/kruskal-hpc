#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

//MD 256 IS 32

#define MAX_DEPTH  256
#define MAX_EDGE_WEIGHT 1000000
#define INSERTION_SORT 32

//CDP -- Cooperative Groups Dynamic Parallelism -- Kernel thread starts new kernels inside GPU

__device__ void selection_sort(unsigned int data[][3], unsigned int left, unsigned int right) {
  for (int i = left; i <= right; ++i) {
    unsigned min_val = data[i][2];
    int min_idx = i;

    // Find the smallest value in the range [left, right].
    for (int j = i + 1; j <= right; ++j) {
      unsigned val_j = data[j][2];

      if (val_j < min_val) {
        min_idx = j;
        min_val = val_j;
      }
    }

    // Swap the values.
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
        

        //can pointers make it faster?
        //unsigned int* temp = data[min_idx];
        //data[min_idx] = data[i][0];
        //data[i][0] = temp;
    }
  }
}


__global__
void cdp_simple_quicksort(unsigned int data[][3], unsigned int left, unsigned int right, int depth){

    if( depth >= MAX_DEPTH || right-left <= INSERTION_SORT ){
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
        // Find the next left- and right-hand values to swap
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

        // If the swap points are valid, do the swap!
        if (lptr <= rptr){
            unsigned int temp[3];

            // Copy the values from lptr to temp
            temp[0] = (*lptr)[0];
            temp[1] = (*lptr)[1];
            temp[2] = (*lptr)[2];

            // Swap values from rptr to lptr
            (*lptr)[0] = (*rptr)[0];
            (*lptr)[1] = (*rptr)[1];
            (*lptr)[2] = (*rptr)[2];

            // Copy the values from temp (original lptr) to rptr
            (*rptr)[0] = temp[0];
            (*rptr)[1] = temp[1];
            (*rptr)[2] = temp[2];
            lptr++;
            rptr--;
        }
    }

    // Now the recursive part
    nright = rptr - data;
    nleft  = lptr - data;

    // Launch a new block to sort the left part.
    if (left < (rptr-data)){
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        // gridDim, blockDim, sharedMem = 0, stream = async parallel func  
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


// CPU mem -> GPU mem -> kernal does the job -> CPU mem
void gpu_qsort(unsigned int data[][3], int n){
    unsigned int (*gpuData)[3];
    unsigned int left = 0;
    unsigned int right = n-1;

    // Prepare CDP for the max depth 'MAX_DEPTH'.
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH);

    cudaMalloc((void**)&gpuData,n* 3 * sizeof(unsigned int));
    cudaMemcpy(gpuData,data, n* 3 * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Start kernal (gpu function)
    cdp_simple_quicksort<<< 1, 1 >>>(gpuData, left, right, 0);
    cudaDeviceSynchronize();

    cudaMemcpy(data, gpuData, n* 3 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(gpuData);

    cudaDeviceReset();
}

void create_edges(int num_of_edges, int num_of_nodes, unsigned int edges[20][3]) {
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

unsigned int (*copy_array(unsigned int (*array)[3], int len))[3]{
    unsigned int (*ret)[3] = (unsigned int (*)[3]) malloc(len * 3 * sizeof(unsigned int));
    for (int i = 0; i < len; i++){
        ret[i][0] = array[i][0];
        ret[i][1] = array[i][1];
        ret[i][2] = array[i][2];
    }
    return ret;
}


int main() {

    //int num_of_edges = 200;
    //int num_of_nodes = 100;

    //Execution time (CUDA): 42.844318 for qsort, recursion gives big overhead and a dataset too large makes race sorting invalid
    int num_of_edges = 100000000;
    int num_of_nodes = 100000;
    unsigned int (*edges)[3] = (unsigned int (*)[3]) malloc(num_of_edges * 3 * sizeof(unsigned int));
    create_edges(num_of_edges, num_of_nodes, edges);

    printf("Before sorting: ");
    printf("\n");
    for (int i = 0; i < 20; ++i)
        printf("%d %d %d\n", edges[i][0], edges[i][1], edges[i][2]);
    printf("\n");

    double end, start = omp_get_wtime();
    gpu_qsort(edges, num_of_edges);
    end = omp_get_wtime();
    printf("Execution time (CUDA): %lf.\n", end - start);

    printf("After sorting: ");
    printf("\n");
    for (int i = 0; i < 20; ++i)
        printf("%d %d %d\n", edges[i][0], edges[i][1], edges[i][2]);
    printf("\n");

    free(edges);
    return 0;
}