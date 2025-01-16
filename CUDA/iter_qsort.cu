#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define MAX_DEPTH  16
#define MAX_EDGE_WEIGHT 1000000

#define N 20
#define M 10

 __global__ static void quicksort(unsigned int data[][3]) {
 #define MAX_LEVELS	30000

	int L, R;
    unsigned int pivot;
	int idx =  threadIdx.x + blockIdx.x * blockDim.x;
	int start[MAX_LEVELS];
	int end[MAX_LEVELS];

	start[idx] = idx;
	end[idx] = N - 1;
	while (idx >= 0) {
		L = start[idx];
		R = end[idx];
		if (L < R) {
			pivot = data[L][2];
			while (L < R) {
				while (data[R][2] >= pivot && L < R)
					R--;
				if(L < R) {
                    unsigned int temp[3];
                    temp[0] = data[L][0];
                    temp[1] = data[L][1];
                    temp[2] = data[L][2];

                    data[L][0] = data[R][0];
                    data[L][1] = data[R][1];
                    data[L][2] = data[R][2];


                    data[R][0] = temp[0];
                    data[R][1] = temp[1];
                    data[R][2] = temp[2];                   
                    L++;
                }    
				while (data[L][2] < pivot && L < R)
					L++;
				if (L < R) { 
                    unsigned int temp[3];
                    temp[0] = data[L][0];
                    temp[1] = data[L][1];
                    temp[2] = data[L][2];

                    data[L][0] = data[R][0];
                    data[L][1] = data[R][1];
                    data[L][2] = data[R][2];


                    data[R][0] = temp[0];
                    data[R][1] = temp[1];
                    data[R][2] = temp[2]; 
                    R--;
                }
			}
			data[L][2] = pivot;
			start[idx + 1] = L + 1;
			end[idx + 1] = end[idx];
			end[idx++] = L;
			if (end[idx] - start[idx] > end[idx - 1] - start[idx - 1]) {
	                        // swap start[idx] and start[idx-1]
        	                int tmp = start[idx];
                	        start[idx] = start[idx - 1];
                        	start[idx - 1] = tmp;

	                        // swap end[idx] and end[idx-1]
        	                tmp = end[idx];
                	        end[idx] = end[idx - 1];
                        	end[idx - 1] = tmp;
	                }

		}
		else
			idx--;
	}
}


// CPU mem -> GPU mem -> kernal does the job -> CPU mem
void gpu_qsort(unsigned int data[][3], int n){
    unsigned int (*gpuData)[3];
    unsigned int left = 0;
    unsigned int right = n-1;
    unsigned int cThreadsPerBlock = 128;

    // Prepare CDP for the max depth 'MAX_DEPTH'.
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH);

    cudaMalloc((void**)&gpuData,n* 3 * sizeof(unsigned int));
    cudaMemcpy(gpuData, data, n* 3 * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Start kernal (gpu function)
    quicksort <<< MAX_THREADS / cThreadsPerBlock, MAX_THREADS / cThreadsPerBlock, cThreadsPerBlock >>> (gpuData);
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

    int num_of_edges = N;
    int num_of_nodes = M;

    //int num_of_edges = 100000000;
    //int num_of_nodes = 100000;
    unsigned int (*edges)[3] = (unsigned int (*)[3]) malloc(num_of_edges * 3 * sizeof(unsigned int));
    create_edges(num_of_edges, num_of_nodes, edges);

    printf("Before sorting: ");
    printf("\n");
    for (int i = 0; i < num_of_edges; ++i)
        printf("%d %d %d\n", edges[i][0], edges[i][1], edges[i][2]);
    printf("\n");

    double end, start = omp_get_wtime();
    gpu_qsort(edges, num_of_edges);
    end = omp_get_wtime();
    printf("Execution time (CUDA): %lf.\n", end - start);

    printf("After sorting: ");
    printf("\n");
    for (int i = 0; i < num_of_edges; ++i)
        printf("%d %d %d\n", edges[i][0], edges[i][1], edges[i][2]);
    printf("\n");

    free(edges);
    return 0;
}