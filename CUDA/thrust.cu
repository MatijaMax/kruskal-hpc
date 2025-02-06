/* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
*/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define MAX_EDGE_WEIGHT 1000000
#define NUM_OF_EDGES 100000000
#define NUM_OF_NODES 100000

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
void gpu_sort(unsigned int (*edges)[3], int numEdges) {

    //Host
    thrust::host_vector<unsigned int> host_edge_weights1(numEdges);
    thrust::host_vector<unsigned int> host_edge_weights2(numEdges);
    thrust::host_vector<unsigned int> host_nodes1(numEdges);
    thrust::host_vector<unsigned int> host_nodes2(numEdges);

    //double endb, startb = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < numEdges; ++i) {
        host_edge_weights1[i] = edges[i][2];
        host_edge_weights2[i] = edges[i][2]; 
        host_nodes1[i] = edges[i][0]; 
        host_nodes2[i] = edges[i][1];  
    }

    //endb = omp_get_wtime();
    //printf("Setup time: %lf.\n", endb - startb);

    //Device
    thrust::device_vector<unsigned int> device_edge_weights1 = host_edge_weights1;
    thrust::device_vector<unsigned int> device_edge_weights2 = host_edge_weights2;
    thrust::device_vector<unsigned int> device_nodes1 = host_nodes1;
    thrust::device_vector<unsigned int> device_nodes2 = host_nodes1;

    //cudaEvent_t start, stop;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);
    //cudaEventRecord(start);

    thrust::sort_by_key(device_edge_weights1.begin(), device_edge_weights1.end(), device_nodes1.begin());
    thrust::sort_by_key(device_edge_weights2.begin(), device_edge_weights2.end(), device_nodes2.begin());     
 
    thrust::copy(device_edge_weights1.begin(), device_edge_weights1.end(), host_edge_weights1.begin());
    thrust::copy(device_nodes1.begin(), device_nodes1.end(), host_nodes1.begin());
    thrust::copy(device_nodes2.begin(), device_nodes2.end(), host_nodes1.begin());

    //cudaEventRecord(stop);
    //cudaEventSynchronize(stop);

    //float milliseconds = 0;
    //cudaEventElapsedTime(&milliseconds, start, stop);
    //printf("Execution time (THRUST): %f.\n", milliseconds);
    //float seconds = milliseconds / 1000.0f;
    //printf("Execution time (THRUST): %f.\n", seconds);

    // Assemble original array
    #pragma omp parallel for
    for (int i = 0; i < numEdges; ++i) {
        edges[i][0] = host_nodes1[i];  //first node
        edges[i][1] = host_nodes2[i];  // second node
        edges[i][2] = host_edge_weights1[i];  // weight
    } 
}

int main() {
    // 8 cores, 16 threads
    omp_set_num_threads(16);
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
    gpu_sort(edges, num_of_edges);
    end = omp_get_wtime();
    printf("Execution time: %lf.\n", end - start);

    printf("After sorting: ");
    printf("\n");
    for (int i = 100000; i < 100020; ++i)
        printf("%d %d %d\n", edges[i][0], edges[i][1], edges[i][2]);
    printf("\n");

    free(edges);
    return 0;
}