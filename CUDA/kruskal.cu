/* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define SWAP_UINT3(a,b) ({unsigned int t[3] = {a[0], a[1], a[2]};a[0] = b[0];a[1] = b[1]; a[2] = b[2]; b[0] = t[0]; b[1] = t[1]; b[2] = t[2];})
#define SWAP_UINT2(a,b) ({unsigned int t[2] = {a[0], a[1]};a[0] = b[0];a[1] = b[1]; b[0] = t[0]; b[1] = t[1];})

#define MAX_EDGE_WEIGHT 1000000
#define NUM_OF_THREADS 4

void make_union_find(unsigned int parent[], unsigned int rank[], int num_of_nodes)
{
    for (unsigned int i = 0; i < num_of_nodes; i++) {
        parent[i] = i;
        rank[i] = 0;
    }
}

unsigned int find_set(unsigned int parent[], unsigned int component)
{
    while (parent[component] != component){
        component = parent[component];
    }
    return component;
}

void union_set(unsigned int u, unsigned int v, unsigned int parent[], unsigned int rank[])
{
    u = find_set(parent, u);
    v = find_set(parent, v);

    if (rank[u] < rank[v]) {
        parent[u] = v;
    }
    else if (rank[u] > rank[v]) {
        parent[v] = u;
    }
    else {
        parent[v] = u;
        rank[u]++;
    }
}

void parallel_quick_sort(int start_idx, int end_idx, unsigned int edges[][3]) { //Both indexes are inclusive
    int len = end_idx - start_idx + 1;
    if (len <= 1) {
        return;
    }
    if (len == 2) {
        if (edges[start_idx][2] > edges[end_idx][2]){
            SWAP_UINT3(edges[start_idx], edges[end_idx]);
        }
        return;
    }

    //Picking a pivot
    int slen = (int) sqrt(len);

    unsigned int temp[slen][2];
    temp[0][0] = edges[start_idx][2];
    temp[0][1] = start_idx;
    for (unsigned int i = 1; i < slen; i++){
        unsigned int j = 0;
        unsigned int holder[2];
        holder[0] = edges[start_idx + i][2];
        holder[1] = start_idx + i;
        while (j < i && temp[j][0] < holder[0]){
            j++;
        }
        while (j <= i) {
            SWAP_UINT2(holder, temp[j]);
            j++;
        }
    }

    unsigned int pivot_val = temp[(int) slen / 2][0];
    int pivot_idx = temp[(int) slen / 2][1];

    //Switch so that pivot is the last element
    SWAP_UINT3(edges[pivot_idx], edges[end_idx]);

    int left_idx = start_idx, right_idx = end_idx - 1;
    while (1 > 0) {   
        while (left_idx < right_idx && edges[left_idx][2] < pivot_val) {
            left_idx++;
        }
        while (left_idx < right_idx && edges[right_idx][2] >= pivot_val) {
            right_idx--;
        }     

        if (left_idx >= right_idx) {
            if (right_idx == end_idx - 1 && edges[right_idx][2] < pivot_val){
                //if pivot is greater than all the other elements, move right_idx, and break
                right_idx += 1;
                break;
            }
            SWAP_UINT3(edges[end_idx], edges[right_idx]);
            break;
        } else {
            SWAP_UINT3(edges[left_idx], edges[right_idx]);
            left_idx++;
        }
    }

    if (right_idx - start_idx - 1 < 1024) {
        parallel_quick_sort(start_idx, right_idx - 1, edges);
    } else {
        #pragma omp task
        parallel_quick_sort(start_idx, right_idx - 1, edges);
    }
    if (end_idx - right_idx < 1024) {
        parallel_quick_sort(right_idx + 1, end_idx, edges);
    } else {
        #pragma omp task
        parallel_quick_sort(right_idx + 1, end_idx, edges);
    }
}

void run_parallel_quick_sort(int start_idx, int end_idx, unsigned int edges[][3]) {
# pragma omp parallel num_threads(NUM_OF_THREADS)
#pragma omp single
{
    parallel_quick_sort(start_idx, end_idx, edges);
} 
}

unsigned int parallel_kruskal(int num_of_edges, int num_of_nodes, unsigned int edges[][3], unsigned int result_edges[][3]) {
    run_parallel_quick_sort(0, num_of_edges- 1, edges);

    unsigned int *parent = (unsigned int *) malloc(num_of_nodes * sizeof(unsigned int));
    unsigned int *rank = (unsigned int *)malloc(num_of_nodes * sizeof(unsigned int));

    make_union_find(parent, rank, num_of_nodes);

    unsigned int min_cost = 0;
    int l = 0;

    for (unsigned int i = 0; i < num_of_edges; i++) {
        unsigned int v1 = find_set(parent, edges[i][0]);
        unsigned int v2 = find_set(parent, edges[i][1]);
        unsigned int wt = edges[i][2];

        if (v1 != v2) {
            union_set(v1, v2, parent, rank);
            min_cost += wt;
            result_edges[l][0] = edges[i][0];
            result_edges[l][1] = edges[i][1];
            result_edges[l][2] = edges[i][2];
            l += 1;
            if (l == num_of_nodes) {
                break;
            }
        }
    }
    free(parent);
    free(rank);
    return min_cost;
}

unsigned int CUDA_parallel_kruskal(int num_of_edges, int num_of_nodes, unsigned int edges[][3], unsigned int result_edges[][3]) {
    //run_parallel_quick_sort(0, num_of_edges- 1, edges);

    unsigned int *parent = (unsigned int *) malloc(num_of_nodes * sizeof(unsigned int));
    unsigned int *rank = (unsigned int *)malloc(num_of_nodes * sizeof(unsigned int));

    make_union_find(parent, rank, num_of_nodes);

    unsigned int min_cost = 0;
    int l = 0;

    for (unsigned int i = 0; i < num_of_edges; i++) {
        unsigned int v1 = find_set(parent, edges[i][0]);
        unsigned int v2 = find_set(parent, edges[i][1]);
        unsigned int wt = edges[i][2];

        if (v1 != v2) {
            union_set(v1, v2, parent, rank);
            min_cost += wt;
            result_edges[l][0] = edges[i][0];
            result_edges[l][1] = edges[i][1];
            result_edges[l][2] = edges[i][2];
            l += 1;
            if (l == num_of_nodes) {
                break;
            }
        }
    }
    free(parent);
    free(rank);
    return min_cost;
}

void sequential_quick_sort(int start_idx, int end_idx, unsigned int edges[][3]) { //Both indexes are inclusive
    int len = end_idx - start_idx + 1;
    if (len <= 1) {
        return;
    }
    if (len == 2) {
        if (edges[start_idx][2] > edges[end_idx][2]){
            SWAP_UINT3(edges[start_idx], edges[end_idx]);
        }
        return;
    }

    //Picking a pivot
    int slen = (int) sqrt(len);

    unsigned int temp[slen][2];
    temp[0][0] = edges[start_idx][2];
    temp[0][1] = start_idx;
    for (unsigned int i = 1; i < slen; i++){
        unsigned int j = 0;
        unsigned int holder[2];
        holder[0] = edges[start_idx + i][2];
        holder[1] = start_idx + i;
        while (j < i && temp[j][0] < holder[0]){
            j++;
        }
        while (j <= i) {
            SWAP_UINT2(holder, temp[j]);
            j++;
        }
    }

    unsigned int pivot_val = temp[(int) slen / 2][0];
    int pivot_idx = temp[(int) slen / 2][1];

    //Switch so that pivot is the last element
    SWAP_UINT3(edges[pivot_idx], edges[end_idx]);

    int left_idx = start_idx, right_idx = end_idx - 1;
    while (1 > 0) {   
        while (left_idx < right_idx && edges[left_idx][2] < pivot_val) {
            left_idx++;
        }
        while (left_idx < right_idx && edges[right_idx][2] >= pivot_val) {
            right_idx--;
        }     

        if (left_idx >= right_idx) {
            if (right_idx == end_idx - 1 && edges[right_idx][2] < pivot_val){
                //if pivot is greater than all the other elements, move right_idx, and break
                right_idx += 1;
                break;
            }
            SWAP_UINT3(edges[end_idx], edges[right_idx]);
            break;
        } else {
            SWAP_UINT3(edges[left_idx], edges[right_idx]);
            left_idx++;
        }
    }

    sequential_quick_sort(start_idx, right_idx - 1, edges);
    sequential_quick_sort(right_idx + 1, end_idx, edges);

}

int sequential_kruskal(int num_of_edges, int num_of_nodes, unsigned int edges[][3], unsigned int result_edges[][3]) {
    sequential_quick_sort(0, num_of_edges- 1, edges);

    unsigned int *parent = (unsigned int *) malloc(num_of_nodes * sizeof(unsigned int));
    unsigned int *rank = (unsigned int *) malloc(num_of_nodes * sizeof(unsigned int));

    make_union_find(parent, rank, num_of_nodes);

    unsigned int min_cost = 0;
    int l = 0;

    for (unsigned int i = 0; i < num_of_edges; i++) {
        unsigned int v1 = find_set(parent, edges[i][0]);
        unsigned int v2 = find_set(parent, edges[i][1]);
        unsigned int wt = edges[i][2];

        if (v1 != v2) {
            union_set(v1, v2, parent, rank);
            min_cost += wt;
            result_edges[l][0] = edges[i][0];
            result_edges[l][1] = edges[i][1];
            result_edges[l][2] = edges[i][2];
            l += 1;
            if (l == num_of_nodes) {
                break;
            }
        }
    }
    free(parent);
    free(rank);
    return min_cost;
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

void create_edges(int num_of_edges, int num_of_nodes, unsigned int edges[num_of_edges][3]) {
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


int main()
{
    int num_of_edges = 100000000;
    int num_of_nodes = 100000;
    unsigned int (*edges)[3] = (unsigned int (*)[3]) malloc(num_of_edges * 3 * sizeof(unsigned int));
    create_edges(num_of_edges, num_of_nodes, edges);
    unsigned int (*copy)[3] = copy_array(edges, num_of_edges);

    //unsigned int (*CUDA_copy)[3] = copy_array(edges, num_of_edges);

    unsigned int (*parallel_resulting_edges)[3] = (unsigned (*)[3]) malloc(num_of_nodes * 3 * sizeof(unsigned int));
    //unsigned int (*CUDA_parallel_resulting_edges)[3] = (unsigned (*)[3]) malloc(num_of_nodes * 3 * sizeof(unsigned int));
    unsigned int (*sequential_resulting_edges)[3] = (unsigned (*)[3]) malloc(num_of_nodes * 3 * sizeof(unsigned int));

    double end, start = omp_get_wtime();
    unsigned int parallel_min_cost = parallel_kruskal(num_of_edges, num_of_nodes, edges, parallel_resulting_edges);
    end = omp_get_wtime();
    printf("Resulting minimal cost is: %d\n", parallel_min_cost);
    printf("Execution time (parallel:OpenMP): %lf.\n", end - start);

/*     start = omp_get_wtime();
    unsigned int sequential_min_cost = sequential_kruskal(num_of_edges, num_of_nodes, copy, sequential_resulting_edges);
    end = omp_get_wtime();
    printf("Resulting minimal cost is: %d\n", sequential_min_cost);
    printf("Execution time (parallel:CUDA): %lf.\n", end - start); */

    start = omp_get_wtime();
    unsigned int sequential_min_cost = sequential_kruskal(num_of_edges, num_of_nodes, copy, sequential_resulting_edges);
    end = omp_get_wtime();
    printf("Resulting minimal cost is: %d\n", sequential_min_cost);
    printf("Execution time (sequential): %lf.\n", end - start);


// ?
    for (int i = 0; i < num_of_edges; i++) {
        if (edges[i][2] != copy[i][2] || edges[i][1] != copy[i][1] || edges[i][0] != copy[i][0]) {
            printf("There was an error in sorting arrays, arrays are not the same\n");
            break;
        }
    }

// ?
    for (int i = 0; i < num_of_nodes; i++) {
        if (parallel_resulting_edges[i][2] != sequential_resulting_edges[i][2] || 
            parallel_resulting_edges[i][1] != sequential_resulting_edges[i][1] || 
            parallel_resulting_edges[i][0] != sequential_resulting_edges[i][0]) {
            printf("There was an error in the resulting MST, MSTs are not the same\n");
            break;
        }
    }

    free(parallel_resulting_edges);
    //free(CUDA_parallel_resulting_edges)
    free(sequential_resulting_edges);
    free(edges);
    free(copy);
    return 0;
}