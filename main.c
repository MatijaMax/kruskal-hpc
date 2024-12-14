#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


///// SEQUENTIAL /////

// Comparator function to use in sorting
int comparator(const void* p1, const void* p2)
{
    const int(*x)[3] = p1;
    const int(*y)[3] = p2;

    return (*x)[2] - (*y)[2];
}

// Initialization of parent[] and rank[] arrays
void makeSet(int parent[], int rank[], int n)
{
    for (int i = 0; i < n; i++) {
        parent[i] = i;
        rank[i] = 0;
    }
}

// Function to find the parent of a node
int findParent(int parent[], int component)
{
    while (parent[component] != component)
        component = parent[component];
    return component;
}

// Function to unite two sets
void unionSet(int u, int v, int parent[], int rank[], int n)
{
    // Finding the parents
    u = findParent(parent, u);
    v = findParent(parent, v);

    if (rank[u] < rank[v]) {
        parent[u] = v;
    }
    else if (rank[u] > rank[v]) {
        parent[v] = u;
    }
    else {
        parent[v] = u;

        // Since the rank increases if
        // the ranks of two sets are same
        rank[u]++;
    }
}

// Function to find the MST
void kruskalAlgo(int n, int edge[n][3])
{

    double end, start = omp_get_wtime();
    // First we sort the edge array in ascending order
    // so that we can access minimum distances/cost
    qsort(edge, n, sizeof(edge[0]), comparator);

    int parent[n];
    int rank[n];

    // Function to initialize parent[] and rank[]
    makeSet(parent, rank, n);

    // To store the minimun cost
    int minCost = 0;

    printf(
        "Following are the edges in the constructed MST\n");
    for (int i = 0; i < n; i++) {
        int v1 = findParent(parent, edge[i][0]);
        int v2 = findParent(parent, edge[i][1]);
        int wt = edge[i][2];

        // If the parents are different that
        // means they are in different sets so
        // union them
        if (v1 != v2) {
            unionSet(v1, v2, parent, rank, n);
            minCost += wt;
            printf("%d -- %d == %d\n", edge[i][0],
                   edge[i][1], wt);
        }
    }

    printf("Minimum Cost Spanning Tree: %d\n", minCost);
    end = omp_get_wtime();
    printf("Calculation time, sequential: %lf\n", end - start);
}

///// PARALLEL /////


// Driver code
int main()
{
    int edge[5][3] = { { 0, 1, 10 },
                       { 0, 2, 6 },
                       { 0, 3, 5 },
                       { 1, 3, 15 },
                       { 2, 3, 4 } };

    // int edge[10][3] = { { 0, 2, 1 },
    //                    { 1, 2, 1 },
    //                    { 2, 6, 2 },
    //                    { 3, 6, 1 },
    //                    { 4, 6, 1 },
    //                    { 5, 6, 1 },
    //                    { 7, 1, 4 },
    //                    { 8, 5, 3 },
    //                    { 8, 7, 5 },
    //                    { 8, 9, 6 },
    //                    };

//     int edge[100][3] = {
//     {0, 2, 1}, {1, 2, 1}, {2, 6, 2}, {3, 6, 1}, {4, 6, 1},
//     {5, 6, 1}, {7, 1, 4}, {8, 5, 3}, {8, 7, 5}, {8, 9, 6},
//     {10, 3, 2}, {11, 4, 1}, {12, 5, 4}, {13, 6, 2}, {14, 0, 3},
//     {15, 7, 2}, {16, 8, 3}, {17, 9, 5}, {18, 1, 1}, {19, 10, 2},
//     {20, 11, 3}, {21, 12, 4}, {22, 13, 2}, {23, 14, 3}, {24, 15, 5},
//     {25, 16, 1}, {26, 17, 2}, {27, 18, 3}, {28, 19, 4}, {29, 20, 5},
//     {30, 21, 1}, {31, 22, 2}, {32, 23, 3}, {33, 24, 4}, {34, 25, 5},
//     {35, 26, 1}, {36, 27, 2}, {37, 28, 3}, {38, 29, 4}, {39, 30, 5},
//     {40, 31, 1}, {41, 32, 2}, {42, 33, 3}, {43, 34, 4}, {44, 35, 5},
//     {45, 36, 1}, {46, 37, 2}, {47, 38, 3}, {48, 39, 4}, {49, 40, 5},
//     {50, 41, 1}, {0, 42, 2}, {1, 43, 3}, {2, 44, 4}, {3, 45, 5},
//     {4, 46, 1}, {5, 47, 2}, {6, 48, 3}, {7, 49, 4}, {8, 50, 5},
//     {9, 0, 1}, {10, 1, 2}, {11, 2, 3}, {12, 3, 4}, {13, 4, 5},
//     {14, 5, 1}, {15, 6, 2}, {16, 7, 3}, {17, 8, 4}, {18, 9, 5},
//     {19, 10, 1}, {20, 11, 2}, {21, 12, 3}, {22, 13, 4}, {23, 14, 5},
//     {24, 15, 1}, {25, 16, 2}, {26, 17, 3}, {27, 18, 4}, {28, 19, 5},
//     {29, 20, 1}, {30, 21, 2}, {31, 22, 3}, {32, 23, 4}, {33, 24, 5},
//     {34, 25, 1}, {35, 26, 2}, {36, 27, 3}, {37, 28, 4}, {38, 29, 5},
//     {39, 30, 1}, {40, 31, 2}, {41, 32, 3}, {42, 33, 4}, {43, 34, 5},
//     {44, 35, 1}, {45, 36, 2}, {46, 37, 3}, {47, 38, 4}, {48, 39, 5}
// };

    kruskalAlgo(5, edge);

    return 0;
}