#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <cuda_runtime.h>

#define MAX_VERTICES 100


typedef struct Node {
    int vertex;
    struct Node* next;
} Node;

typedef struct Graph {
    int numVertices;
    Node** adjLists;
} Graph;

typedef struct cuGraph {
    int *froms;
    int *nhbrs;
    int numEdges;
} cuGraph;

typedef struct cuBCData {
    int *distances;
    int *numSPs;
    bool *predecessor;
    float *dependencies;
    float *nodeBCs;
} cuBCData;



Node* createNode(int v) {
    Node* newNode = (Node*)malloc(sizeof(Node));
    newNode->vertex = v;
    newNode->next = NULL;
    return newNode;
}

Graph* createGraph(int numVertices) {
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    graph->numVertices = numVertices;
    graph->adjLists = (Node**)malloc(numVertices * sizeof(Node*));
    for (int i = 0; i < numVertices; i++) {
        graph->adjLists[i] = NULL;
    }
    return graph;
}

cuGraph* createCuGraph(Graph* graph) {
    cuGraph* cu_graph = (cuGraph*)malloc(sizeof(cuGraph));
    int numEdges = 0;
    for (int i = 0; i < graph->numVertices; ++i) {
        Node* node = graph->adjLists[i];
        while (node != NULL) {
            ++numEdges;
            node = node->next;
        }
    }
    cu_graph->numEdges = numEdges;
    cu_graph->froms = (int*)malloc(numEdges * sizeof(int));
    cu_graph->nhbrs = (int*)malloc(numEdges * sizeof(int));

    int edgeIndex = 0;
    for (int i = 0; i < graph->numVertices; ++i) {
        Node* node = graph->adjLists[i];
        while (node != NULL) {
            cu_graph->froms[edgeIndex] = i;
            cu_graph->nhbrs[edgeIndex] = node->vertex;
            edgeIndex++;
            node = node->next;
        }
    }
    return cu_graph;
}

void addEdge(Graph* graph, int src, int dest) {
    // Check if the edge already exists in src's adjacency list
    Node* current = graph->adjLists[src];
    Node* prev = NULL;
    while (current) {
        if (current->vertex == dest) {
            // Edge already exists, no need to add it
            return;
        }
        if (current->vertex > dest) {
            break; // Found the position to insert the new node
        }
        prev = current;
        current = current->next;
    }

    // The edge doesn't exist, so add it to src's list
    Node* newNode = createNode(dest);
    newNode->next = current;
    if (prev == NULL) {
        graph->adjLists[src] = newNode; // Insert at the beginning
    } else {
        prev->next = newNode; // Insert in the middle or at the end
    }

    // Check if the edge already exists in dest's adjacency list to prevent adding it twice
    current = graph->adjLists[dest];
    prev = NULL;
    while (current) {
        if (current->vertex == src) {
            // Edge already exists, no need to add it
            return;
        }
        if (current->vertex > src) {
            break; // Found the position to insert the new node
        }
        prev = current;
        current = current->next;
    }

    // The edge doesn't exist in dest's adjacency list, so add it
    newNode = createNode(src);
    newNode->next = current;
    if (prev == NULL) {
        graph->adjLists[dest] = newNode; // Insert at the beginning
    } else {
        prev->next = newNode; // Insert in the middle or at the end
    }
}

void printGraph(Graph* graph) {

    printf("\nthe number of vertices in the graph are %d\n\n",graph->numVertices);
    for (int i = 0; i < graph->numVertices; i++) {
        Node* temp = graph->adjLists[i];
        printf("Adjacency list of vertex %d: ", i);
        while (temp) {
            printf("%d ", temp->vertex);
            temp = temp->next;
        }
        printf("\n");
    }
}

void freeGraph(Graph* graph) {
    for (int i = 0; i < graph->numVertices; i++) {
        Node* temp = graph->adjLists[i];
        while (temp != NULL) {
            Node* next = temp->next;
            free(temp);
            temp = next;
        }
    }
    free(graph->adjLists);
    free(graph);
}

void freeCuGraph(cuGraph* cu_graph) {
    free(cu_graph->froms);
    free(cu_graph->nhbrs);
    free(cu_graph);
}

__global__ void forwardPropagation1(int* nedge, int *froms, int *nhbrs, int *distances, int *numSPs,int* d_d, bool *predecessor/*,int *done*/) {
    
    __shared__ bool done;
    if (threadIdx.x == 0) done = false;
    __syncthreads(); 

    while(!(done)) {
        __syncthreads();
        
        done = true;

        __syncthreads();
        
        for (int eid = threadIdx.x; eid <  *nedge; eid += blockDim.x) {
             
            int from = froms[eid];
            if (distances[from] == *d_d) {
                
                int nhbr = nhbrs[eid];
                int nhbr_dist = distances[nhbr];
                if (nhbr_dist == -1) {
                    // printf("amm12 %d eid %d block dim\n",eid, blockDim.x);
                    distances[nhbr] = (*d_d) + 1;
                    nhbr_dist = (*d_d) + 1;
                    done = false;
                }
                else if (nhbr_dist < (*d_d))
                { 
                    
                    predecessor[eid] = true;
                }
                if (nhbr_dist == (*d_d) + 1) {
                   
                    atomicAdd(&numSPs[nhbr], numSPs[from]);
                }
            }
        }

        (*d_d)++;
        
        __syncthreads();
    }
}

__global__ void backwardPropagation1(int *nedge, int *nnode, int *froms, int *nhbrs, int *distances, bool *predecessor, float *dependencies, int *numSPs, float *nodeBCs,int *d_d) {
    while((*d_d) > 1) {
        (*d_d)--;
        __syncthreads();    
        for (int eid = threadIdx.x; eid <  *nedge; eid += blockDim.x) {
            int from = froms[eid];
            if (distances[from] == (*d_d)) { 
                if (predecessor[from]) { 
                    int nhbr = nhbrs[eid];
                    float delta = (1.0f + dependencies[from]) * ((float)numSPs[nhbr] / numSPs[from]);
                    atomicAdd(&dependencies[nhbr], delta);
                }
            }
        }

    }

    for (int nid = threadIdx.x; nid < *nnode; nid += blockDim.x) {
        nodeBCs[nid] += dependencies[nid];
    }
}


void initializeSource(int source, int *distances, int *numSPs, bool *predecessor,float *dependencies, int numVertices,int numEdges) {
    // Initialize all vertices with 'infinity' for distances and 0 for numSPs and false for predecessors
    for (int i = 0; i < numVertices; i++) {
        distances[i] = -1; // Set to 'infinity'
        numSPs[i] = 0;
        dependencies[i]=0;
    }
    for(int i=0;i<numEdges;i++)
    {
         predecessor[i] = false;
    }
    
    // Set the source vertex with a distance of 0 and a numSPs of 1
    distances[source] = 0;
    numSPs[source] = 1;
}

void computeBetweennessCentrality(cuGraph *cu_graph, cuBCData *bc_data, int numVertices) {
    dim3 blocks(128);  // Assuming a moderate number of blocks
    dim3 threadsPerBlock(512);  // Assuming up to 512 threads per block

    int *d_froms, *d_nhbrs, *d_distances, *d_numSPs;
    bool *d_predecessor;  // Device memory for predecessor array
    float *d_dependencies, *d_nodeBCs;
    // int *d_done;  // To monitor changes in the forward propagation loop

    // Allocate device memory

    int *d_numEdges; // Device pointer for the number of edges

    // Allocate device memory for the number of edges
    cudaError_t cudaStatus = cudaMalloc((void **)&d_numEdges, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for numEdges!");
        // Handle the error, perhaps by returning an error code or exiting.
    }

    // Copy the number of edges from the host to the device
    cudaStatus = cudaMemcpy(d_numEdges, &(cu_graph->numEdges), sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for numEdges!");
        // Handle the error.
    }

    int Vertices = numVertices; // Initialize the number of vertices on the host
    int *d_numVertices; // Device pointer for the number of vertices

    // Allocate device memory for the number of vertices
    cudaError_t err = cudaMalloc((void **)&d_numVertices, sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for numVertices: %s\n", cudaGetErrorString(err));
        // Handle the error, for example by exiting
        exit(EXIT_FAILURE);
    }

    // Copy the number of vertices from host to device
    err = cudaMemcpy(d_numVertices, &Vertices, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy numVertices to device: %s\n", cudaGetErrorString(err));
        // Handle the error, for example by freeing the allocated memory and then exiting
        cudaFree(d_numVertices);
        exit(EXIT_FAILURE);
    }

    cudaError_t err1,err2,err3,err4,err5,err6,err7;

    // Allocate device memory
    err1 = cudaMalloc((void **)&d_froms, sizeof(int) * cu_graph->numEdges);
    if (err1 != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n for froms", cudaGetErrorString(err1));
        // Handle error...
    }
    err2 = cudaMalloc((void **)&d_nhbrs, sizeof(int) * cu_graph->numEdges);
    if (err2 != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n for nhbrs", cudaGetErrorString(err2));
        // Handle error...
    }
    err3 = cudaMalloc((void **)&d_distances, sizeof(int) * numVertices);
    if (err3 != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n for distances", cudaGetErrorString(err3));
        // Handle error...
    }
    err4 = cudaMalloc((void **)&d_numSPs, sizeof(int) * numVertices);
    if (err4 != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n for numSPs", cudaGetErrorString(err4));
        // Handle error...
    }
    err5 = cudaMalloc((void **)&d_dependencies, sizeof(float) * numVertices);
    if (err5 != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n for dependencies", cudaGetErrorString(err5));
        // Handle error...
    }
    err6 = cudaMalloc((void **)&d_nodeBCs, sizeof(float) * numVertices);
    if (err6 != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n for nodeBCs", cudaGetErrorString(err6));
        // Handle error...
    }
    err7 = cudaMalloc((void **)&d_predecessor, sizeof(bool) * cu_graph->numEdges);
    if (err7 != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n for predecessors", cudaGetErrorString(err7));
        // Handle error...
    }

    for (int i = 0; i < numVertices; i++) {
        bc_data->nodeBCs[i]=0;
    }
    cudaMemcpy(d_froms, cu_graph->froms, sizeof(int) * cu_graph->numEdges , cudaMemcpyHostToDevice);
    cudaMemcpy(d_nhbrs, cu_graph->nhbrs, sizeof(int) * cu_graph->numEdges , cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodeBCs,  bc_data->nodeBCs, sizeof(float) * numVertices, cudaMemcpyHostToDevice);  

    for (int s = 0; s < numVertices; ++s) {
        // Initialize distances, numSPs, and predecessors for all vertices
        initializeSource(s, bc_data->distances, bc_data->numSPs, bc_data->predecessor,bc_data->dependencies, numVertices, cu_graph->numEdges);
        int d = 0;
        int *d_d; // Device pointer for d
        cudaMalloc(&d_d, sizeof(int)); // Allocate memory on the device for d
        cudaMemcpy(d_d, &d, sizeof(int), cudaMemcpyHostToDevice); // Copy d to device
        // Copy initialized data to device
        cudaMemcpy(d_distances, bc_data->distances, sizeof(int) * numVertices, cudaMemcpyHostToDevice);
        cudaMemcpy(d_numSPs, bc_data->numSPs, sizeof(int) * numVertices, cudaMemcpyHostToDevice);
        cudaMemcpy(d_predecessor, bc_data->predecessor, sizeof(bool) * cu_graph->numEdges , cudaMemcpyHostToDevice);
        cudaMemcpy(d_dependencies, bc_data->dependencies, sizeof(float) * numVertices,cudaMemcpyHostToDevice);  // Clear dependencies on the device

        // Call forward propagation
        forwardPropagation1<<<blocks, threadsPerBlock/*, sizeof(int) * threadsPerBlock.x*/>>>(d_numEdges, d_froms, d_nhbrs, d_distances, d_numSPs, d_d, d_predecessor);
        cudaDeviceSynchronize();

        // Call backward propagation
        backwardPropagation1<<<blocks, threadsPerBlock>>>(d_numEdges, d_numVertices, d_froms, d_nhbrs, d_distances, d_predecessor, d_dependencies, d_numSPs, d_nodeBCs,d_d);
        cudaDeviceSynchronize();
        cudaFree(d_d);
    }

    // Copy final centrality values back to host
    cudaMemcpy(bc_data->nodeBCs, d_nodeBCs, sizeof(float) * numVertices, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_froms);
    cudaFree(d_nhbrs);
    cudaFree(d_distances);
    cudaFree(d_numSPs);
    cudaFree(d_dependencies);
    cudaFree(d_nodeBCs);
    cudaFree(d_predecessor);
    // cudaFree(d_done);
    // cudaFree(d_d);

}


int main() {
    srand(time(0));
    Graph* graph = createGraph(MAX_VERTICES);

    // Generating random graph edges with unique and sorted neighbors
    for (int i = 0; i < MAX_VERTICES; i++) {
        for (int j = i + 1; j < MAX_VERTICES; j++) {
            if (rand() % 2) {
                addEdge(graph, i, j);
            }
        }
    }


    // Print the original graph structure
    // printf("Original Graph:\n");

    // Convert the Graph to cuGraph format
    cuGraph* cu_graph = createCuGraph(graph);

    // Print the cuGraph structure
    // printf("\nCUDAGraph:\n");
    // for (int i = 0; i < cu_graph->numEdges; i++) {
    //     printf("%d ", cu_graph->froms[i]);
    // }
    // printf("\n");
    // for (int i = 0; i < cu_graph->numEdges; i++) {
    //     printf("%d ", cu_graph->nhbrs[i]);
    // }
    // printf("\n");
    // printf("Number of Edges %d\n ", cu_graph->numEdges);
    // Allocate memory for betweenness centrality computation
    cuBCData* bc_data = (cuBCData*)malloc(sizeof(cuBCData));
    bc_data->distances = (int*)malloc(MAX_VERTICES * sizeof(int));
    bc_data->numSPs = (int*)malloc(MAX_VERTICES * sizeof(int));
    bc_data->predecessor = (bool*)malloc(cu_graph->numEdges * sizeof(bool));
    bc_data->dependencies = (float*)malloc(MAX_VERTICES * sizeof(float));
    bc_data->nodeBCs = (float*)malloc(MAX_VERTICES * sizeof(float));
    // Create events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start);

    // Compute betweenness centrality
    computeBetweennessCentrality(cu_graph, bc_data, graph->numVertices);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print betweenness centrality
    // printf("\nBetweenness Centrality:\n");
    // for (int i = 0; i < MAX_VERTICES; i++) {
    //     printf("Vertex %d: BC = %f\n", i, bc_data->nodeBCs[i]);
    // }

    printf("Elapsed time: %f milliseconds\n", milliseconds);

    // Free memory
    freeGraph(graph);
    freeCuGraph(cu_graph);
    free(bc_data->distances);
    free(bc_data->numSPs);
    free(bc_data->predecessor);
    free(bc_data->dependencies);
    free(bc_data->nodeBCs);
    free(bc_data);

    return 0;
}
