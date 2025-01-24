#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>

#define MAX_VERTICES 9

typedef struct Node {
    int vertex;
    struct Node* next;
} Node;

typedef struct Graph {
    int numVertices;
    Node** adjLists;
} Graph;

typedef struct cuGraph {
    int *froms; // size 2m
    int *nhbrs; // size 2m
    int numEdges;
} cuGraph;

Node* createNode(int v) {
    Node* newNode = malloc(sizeof(Node));
    newNode->vertex = v;
    newNode->next = NULL;
    return newNode;
}

Graph* createGraph(int numVertices) {
    Graph* graph = malloc(sizeof(Graph));
    graph->numVertices = numVertices;
    graph->adjLists = malloc(numVertices * sizeof(Node*));
    int i;
    for (i = 0; i < numVertices; i++) {
        graph->adjLists[i] = NULL;
    }
    return graph;
}
void addEdge(Graph* graph, int src, int dest) {
    // Insert new node for dest in the sorted order in adjacency list of src
    Node* newNode = createNode(dest);
    Node** curr = &graph->adjLists[src];

    // Find the correct position to insert the new node to keep the list sorted
    while (*curr != NULL && (*curr)->vertex < dest) {
        curr = &(*curr)->next;
    }
    
    // Check if current node is not the same as the one we want to insert to maintain uniqueness
    if (*curr == NULL || (*curr)->vertex != dest) {
        // Insert the new node at the found position
        newNode->next = *curr;
        *curr = newNode;
    } else {
        // Free the allocated node if the edge already exists
        free(newNode);
    }

    // Insert new node for src in the sorted order in adjacency list of dest
    newNode = createNode(src);
    curr = &graph->adjLists[dest];

    // Find the correct position to insert the new node to keep the list sorted
    while (*curr != NULL && (*curr)->vertex < src) {
        curr = &(*curr)->next;
    }
    
    // Check if current node is not the same as the one we want to insert to maintain uniqueness
    if (*curr == NULL || (*curr)->vertex != src) {
        // Insert the new node at the found position
        newNode->next = *curr;
        *curr = newNode;
    } else {
        // Free the allocated node if the edge already exists
        free(newNode);
    }
}

void printGraph(Graph* graph) {
    int i;
    for (i = 0; i < graph->numVertices; i++) {
        Node* temp = graph->adjLists[i];
        printf("\nAdjacency list of vertex %d\n head ", i+1);
        while (temp) {
            printf("-> %d ", temp->vertex+1);
            temp = temp->next;
        }
        printf("\n");
    }
}

void freeGraph(Graph* graph) {
    int i;
    for (i = 0; i < graph->numVertices; i++) {
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

cuGraph* createCuGraph(Graph* graph) {
    int numEdges = 0;
    int i;
    for (i = 0; i < graph->numVertices; ++i) {
        Node* node = graph->adjLists[i];
        while (node != NULL) {
            ++numEdges;
            node = node->next;
        }
    }

    cuGraph* cu_graph = malloc(sizeof(cuGraph));
    cu_graph->froms = malloc(numEdges * sizeof(int));
    cu_graph->nhbrs = malloc(numEdges * sizeof(int));

    int edgeIndex = 0;
    
    for (i = 0; i < graph->numVertices; ++i) {
        Node* node = graph->adjLists[i];
        while (node != NULL) {
            cu_graph->froms[edgeIndex] = i;
            cu_graph->nhbrs[edgeIndex] = node->vertex;
            edgeIndex++;
            node = node->next;
        }
    }

    // Sort nhbrs based on froms
    // Since we need a stable sort to maintain the 'froms' ordering, we implement it manually if needed
    cu_graph->numEdges=numEdges;
    return cu_graph;
}

void freeCuGraph(cuGraph* cu_graph) {
    free(cu_graph->froms);
    free(cu_graph->nhbrs);
    free(cu_graph);
}



void computeBetweennessCentrality(Graph* graph, double* betweennessCentrality) {
    int s;
    for (s = 0; s < graph->numVertices; s++) {
        double distances[MAX_VERTICES], spCount[MAX_VERTICES], delta[MAX_VERTICES];
        int i;
        for (i = 0; i < graph->numVertices; i++) {
            distances[i] = DBL_MAX;
            spCount[i] = 0;
        }
        distances[s] = 0;
        spCount[s] = 1;

        int d=0;
        // Forward propagation
        int done = 0;
        while (!done) {
            done = 1;
            int v;
            for (v = 0; v < graph->numVertices; v++) {
                if (distances[v] == d) {
                    Node* temp = graph->adjLists[v];
                    while (temp != NULL) {
                        int w = temp->vertex;
                        if (distances[w] == DBL_MAX) {
                            distances[w] = d + 1;
                            done = 0;
                        }
                        if (distances[w] == d + 1) {
                            spCount[w] += spCount[v];
                        }
                        temp = temp->next;
                    }
                }
            }
            d=d+1;
        }
        // printf("hiii %d d_d \n\n",d);
        // Backward propagation
        for (i = 0; i < graph->numVertices; i++) {
            delta[i] = 0;
        }

        while(d>1)
        {
            d=d-1;
            int v;
            for (v = 0; v < graph->numVertices; v++) {
                if (distances[v] == d) {
                    Node* temp = graph->adjLists[v];
                    while (temp != NULL) {
                        int w = temp->vertex;
                        if (distances[w] == d + 1) {
                            delta[v] += (spCount[v] / spCount[w]) * (1 + delta[w]);
                        }
                        temp = temp->next;
                    }  
                    betweennessCentrality[v] += delta[v]/2;                  
                }
            }

        }

        // int i;
        // for (i = 0; i < MAX_VERTICES; i++) {
        //     printf("Vertex %d: Betweenness Centrality: %f\n", i, betweennessCentrality[i]);
        // }


    }
}

int main() {
    srand(time(0));
    Graph* graph = createGraph(9);
    double betweennessCentrality[9] = {0};

    addEdge(graph, 0, 1); // Vertex 1 is connected to vertex 2
    addEdge(graph, 0, 2); // Vertex 1 is connected to vertex 3
    addEdge(graph, 0, 3); // Vertex 1 is connected to vertex 4
    
    addEdge(graph, 1, 0); // Vertex 2 is connected to vertex 1
    addEdge(graph, 1, 2); // Vertex 2 is connected to vertex 3
    
    addEdge(graph, 2, 0); // Vertex 3 is connected to vertex 1
    addEdge(graph, 2, 1); // Vertex 3 is connected to vertex 2
    addEdge(graph, 2, 3); // Vertex 3 is connected to vertex 4
    
    addEdge(graph, 3, 0); // Vertex 4 is connected to vertex 1
    addEdge(graph, 3, 2); // Vertex 4 is connected to vertex 3
    addEdge(graph, 3, 4); // Vertex 4 is connected to vertex 5
    addEdge(graph, 3, 5); // Vertex 4 is connected to vertex 6
    
    addEdge(graph, 4, 3); // Vertex 5 is connected to vertex 4
    addEdge(graph, 4, 5); // Vertex 5 is connected to vertex 6
    addEdge(graph, 4, 6); // Vertex 5 is connected to vertex 7
    addEdge(graph, 4, 7); // Vertex 5 is connected to vertex 8
    
    addEdge(graph, 5, 3); // Vertex 6 is connected to vertex 4
    addEdge(graph, 5, 4); // Vertex 6 is connected to vertex 5
    addEdge(graph, 5, 6); // Vertex 6 is connected to vertex 7
    addEdge(graph, 5, 7); // Vertex 6 is connected to vertex 8
    
    addEdge(graph, 6, 4); // Vertex 7 is connected to vertex 5
    addEdge(graph, 6, 5); // Vertex 7 is connected to vertex 6
    addEdge(graph, 6, 7); // Vertex 7 is connected to vertex 8
    addEdge(graph, 6, 8); // Vertex 7 is connected to vertex 9
    
    addEdge(graph, 7, 4); // Vertex 8 is connected to vertex 5
    addEdge(graph, 7, 5); // Vertex 8 is connected to vertex 6
    addEdge(graph, 7, 6); // Vertex 8 is connected to vertex 7
    
    addEdge(graph, 8, 6); // Vertex 9 is connected to vertex 7

    // Print the graph structure
    printGraph(graph);

    // Convert the Graph to cuGraph format
    cuGraph* cu_graph = createCuGraph(graph);

    // Print the 'froms' and 'nhbrs' arrays
    printf("\nfroms: ");
    int i;
    for (i = 0; i < cu_graph->numEdges; i++) {
        printf("%d ", cu_graph->froms[i]);
    }
    printf("\nnhbrs: ");
    for (i = 0; i < cu_graph->numEdges; i++) {
        printf("%d ", cu_graph->nhbrs[i]);
    }
    printf("\n");


    // Computing betweenness centrality
    computeBetweennessCentrality(graph, betweennessCentrality);

    // Printing betweenness centrality
    
    for (i = 0; i < MAX_VERTICES; i++) {
        printf("Vertex %d: Betweenness Centrality: %f\n", i, betweennessCentrality[i]);
    }


    // freeCuGraph(cu_graph);
    freeGraph(graph);

    return 0;
}

