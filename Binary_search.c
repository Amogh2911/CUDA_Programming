#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void BinarySearch(int *data, int arraySize, int *search_keys, int *results, int numKeys) {
    int keyIndex;
    for (keyIndex = 0; keyIndex < numKeys; keyIndex++) {
        int key = search_keys[keyIndex];
        int low = 0;
        int high = arraySize - 1;
        int mid;

        while (low <= high) {
            mid = low + (high - low)/2;
            if (data[mid] == key) {
                results[keyIndex] = mid;
                break;
            } else if (data[mid] < key) {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
    }
}

int main() {
    const int arraySize = 300000000; // Array size of 10000 for elements 0 to 9999
    const int numKeys = 20; // Number of keys to search for
    int *data, *search_keys, *results;

    // Allocate host memory
    data = (int *)malloc(arraySize * sizeof(int));
    search_keys = (int *)malloc(numKeys * sizeof(int));
    results = (int *)malloc(numKeys * sizeof(int));

    // Initialize data with elements from 0 to 9999
    int i;
    for(i = 0; i < arraySize; i++) {
        data[i] = i;
    }
    
    //Initialize results to -1 initially
     for(i = 0; i < numKeys; i++) {
        results[i] = -1;
    }   

    // Initialize search keys with random numbers between 0 and 9999
    srand(time(NULL)); // Seed for random number generation
    for(i = 0; i < numKeys-1; i++) {
        search_keys[i] = rand() % arraySize;
    }
    search_keys[ numKeys-1]=40000;
    
    clock_t start = clock(); // Start time
    BinarySearch(data, arraySize, search_keys, results, numKeys);
    clock_t end = clock(); // End time

    double elapsed_time_ms = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC; // Convert to milliseconds

    // Output the results
    for (i = 0; i < numKeys; i++) {
        printf("Search key %d found at index: %d\n", search_keys[i], results[i]);
    }
    printf("Elapsed time for Binary Search: %f milliseconds\n", elapsed_time_ms);

    // Free host memory
    free(data);
    free(search_keys);
    free(results);

    return 0;
}

