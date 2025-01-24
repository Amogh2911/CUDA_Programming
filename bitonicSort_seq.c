#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <limits.h>

void compareAndSwap(int *a, int *b, int dir) {
    if (dir == (*a > *b)) {
        int temp = *a;
        *a = *b;
        *b = temp;
    }
}

void bitonicMerge(int *array, int low, int cnt, int dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        int i;
        for (i = low; i < low + k; i++) {
            compareAndSwap(&array[i], &array[i + k], dir);
        }
        bitonicMerge(array, low, k, dir);
        bitonicMerge(array, low + k, k, dir);
    }
}

void bitonicSort(int *array, int low, int cnt, int dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        // Sort in ascending order since dir here is 1
        bitonicSort(array, low, k, 1);
        // Sort in descending order since dir here is 0
        bitonicSort(array, low + k, k, 0);
        // Merge the whole sequence in ascending order
        bitonicMerge(array, low, cnt, dir);
    }
}

void printArray(int *arr, int size) {
    int i;
    for (i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int main() {
    int k = 25; //2^k will be the number of elements of the array
    int n = pow(2, k);
    int *array = (int *)malloc(sizeof(int) * n);

    srand(time(NULL));
    int i;
    for (i = 0; i < n; i++) {
        array[i] = rand() % INT_MAX;
    }

    // printf("\nThe initial unsorted array values are\n");
    // printArray(array, n);

    clock_t start_time = clock(); // Start timing

    bitonicSort(array, 0, n, 1); // Perform the sort

    clock_t end_time = clock(); // End timing

    // printf("\nThe final sorted array values are\n");
    // printArray(array, n);

    double time_spent = (double)(end_time - start_time) / CLOCKS_PER_SEC * 1000; // Convert to milliseconds
    printf("Time taken for bitonic sort: %.3f milliseconds\n", time_spent);


    free(array);
    return 0;
}
