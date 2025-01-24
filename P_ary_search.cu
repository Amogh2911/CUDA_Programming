
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>  // Include for LLONG_MAX

#define BLOCKSIZE 128// Define according to your GPU's capability and the problem's requirement

// CUDA kernel for P-ary search
__global__ void parySearchGPU(long long int *data, long long int arraySize, long long int range_length, long long int *search_keys, long long int *results) {
    __shared__ long long int cache[BLOCKSIZE + 2];
    __shared__ long long int range_offset;
    __shared__ bool found;
    
    long long int threadId = threadIdx.x;
    long long int blockId = blockIdx.x;
    long long int sk, old_range_length = range_length, range_start;

    if (threadId == 0) {
        range_offset = 0;
        cache[BLOCKSIZE] = LLONG_MAX;  // Use LLONG_MAX instead of 0x7FFFFFFF
        cache[BLOCKSIZE + 1] = search_keys[blockId];
    }
    __syncthreads();
    
    sk = cache[BLOCKSIZE + 1];
    while (range_length > BLOCKSIZE) {
        range_length = range_length / BLOCKSIZE;
        if (range_length * BLOCKSIZE < old_range_length)
            range_length += 1;
        old_range_length = range_length;

        range_start = range_offset + threadId * range_length;
        cache[threadId] = data[range_start];
        __syncthreads();

        if (sk >= cache[threadId] && sk < cache[threadId + 1])
            range_offset = range_start;

        __syncthreads();
    }

    // After narrowing down the search range
    range_start = range_offset + threadId;
    if (range_start < arraySize && sk == data[range_start]) {
        results[blockId] = range_start;
        found = true;
    }
    __syncthreads(); // Synchronize before checking the found flag

    // If no thread found the key, write -1 for the block
    if (threadId == 0 && !found) {
        results[blockId] = -1;
    }
}

// Function to check CUDA errors
void checkCudaError(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    const long long int arraySize = 300000000; // Array size of 10000 for elements 0 to 9999
    const long long int numKeys = 20; // Number of keys to search for
    long long int *data, *search_keys, *results;
    long long int *data_d, *search_keys_d, *results_d;

    // Allocate host memory
    data = (long long int *)malloc(arraySize * sizeof(long long int));
    search_keys = (long long int *)malloc(numKeys * sizeof(long long int));
    results = (long long int *)malloc(numKeys * sizeof(long long int));

    // Initialize data with elements from 0 to 9999
    for(long long int i = 0; i < arraySize; i++) {
        data[i] = i;
    }
    
    // Initialize results to -1 initially
    for(long long int i = 0; i < numKeys; i++) {
        results[i] = -1;
    }   

    // Initialize search keys with random numbers between 0 and 9999
    srand(time(NULL)); // Seed for random number generation
    for(long long int i = 0; i < numKeys; i++) {
        search_keys[i] = rand() % arraySize;
    }
    
    // Allocate device memory
    checkCudaError("before malloc data_d");
    cudaMalloc((void **)&data_d, arraySize * sizeof(long long int));
    checkCudaError("malloc data_d");
    cudaMalloc((void **)&search_keys_d, numKeys * sizeof(long long int));
    checkCudaError("malloc search_keys_d");
    cudaMalloc((void **)&results_d, numKeys * sizeof(long long int));
    checkCudaError("malloc results_d");

    // Copy data from host to device
    cudaMemcpy(data_d, data, arraySize * sizeof(long long int), cudaMemcpyHostToDevice);
    checkCudaError("copy data_d");
    cudaMemcpy(search_keys_d, search_keys, numKeys * sizeof(long long int), cudaMemcpyHostToDevice);
    checkCudaError("copy search_keys_d");

    // Create events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start);

    // Launch the kernel
    parySearchGPU<<<numKeys, BLOCKSIZE>>>(data_d, arraySize, arraySize, search_keys_d, results_d);
    cudaDeviceSynchronize();  // Ensure kernel completion before stopping timing
    checkCudaError("kernel execution");

    // Record stop event
    cudaEventRecord(stop);
    // cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy results from device to host
    cudaMemcpy(results, results_d, numKeys * sizeof(long long int), cudaMemcpyDeviceToHost);
    checkCudaError("copy results_d");

    // Output the results
    for (long long int i = 0; i < numKeys; ++i) {
        printf("Search key %lld found at index: %lld\n", search_keys[i], results[i]);
    }

     printf("Elapsed time: %f milliseconds\n", milliseconds);

    // Free device memory
    cudaFree(data_d);
    cudaFree(search_keys_d);
    cudaFree(results_d);

    // Free host memory
    free(data);
    free(search_keys);
    free(results);

    return 0;
}
