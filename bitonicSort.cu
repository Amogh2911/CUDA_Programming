#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <math.h>
#include <limits.h>
#include <cuda_runtime.h>

void printArray(int* arr,int size)
{
    for(int i=0;i<size;i++)
    {
        printf("%d ",arr[i]);
    }
    printf("\n");
}

__device__ void compareAndSwap(int &a, int &b, bool dir) {
    int temp;
    if (dir == (a > b)) {
        temp = a;
        a = b;
        b = temp;
    }
}

// The kernel now expects a boolean `dir` indicating the sorting direction.
__global__ void bitonicSortKernel(int *dev_values, int j, int k, bool dir) {
    unsigned int i, ixj;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    ixj = i ^ j;

    // Bitwise operations are fine within the device code.
    if (ixj > i) {
        if ((i & k) == 0) {
            compareAndSwap(dev_values[i], dev_values[ixj], dir);
        } else {
            compareAndSwap(dev_values[i], dev_values[ixj], !dir);
        }
    }
}

void cudaBitonicSort(int* arr, int n) {
    int *d_arr;
    cudaMalloc((void**)&d_arr, sizeof(int) * n);
    cudaMemcpy(d_arr, arr, sizeof(int) * n, cudaMemcpyHostToDevice);

    // Ensure we have enough blocks to cover all pairs in the array.
    int threadsPerBlock = 512;
    int blocksNeeded = (n + threadsPerBlock - 1) / threadsPerBlock;
    dim3 blocks(blocksNeeded);
    dim3 threads(threadsPerBlock);

    // The sort direction (ascending or descending) can be a parameter or a constant.
    bool ascending = true;  // Choose according to the desired initial direction.

    for (int step = 2; step <= n; step <<= 1) {
        for (int j = step / 2; j > 0; j >>= 1) {
            // Now we properly pass the direction to the kernel.
            bitonicSortKernel<<<blocks, threads>>>(d_arr, j, step, ascending);
            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(arr, d_arr, sizeof(int) * n, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
}


int main(){

    int k = 25; //2^k will be the number of elements of the array 
    int n=pow(2,k);
    int* array=(int *)malloc(sizeof(int)*n);

    srand(time(NULL));
    for(int i=0;i<n;i++)
    {
        array[i]=rand()%INT_MAX;
    }
    // printf("\nThe initial unsorted array values are\n");
    // printArray(array,n);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaBitonicSort(array,n);

   

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken for bitonic sort: %f ms\n", milliseconds);

    // printf("\nThe final sorted array values are\n");
    // printArray(array,n);
    return 0;
}