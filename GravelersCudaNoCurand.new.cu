#include <curand_kernel.h>
#include <stdio.h>

#define N 10000000  // Number of roll instances
#define ROLLS 231  // Number of rolls per instance


//Inserting xorshift32 onto device for hopefully faster RNG production
__device__ unsigned int xorshift32(unsigned int* state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    x ^= clock();
    *state = x;
    return x;
}

__global__ void rollAndFindMaxKernel(int* maxResult, int* d_numberOfIterations, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        unsigned int state = seed ^ clock() + (idx + 1);
        for (int j=0;j<100;j++){
            //Init xorshift

            int count = 0;
            for (int i = 0; i < ROLLS; i++) {
                int roll = xorshift32(&state) % 4;  // Generate a random number between 0 and 3
                if (roll == 0) count++;  // Increment if the roll is zero
            }

            // Use atomic operation to update the maximum result, and iteration count
            atomicMax(maxResult, count);
            atomicAdd(d_numberOfIterations, 1);
        }
    }
}

int main() {
    int* d_maxResult;
    int h_maxResult = 0;
    int* d_numberOfIterations;
    int h_numberOfIterations = 0;

    

    // Allocate memory on the device
    cudaMalloc((void**)&d_maxResult, sizeof(int));
    cudaMalloc(&d_numberOfIterations, sizeof(int));
    // Initialize the max result on the device
    cudaMemcpy(d_maxResult, &h_maxResult, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_numberOfIterations, &h_numberOfIterations, sizeof(int), cudaMemcpyHostToDevice);
    // Kernel setup
    int threadsPerBlock = 512;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    //Time recording stuff
    float totalTime=0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);


    rollAndFindMaxKernel << <blocksPerGrid, threadsPerBlock >> > (d_maxResult, d_numberOfIterations, time(NULL));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&totalTime, start, stop);


    // Copy the max result back to the host
    cudaMemcpy(&h_maxResult, d_maxResult, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_numberOfIterations, d_numberOfIterations, sizeof(int), cudaMemcpyDeviceToHost);
    // Print the results
    printf("Max ones: %d\n", h_maxResult);
    printf("Iterations: %d\n", h_numberOfIterations);
    printf("Total time (ms): %f\n\n", totalTime);
    // Free device memory
    cudaFree(d_maxResult);
    cudaFree(d_numberOfIterations);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
