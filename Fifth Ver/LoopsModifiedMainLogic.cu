#include <curand_kernel.h>
#include <stdio.h>

#define N 5000000  // Number of roll instances. 
#define ROLLS 231  // Number of rolls per instance
#define THREADROLLS 200 // Each instance is running this many times. THREADROLLS * N should equal 1 billion

__global__ void rollAndFindMaxKernel(int* maxResult, int* d_numberOfIterations, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        //Init xorshift
        unsigned int state = seed ^ clock() + (idx + 1); //Ensure each thread starts with a new xorshift seed
        for (int j=0;j<THREADROLLS;j++){
            int count = 0;
            for (int i = 0; i < ROLLS; i++) {
                //Moved xorshift directly into the code. This sped it up a little bit
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                //Now the actual counting looks like this
                if (state % 4 == 0) count++;
            }

            // Use atomic operation to update the maximum result, and iteration count
            atomicMax(maxResult, count);
            atomicAdd(d_numberOfIterations, 1);

        }
    }
}

int main() {

    //Time recording stuff
    float totalTime=0;
    float kernelTime=0;
    cudaEvent_t start, stop, startKernel, stopKernel;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&startKernel);
    cudaEventCreate(&stopKernel);
    cudaEventRecord(start, 0);

    //Init both variables
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


    cudaEventRecord(startKernel, 0);
    //Run the kernel (The actual simulation)
    rollAndFindMaxKernel << <blocksPerGrid, threadsPerBlock >> > (d_maxResult, d_numberOfIterations, time(NULL));

    cudaEventRecord(stopKernel, 0);
    // Copy the max result back to the host
    cudaMemcpy(&h_maxResult, d_maxResult, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_numberOfIterations, d_numberOfIterations, sizeof(int), cudaMemcpyDeviceToHost);

    //Finishing time recording stuff
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&totalTime, start, stop);
    cudaEventElapsedTime(&kernelTime, startKernel, stopKernel);

    // Print the results
    printf("Max ones: %d\n", h_maxResult);
    printf("Iterations: %d\n", h_numberOfIterations);
    printf("Total time (ms): %f\n", totalTime);
    printf("Total kernel time (ms): %f\n\n", kernelTime);

    // Free device memory
    cudaFree(d_maxResult);
    cudaFree(d_numberOfIterations);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(startKernel);
    cudaEventDestroy(stopKernel);

    
    return 0;
}
