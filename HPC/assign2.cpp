#include <iostream>
#include <cuda.h>
#include <ctime>
using namespace std;

// Sequential vector addition
void vectorAddSequential(int *a, int *b, int *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// CUDA kernel for parallel vector addition
__global__ void vectorAddParallel(int *a, int *b, int *c, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 1 << 24; // 1024 elements
    size_t size = n * sizeof(int);

    // Host allocations
    int *h_a = (int*)malloc(size);
    int *h_b = (int*)malloc(size);
    int *h_c = (int*)malloc(size);
    int *h_c_parallel = (int*)malloc(size);

    // Initialize host vectors
    srand(time(0));
    for (int i = 0; i < n; i++) {
        h_a[i] = rand() % 100;
        h_b[i] = rand() % 100;
    }

    // Device allocations
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Sequential execution
    clock_t start_time = clock();
    vectorAddSequential(h_a, h_b, h_c, n);
    clock_t end_time = clock();
    double seq_time = double(end_time - start_time) / CLOCKS_PER_SEC;
    cout << "Sequential vector addition time: " << seq_time << " seconds" << endl;

    // CUDA execution
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    start_time = clock();
    vectorAddParallel<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    end_time = clock();
    double parallel_time = double(end_time - start_time) / CLOCKS_PER_SEC;
    cout << "Parallel vector addition time: " << parallel_time << " seconds" << endl;

    // Copy result to host
    cudaMemcpy(h_c_parallel, d_c, size, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_parallel);

    return 0;
} 