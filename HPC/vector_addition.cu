#include <iostream>
#include <cuda.h>
#include <ctime>
using namespace std;

// Error checking macro
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        cerr << "CUDA Error: " << cudaGetErrorString(code) << " " << file << " " << line << endl;
        exit(code);
    }
}

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
    int n = 1 << 24;
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
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_c, size));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

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
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    end_time = clock();
    double parallel_time = double(end_time - start_time) / CLOCKS_PER_SEC;
    cout << "Parallel vector addition time: " << parallel_time << " seconds" << endl;

    // Copy result to host
    CUDA_CHECK(cudaMemcpy(h_c_parallel, d_c, size, cudaMemcpyDeviceToHost));

    // Output sample results
    // cout << "\nIndex\t\th_a\th_b\t\th_c (Seq)\th_c_parallel (CUDA)" << endl;
    // for (int i = 0; i < n; i++) {
    //     cout << i << "\t\t"
    //          << h_a[i] << "\t"
    //          << h_b[i] << "\t\t"
    //          << h_c[i] << "\t\t"
    //          << h_c_parallel[i] << endl;
    // }

    // Free memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_parallel);

    return 0;
}


// %%writefile vector_addition.cu
// !nvcc -arch=sm_75 vector_addition.cu -o vector_addition
// !./vector_addition