#include <iostream>
#include <cuda.h>
#include <ctime>
using namespace std;

void printMatrix(int* mat, int rows, int cols, const string& name) {
    cout << name << " = " << endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << mat[i * cols + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

// Sequential Matrix Multiplication
void matrixMulSequential(int *A, int *B, int *C, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int sum = 0;
            for (int p = 0; p < k; p++) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// CUDA Kernel for Matrix Multiplication
__global__ void matrixMulParallel(int *A, int *B, int *C, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        int sum = 0;
        for (int p = 0; p < k; p++) {
            sum += A[row * k + p] * B[p * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main() {
    int M = 100, K = 80, N = 100;
    int size_A = M * K * sizeof(int);
    int size_B = K * N * sizeof(int);
    int size_C = M * N * sizeof(int);

    int *h_A = (int*)malloc(size_A);
    int *h_B = (int*)malloc(size_B);
    int *h_C = (int*)malloc(size_C);
    int *h_C_parallel = (int*)malloc(size_C);

    srand(time(0));
    // Initialize matrices A and B with random integers between 1 and 19
    for (int i = 0; i < M * K; i++) 
        h_A[i] = rand() % 19 + 1;
    for (int i = 0; i < K * N; i++) 
        h_B[i] = rand() % 19 + 1;

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    clock_t start_time = clock();
    matrixMulSequential(h_A, h_B, h_C, M, K, N);
    clock_t end_time = clock();
    double seq_time = double(end_time - start_time) / CLOCKS_PER_SEC;
    cout << "Sequential Matrix Multiplication Time: " << seq_time << " seconds" << endl;

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);

    start_time = clock();
    matrixMulParallel<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();
    end_time = clock();

    double parallel_time = double(end_time - start_time) / CLOCKS_PER_SEC;
    cout << "Parallel Matrix Multiplication Time (CUDA): " << parallel_time << " seconds" << endl;

    cudaMemcpy(h_C_parallel, d_C, size_C, cudaMemcpyDeviceToHost);

    printMatrix(h_A, M, K, "Matrix A (h_A)");
    printMatrix(h_B, K, N, "Matrix B (h_B)");
    printMatrix(h_C, M, N, "Matrix C (Sequential Result)");
    printMatrix(h_C_parallel, M, N, "Matrix C (Parallel Result)");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_C_parallel);

    return 0;
}

//!nvcc -arch=sm_75 matrix_multiplication.cu -o matrix_multiplication
// !./matrix_multiplication