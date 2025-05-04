#include<iostream>
#include<vector>
#include<cuda.h>
using namespace std;

void printMatrix(int *mat,int rows,int cols,const string &name){
    cout<<name<<" : "<<endl;
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            cout<<mat[i*cols+j]<<" ";
        }
        cout<<endl;
    }
    cout<<endl;
}

void matrixMultiSequential(int *A, int *B,int *C,int m,int n,int k){
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            int sum=0;
            for(int p=0;p<k;p++){
                sum+=A[i*k+p]*B[p*n+j];
            }
            C[i*n+j]=sum;
        }
    }
}

__global__ void matrixMultiParallel(int *A,int *B,int *C,int m,int n,int k){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row<m && col<n){
        int sum=0;
        for(int p=0;p<k;p++){
            sum+=A[row*k+p]*B[p*n+col];
        }
        C[row*n+col]=sum;
    }
}

int main(){
    int M=100, K=80, N=100;
    int size_A = M*K*sizeof(int);
    int size_B = K*N*sizeof(int);
    int size_C = M*N*sizeof(int);

    int *h_A = (int*)malloc(size_A);
    int *h_B = (int*)malloc(size_B);
    int *h_C = (int*)malloc(size_C);
    int *h_C_parallel = (int*)malloc(size_C);

    srand(time(0));
    for(int i=0;i<M*K;i++) h_A[i] = rand()%19 +1;
    for(int i=0;i<K*N;i++) h_B[i] = rand()%19 +1;

    int *d_A,*d_B,*d_C;
    cudaMalloc(&d_A,size_A);
    cudaMalloc(&d_B,size_B);
    cudaMalloc(&d_C,size_C);

    cudaMemcpy(d_A,h_A,size_A,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,size_B,cudaMemcpyHostToDevice);

    clock_t start = clock();
    matrixMultiSequential(h_A,h_B,h_C,M,N,K);
    clock_t end = clock();
    double seq = double(end-start)/CLOCKS_PER_SEC;
    cout<<"\nSequential Time : "<<seq<<endl;

    dim3 threads(16,16);
    dim3 blocks((N+15)/16,(M+15)/16);

    start = clock();
    matrixMultiParallel<<<blocks,threads>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    end = clock();
    double par = double(end-start)/CLOCKS_PER_SEC;
    cout<<"\nParallel Time : "<<par<<endl;

    cudaMemcpy(h_C_parallel, d_C, size_C, cudaMemcpyDeviceToHost);

    printMatrix(h_A,M,K,"Matrix A");
    printMatrix(h_B,K,N,"Matrix B");
    printMatrix(h_C,M,N,"Sequential");
    printMatrix(h_C_parallel,M,N,"Paralle");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_parallel);

    return 0;
}

//!nvcc -arch=sm_75 matrix_multiplication.cu -o matrix_multiplication
// !./matrix_multiplication