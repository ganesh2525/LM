#include<iostream>
#include<cuda.h>
#include<ctime>
using namespace std;

void vectorAddSequential(int *a, int *b, int *c,int n){
    for(int i=0;i<n;i++){
        c[i]=a[i]+b[i];
    }
}

__global__ void vectorAddParallel(int *a,int *b,int *c,int n){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx<n){
        c[idx]=a[idx]+b[idx];
    }
}

void vectorPrint(int *a,int *b,int *c,int *par,int n){
    cout<<"\nIndex\t\t"<<"a\t"<<"b\t\t"<<"Seq\t\t"<<"Parallel\t\t"<<endl;
    for(int i=0;i<n;i++){
        cout<<i<<"\t\t"
        <<a[i]<<"\t"
        <<b[i]<<"\t\t"
        <<c[i]<<"\t\t"
        <<par[i]<<endl;
    }
}

int main(){
    int n = 1<<10;
    size_t size = n * sizeof(int);

    int *h_a = (int*)malloc(size);
    int *h_b = (int*)malloc(size);
    int *h_c = (int*)malloc(size);
    int *h_c_parallel = (int*)malloc(size);

    srand(time(0));
    for(int i=0;i<n;i++){
        h_a[i] = rand()%1000;
        h_b[i] = rand()%1000;
    }

    int *d_a,*d_b,*d_c;
    cudaMalloc(&d_a,size);
    cudaMalloc(&d_b,size);
    cudaMalloc(&d_c,size);

    cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,size,cudaMemcpyHostToDevice);

    clock_t start = clock();
    vectorAddSequential(h_a,h_b,h_c,n);
    clock_t end = clock();
    double seq = double(end-start)/CLOCKS_PER_SEC;
    cout<<"Sequential Time: "<<seq<<endl;

    int threads = 256;
    int blocks = (n+threads-1) / threads;

    start = clock();
    vectorAddParallel<<<blocks,threads>>>(d_a,d_b,d_c,n);
    cudaDeviceSynchronize();
    end = clock();
    double par = double(end-start)/CLOCKS_PER_SEC;
    cout<<"Parallel Time: "<<par<<endl;

    cudaMemcpy(h_c_parallel,d_c,size,cudaMemcpyDeviceToHost);

    vectorPrint(h_a,h_b,h_c,h_c_parallel,n);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_parallel);

    return 0;
}

// %%writefile vector_addition.cu
// !nvcc -arch=sm_75 vector_addition.cu -o vector_addition
// !./vector_addition