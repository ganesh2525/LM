#include<iostream>
#include<vector>
#include<omp.h>
#include<algorithm>
using namespace std;

void printArray(vector<int> &arr){
    cout<<"\nArray: ";
    for(int x:arr){
        cout<<x<<" ";
    }
    cout<<endl;
}

void bubbleSortSequential(vector<int>&arr){
    int n=arr.size();
    for(int i=0;i<n-1;i++){
        for(int j=0;j<n-i-1;j++){
            if(arr[j]>arr[j+1]){
                swap(arr[j],arr[j+1]);
            }
        }
    }
}

void bubbleSortParallel(vector<int>&arr){
    int n=arr.size();

    #pragma omp parallel
    {
        for(int i=0;i<n;i++){
            int start=i%2;

            #pragma omp for
            for(int j=start;j<n-1;j+=2){
                if(arr[j]>arr[j+1]) {
                    swap(arr[j],arr[j+1]);
                }
            }
        }
    }
}

void merge(vector<int>&arr, int l,int m, int r){
    int n1=m-l+1, n2=r-m;
    vector<int>L(n1),R(n2);
    for(int i=0;i<n1;i++) L[i]=arr[l+i];
    for(int i=0;i<n2;i++) R[i]=arr[m+i+1];

    int i=0,j=0,k=l;
    while(i<n1 && j<n2) arr[k++] = (L[i]<=R[j]) ? L[i++] : R[j++];
    while(i<n1) arr[k++] = L[i++];
    while(j<n2) arr[k++] = R[j++];
}

void mergeSortSequential(vector<int>&arr, int l,int r){
    if(l<r){
        int m = (l+r)/2;
        mergeSortSequential(arr,l,m);
        mergeSortSequential(arr,m+1,r);
        merge(arr,l,m,r);
    }
}

void mergeSortParallel(vector<int>&arr, int l,int r){
    if(l<r){
        int m = (l+r)/2;
        #pragma omp parallel sections
        {
            #pragma omp section
            mergeSortParallel(arr,l,m);
            #pragma omp section
            mergeSortParallel(arr,m+1,r);
        }
        merge(arr,l,m,r);
    }
}

int main() {
    omp_set_num_threads(4);
    
    int n=100000;
    vector<int>arr(n);
    for(int i=0;i<n;i++) arr[i]=rand()%10000;
    // printArray(arr);
    vector<int>arr1=arr, arr2=arr, arr3=arr, arr4=arr;
    
    double start = omp_get_wtime();
    bubbleSortSequential(arr1);
    double end = omp_get_wtime();
    // printArray(arr1);
    cout<<"\nSequential Bubble Sort Time: "<<end-start<<endl;

    start = omp_get_wtime();
    bubbleSortParallel(arr2);
    end = omp_get_wtime();
    // printArray(arr2);
    cout<<"\nParallel Bubble Sort Time: "<<end-start<<endl;

    start = omp_get_wtime();
    mergeSortSequential(arr3,0,n-1);
    end = omp_get_wtime();
    // printArray(arr3);
    cout<<"\nSequential Bubble Sort Time: "<<end-start<<endl;

    start = omp_get_wtime();
    mergeSortParallel(arr4,0,n-1);
    end = omp_get_wtime();
    // printArray(arr4);
    cout<<"\nParallel Bubble Sort Time: "<<end-start<<endl;

    return 0;
}