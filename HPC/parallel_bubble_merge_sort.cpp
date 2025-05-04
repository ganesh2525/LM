#include <iostream>
#include <omp.h>
#include <vector>
#include <algorithm>

using namespace std;

// Sequential Bubble Sort
void bubbleSortSequential(vector<int>& arr) {
    int n = arr.size();
    for(int i = 0; i < n-1; i++)
        for(int j = 0; j < n-i-1; j++)
            if(arr[j] > arr[j+1])
                swap(arr[j], arr[j+1]);
}

// Parallel Bubble Sort using Odd-Even Transposition
void bubbleSortParallel(vector<int>& arr) {
    int n = arr.size();
    #pragma omp parallel
    {
        for(int i = 0; i < n; i++) {
            int start = i % 2;
            #pragma omp for
            for(int j = start; j < n-1; j += 2) {
                if(arr[j] > arr[j+1])
                    swap(arr[j], arr[j+1]);
            }
        }
    }
}

// Sequential Merge Sort
void merge(vector<int>& arr, int l, int m, int r) {
    int n1 = m - l + 1, n2 = r - m;
    vector<int> L(n1), R(n2);
    for(int i = 0; i < n1; i++) L[i] = arr[l + i];
    for(int i = 0; i < n2; i++) R[i] = arr[m + 1 + i];

    int i = 0, j = 0, k = l;
    while(i < n1 && j < n2) arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    while(i < n1) arr[k++] = L[i++];
    while(j < n2) arr[k++] = R[j++];
}

void mergeSortSequential(vector<int>& arr, int l, int r) {
    if(l < r) {
        int m = l + (r - l) / 2;
        mergeSortSequential(arr, l, m);
        mergeSortSequential(arr, m+1, r);
        merge(arr, l, m, r);
    }
}

// Parallel Merge Sort using OpenMP
void mergeSortParallel(vector<int>& arr, int l, int r) {
    if(l < r) {
        int m = l + (r - l) / 2;
        #pragma omp parallel sections
        {
            #pragma omp section
            mergeSortParallel(arr, l, m);
            #pragma omp section
            mergeSortParallel(arr, m+1, r);
        }
        merge(arr, l, m, r);
    }
}

void printArray(const vector<int>& arr) {
    for(auto x : arr) cout << x << " ";
    cout << endl;
}

// Main Function
int main() {
    omp_set_num_threads(4);
    
    const int SIZE = 100000;
    vector<int> arr(SIZE);
    for(int i = 0; i < SIZE; i++) arr[i] = rand() % 10000;

    vector<int> arr1 = arr, arr2 = arr, arr3 = arr, arr4 = arr;

    double start, end;

    // Bubble Sort Sequential
    start = omp_get_wtime();
    bubbleSortSequential(arr1);
    end = omp_get_wtime();
    // printArray(arr1);
    cout << "Sequential Bubble Sort Time: " << end - start << " seconds\n";

    // Bubble Sort Parallel
    start = omp_get_wtime();
    bubbleSortParallel(arr2);
    end = omp_get_wtime();
    // printArray(arr2);
    cout << "Parallel Bubble Sort Time: " << end - start << " seconds\n";

    // Merge Sort Sequential
    start = omp_get_wtime();
    mergeSortSequential(arr3, 0, SIZE - 1);
    end = omp_get_wtime();
    // printArray(arr3);
    cout << "Sequential Merge Sort Time: " << end - start << " seconds\n";

    // Merge Sort Parallel
    start = omp_get_wtime();
    mergeSortParallel(arr4, 0, SIZE - 1);
    end = omp_get_wtime();
    // printArray(arr4);
    cout << "Parallel Merge Sort Time: " << end - start << " seconds\n";

    return 0;
}

// g++ -o sorting parallel_bubble_merge_sort.cpp -fopenmp
// ./sorting