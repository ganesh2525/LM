#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <omp.h>
using namespace std;

// Sequential computation
void sequential_operations(const vector<int>& arr, int& min_val, int& max_val, int& sum, double& avg) {
    min_val = *min_element(arr.begin(), arr.end());
    max_val = *max_element(arr.begin(), arr.end());
    sum = accumulate(arr.begin(), arr.end(), 0);
    avg = static_cast<double>(sum) / arr.size();
}

// Parallel computation using OpenMP
void parallel_operations(const vector<int>& arr, int& min_val, int& max_val, int& sum, double& avg) {
    int n = arr.size();

    // Parallel Min and Max using OpenMP
    #pragma omp parallel for reduction(min:min_val) reduction(max:max_val)
    for (int i = 0; i < n; ++i) {
        min_val = min(min_val, arr[i]);
        max_val = max(max_val, arr[i]);
    }

    // Parallel Sum using OpenMP
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; ++i) {
        sum += arr[i];
    }

    avg = static_cast<double>(sum) / n;
}

void printVector(const vector<int>& arr) {
    cout << "Array Elements:\n";
    for (int i = 0; i < arr.size(); ++i) {
        cout << arr[i] << " ";
    }
}

int main() {
    omp_set_num_threads(4); 
    // Generate a large vector with random numbers
    int n = 1000000;
    vector<int> arr(n);
    for (int i = 0; i < n; ++i) {
        arr[i] = rand() % 1000 + 1;
    }
    // printVector(arr);

    int min_val, max_val, sum;
    double avg;

    // Measure execution time for sequential operations
    auto start = chrono::high_resolution_clock::now();
    sequential_operations(arr, min_val, max_val, sum, avg);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> sequential_time = end - start;

    cout << "\nSequential Results:" << endl;
    cout << "Min: " << min_val << ", Max: " << max_val << ", Sum: " << sum << ", Average: " << avg << endl;
    cout << "Time taken for sequential: " << sequential_time.count() << " seconds" << endl;

    // Measure execution time for parallel operations
    min_val = INT_MAX;
    max_val = INT_MIN;
    sum = 0;
    start = chrono::high_resolution_clock::now();
    parallel_operations(arr, min_val, max_val, sum, avg);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> parallel_time = end - start;

    cout << "\nParallel Results:" << endl;
    cout << "Min: " << min_val << ", Max: " << max_val << ", Sum: " << sum << ", Average: " << avg << endl;
    cout << "Time taken for parallel: " << parallel_time.count() << " seconds" << endl;

    return 0;
}

// g++ -o parallel_reduction min_max_sum_avg.cpp -fopenmp
// ./parallel_reduction