#include<iostream>
#include<vector>
#include<algorithm>
#include<numeric>
#include<chrono>
#include<omp.h>
using namespace std;

void sequentialOperations(vector<int>&arr, int &min_ele, int &max_ele, int &sum, double &avg){
    max_ele=*max_element(arr.begin(),arr.end());
    min_ele=*min_element(arr.begin(),arr.end());
    sum=accumulate(arr.begin(),arr.end(),0);
    avg=static_cast<double>(sum)/arr.size();
}

void parallelOperations(vector<int>&arr, int &min_ele, int &max_ele, int &sum, double &avg){
    int n=arr.size();

    #pragma omp parallel for reduction(min:min_ele) reduction(max:max_ele)
    for(int i=0;i<n;i++){
        max_ele=max(arr[i],max_ele);
        min_ele=min(arr[i],min_ele);
    }

    #pragma omp parallel for reduction(+:sum)
    for(int i=0;i<n;i++){
        sum+=arr[i];
    }

    avg=static_cast<double>(sum)/n;
}

void printVector(vector<int>&arr){
    cout<<"\nArray: ";
    for(int i=0;i<arr.size();i++){
        cout<<arr[i]<<" ";
    }
}

int main() {
    omp_set_num_threads(4); 

    int n=1000000;
    vector<int>arr(n);
    for(int i=0;i<n;i++){
        arr[i]=rand()%1000+1;
    }
    // printVector(arr);

    int max_ele,min_ele,sum;
    double avg;

    auto start = chrono::high_resolution_clock::now();
    sequentialOperations(arr,min_ele,max_ele,sum,avg);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double>seq=end-start;

    cout<<"\nSequential Results: ";
    cout<<"\n1.Max: "<<max_ele;
    cout<<"\n2.Min: "<<min_ele;
    cout<<"\n3.Sum: "<<sum;
    cout<<"\n4.Avg: "<<avg;
    cout<<"\n5.Time for execution: "<<seq.count();

    max_ele=INT_MIN, min_ele=INT_MAX;
    sum=0;

    start = chrono::high_resolution_clock::now();
    parallelOperations(arr,min_ele,max_ele,sum,avg);
    end= chrono::high_resolution_clock::now();
    chrono::duration<double>par=end-start;

    cout<<"\nParallel Results: ";
    cout<<"\n1.Max: "<<max_ele;
    cout<<"\n2.Min: "<<min_ele;
    cout<<"\n3.Sum: "<<sum;
    cout<<"\n4.Avg: "<<avg;
    cout<<"\n5.Time for execution: "<<par.count();

    return 0;
}