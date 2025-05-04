#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>
using namespace std;

void printAdjacencyList(const vector<vector<int>>& graph) {
    cout << "\nAdjacency List:\n";
    for (int i = 0; i < graph.size(); i++) {
        cout << i << "-> ";
        for (int neighbor : graph[i]) {
            cout << neighbor << " ";
        }
        cout << endl;
    }
}

void sequentialBFS(vector<vector<int>>& graph, int start) {
    int n = graph.size();
    vector<bool> visited(n, false);
    queue<int> q;

    visited[start] = true;
    q.push(start);

    while (!q.empty()) {
        int node = q.front();
        q.pop();

        cout << "( Visited node " << node << " ) ";

        for (int neighbor : graph[node]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
}

void sequentialDFS(vector<vector<int>>& graph, int start) {
    int n = graph.size();
    vector<bool> visited(n, false);
    stack<int> s;

    s.push(start);

    while (!s.empty()) {
        int node = s.top();
        s.pop();

        if (!visited[node]) {
            visited[node] = true;
            cout << "( Visited node " << node << " ) ";

            for (int neighbor : graph[node]) {
                if (!visited[neighbor]) {
                    s.push(neighbor);
                }
            }
        }
    }
}

void parallelBFS(vector<vector<int>>& graph, int start) {
    int n = graph.size();
    vector<bool> visited(n, false);
    queue<int> q;

    visited[start] = true;
    q.push(start);

    while (true) {
        vector<int> current_level;

        // Extract current level nodes under lock
        #pragma omp critical
        {
            while (!q.empty()) {
                current_level.push_back(q.front());
                q.pop();
            }
        }

        if (current_level.empty()) break;

        // Process current level in parallel
        #pragma omp parallel for
        for (int i = 0; i < current_level.size(); i++) {
            int node = current_level[i];
            printf("Thread %d visited node %d\n", omp_get_thread_num(), node);

            for (int neighbor : graph[node]) {
                // First check without lock (optimization)
                if (!visited[neighbor]) {
                    #pragma omp critical
                    {
                        if (!visited[neighbor]) {
                            visited[neighbor] = true;
                            q.push(neighbor);
                        }
                    }
                }
            }
        }
    }
}

void parallelDFS(vector<vector<int>>& graph, int start) {
    int n = graph.size();
    vector<bool> visited(n, false);
    stack<int> s;

    s.push(start);

    while (true) {
        int node;
        bool gotNode = false;

        #pragma omp critical
        {
            if (!s.empty()) {
                node = s.top();
                s.pop();
                gotNode = true;
            }
        }

        if (!gotNode) break;

        if (!visited[node]) {
            visited[node] = true;
            printf("Thread %d visited node %d\n", omp_get_thread_num(), node);

            #pragma omp parallel for
            for (int i = 0; i < graph[node].size(); i++) {
                int neighbor = graph[node][i];
                if (!visited[neighbor]) {
                    #pragma omp critical
                    s.push(neighbor);
                }
            }
        }
    }
}

int main() {
    omp_set_num_threads(4);  // Set number of threads

    // Create a graph with 100 nodes and random connections
    int N = 100;
    vector<vector<int>> graph(N);

    for (int i = 0; i < N; ++i) {
        for (int j = 1; j <= 3; ++j) {
            int neighbor = (i + j) % N;
            graph[i].push_back(neighbor);
        }
    }

    printAdjacencyList(graph);

    // Measure sequential BFS time
    double start_time = omp_get_wtime();
    cout << "\nSequential BFS starting from node 0:\n";
    sequentialBFS(graph, 0);
    double end_time = omp_get_wtime();
    cout << "Sequential BFS execution time: " << (end_time - start_time) << " seconds\n";

    // Measure sequential DFS time
    start_time = omp_get_wtime();
    cout << "\nSequential DFS starting from node 0:\n";
    sequentialDFS(graph, 0);
    end_time = omp_get_wtime();
    cout << "Sequential DFS execution time: " << (end_time - start_time) << " seconds\n";

    // Measure parallel BFS time
    start_time = omp_get_wtime();
    cout << "\nParallel BFS starting from node 0:\n";
    parallelBFS(graph, 0);
    end_time = omp_get_wtime();
    cout << "Parallel BFS execution time: " << (end_time - start_time) << " seconds\n";

    // Measure parallel DFS time
    start_time = omp_get_wtime();
    cout << "\nParallel DFS starting from node 0:\n";
    parallelDFS(graph, 0);
    end_time = omp_get_wtime();
    cout << "Parallel DFS execution time: " << (end_time - start_time) << " seconds\n";

    return 0;
}

// g++ -fopenmp assign1.cpp -o assign1
// ./assign1
