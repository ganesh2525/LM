#include<iostream>
#include<vector>
#include<queue>
#include<stack>
#include<omp.h>
#include<cstdlib>
using namespace std;

void printGraph(vector<vector<int>>&graph){
    cout<<"\nAdjacency List: "<<endl;
    for(int i=0;i<graph.size();i++){
        cout<<i<<" -> ";
        for(int x:graph[i]){
            cout<<x<<" ";
        }
        cout<<endl;
    }
}

void sequentialDfs(vector<vector<int>>&graph, int start){
    int n = graph.size();
    vector<bool>visited(n,false);
    stack<int>st;

    st.push(start);

    while(!st.empty()){
        int node=st.top();
        st.pop();

        if(!visited[node]){
            visited[node]=true;
            cout<<"( Visited node "<<node<<" ) ";
            for(int x:graph[node]){
                if(!visited[node]){
                    st.push(x);
                }
            }
        }
    }
}

void sequentialBfs(vector<vector<int>>&graph, int start){
    int n=graph.size();
    vector<bool>visited(n,false);
    queue<int>q;

    visited[start]=true;
    q.push(start);

    while(!q.empty()){
        int node=q.front();
        q.pop();
        cout<<"( Visited node "<<node<<" ) ";
        
        for(int x:graph[node]){
            if(!visited[x]){
                visited[x]=true;
                q.push(x);
            }
        }
    }
}

void parallelDfs(vector<vector<int>>&graph, int start){
    int n = graph.size();
    vector<bool>visited(n,false);
    stack<int>st;

    st.push(start);

    while(true){
        int node;
        bool gotNode=false;

        #pragma omp critical 
        {
            if(!st.empty()) {
                node=st.top();
                st.pop();
                gotNode=true;
            }
        }

        if(!gotNode) break;

        if(!visited[node]){
            visited[node]=true;
            printf("( Thread %d visited node %d )",omp_get_thread_num(),node);

            #pragma omp parallel for
            for(int i=0; i<graph[node].size(); i++){
                int x=graph[node][i];

                #pragma omp critical
                {
                    if(!visited[x]){
                        visited[x]=true;
                        st.push(x);
                    }
                }
            }
        }
    }
}

void parallelBfs(vector<vector<int>>& graph, int start) {
    int n = graph.size();
    vector<bool> visited(n, false);
    queue<int> q;

    visited[start] = true;
    q.push(start);

    while (true) {
        vector<int> current_level;

        // Extract current level nodes
        #pragma omp critical
        {
            while (!q.empty()) {
                current_level.push_back(q.front());
                q.pop();
            }
        }

        if (current_level.empty()) break;

        // Parallel processing of current level
        #pragma omp parallel for
        for (int i = 0; i < current_level.size(); i++) {
            int node = current_level[i];
            printf("( Thread %d visited node %d )\n", omp_get_thread_num(), node);

            for (int x : graph[node]) {
                #pragma omp critical
                {
                    if (!visited[x]) {
                        visited[x] = true;
                        q.push(x);
                    }
                }
            }
        }
    }
}

int main(){
    
    omp_set_num_threads(4);

    int n=100;
    vector<vector<int>>graph(n);

    for(int i=0;i<n;i++){
        for(int j=1;j<=3;j++){
            int x=(i+j)%n;
            graph[i].push_back(x);
        }
    }

    printGraph(graph);

    double start=omp_get_wtime();
    cout<<"\nSequential DFS: \n";
    sequentialDfs(graph,0);
    double end=omp_get_wtime();
    cout<<"\nTime: "<<end-start<<endl;

    start=omp_get_wtime();
    cout<<"\nSequential BFS: \n";
    sequentialBfs(graph,0);
    end=omp_get_wtime();
    cout<<"\nTime: "<<end-start<<endl;

    start=omp_get_wtime();
    cout<<"\nParallel DFS: \n";
    parallelDfs(graph,0);
    end=omp_get_wtime();
    cout<<"\nTime: "<<end-start<<endl;

    start=omp_get_wtime();
    cout<<"\nParallel BFS: \n";
    parallelBfs(graph,0);
    end=omp_get_wtime();
    cout<<"\nTime: "<<end-start<<endl;

    return 0;
}