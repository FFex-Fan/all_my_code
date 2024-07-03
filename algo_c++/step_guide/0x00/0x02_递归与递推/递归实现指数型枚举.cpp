#include <cstring> 
#include <iostream> 
#include <vector> 

using namespace std; 

int n; 
vector<int> vi; 

void dfs(int c) {
    if (c == n + 1) { 
        for (auto t : vi) 
            printf("%d ", t); 
        puts(""); 
        return; 
    } 
    // 不选 
    dfs(c + 1); // 递归 
    vi.push_back(c); // 选 c 
    dfs(c + 1); // 递归 
    vi.pop_back(); // 回溯 
} 

int main() { 
    cin >> n; 
    dfs(1); 
    return 0; 
}
