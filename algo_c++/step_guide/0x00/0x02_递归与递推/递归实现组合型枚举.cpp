#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

int n, m;
vector<int> vi;

void dfs(int x) {
    if (vi.size() == m) {
        for (int i = 0; i < vi.size(); i ++ ) 
            cout << vi[i] << ' ';
        cout << endl;
        return ;
    }
    for (int i = x + 1; i <= n; i ++ ) {
        vi.push_back(i);
        dfs(i);  
        vi.pop_back();
    }

}

int main() {
    cin >> n >> m;
    dfs(0);
    return 0;
}
