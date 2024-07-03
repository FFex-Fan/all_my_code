#include <cstdio>
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 100010;

int h[N], to[N * 2], ne[N * 2], idx;
int v[N], sz[N];
int pos, ans = 1e9;  // pos 表示重心的位置
int n;

void add(int a, int b) {
    to[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

void dfs(int x) {
    v[x] = 1, sz[x] = 1;
    int max_part = 0; // max_part(x) 表示在删除节点 x 后产生的子树中最大的一颗的大小

    for (int i = h[x]; ~i; i = ne[x]) {
        // cout << " i = " << i << endl;
        int y = to[i];
        if (v[y]) continue;
        dfs(y);
        sz[x] += sz[y];
        max_part = max(max_part, sz[y]);  // 找出最大的子树
    }
    
    max_part = max(max_part, n - sz[x]);
    
    if (max_part < ans) {
        ans = max_part;  // 记录最大一颗子树的大小
        pos = x;  // 记录重心位置
    }
}

int main() {
    cin >> n;
    memset(h, -1, sizeof h);
    for (int i = 0; i < n - 1; i ++ ) {
        int a, b;
        cin >> a >> b;
        add(a, b); add(b, a);
    }
    
    dfs(1);
    
    cout << ans << endl;
    
    return 0;
}
