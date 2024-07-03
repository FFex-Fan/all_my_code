#include <iostream>
#include <algorithm>

using namespace std;

const int N = 20;

int n, m, res;
int cat[N], cab[N];

/*  u: 当前为第 u 只猫
    k: 用了 k 辆车
*/
void dfs(int u, int k) {
   if (k >= res) return; // 现在已使用的数量比 res 多，则当前一定不是最优解
   if (u == n) {
       res = min(res, k);  // 此处可直接写成 res = k, 因为若 k >= res 在上一条语句中会退出
       return;
   }
   
   for (int i = 0; i < k; i ++ ) // 可以放入当前缆车
        if (cat[u] + cab[i] <= m) {
            cab[i] += cat[u];
            dfs(u + 1, k);
            cab[i] -= cat[u];
        }
    
    // 不能放入所有已存在的缆车，需要新开一辆缆车
    cab[k] = cat[u];
    dfs(u + 1, k + 1);
    cab[k] = 0;
}

int main() {
    cin >> n >> m;
    for (int i = 0; i < n; i ++ ) cin >> cat[i];
    sort(cat, cat + n); reverse(cat, cat + n);  // 从大到小排序，在 dfs 中可以更快 return
    res = n;  // 最多 k 辆车（一只猫一辆车）
    dfs(0, 0);
    cout << res << endl;
}
