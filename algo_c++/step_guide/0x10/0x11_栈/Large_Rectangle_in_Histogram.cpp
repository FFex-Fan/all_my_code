#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 100010;

typedef long long ll;

int n;
int l[N], r[N], h[N];
int stk[N], tt;

int main() {
    while (scanf("%d", &n), n) {
        for (int i = 1; i <= n; i ++ ) scanf("%d", &h[i]);
        tt = 0;
        h[0] = h[n + 1] = -1;  // 放置两个小于0的矩形，保证栈永不为空
        stk[0] = 0;  // 开始只有 0
        for (int i = 1; i <= n; i ++ ) {
            while (h[i] <= h[stk[tt]]) tt -- ; // 当前高度小于栈顶高度，则删除栈顶
            l[i] = stk[tt];
            stk[ ++ tt] = i;
        }
        
        tt = 0;
        stk[0] = n + 1;
        for (int i = n; i; i -- ) {
            while (h[i] <= h[stk[tt]]) tt -- ;
            r[i] = stk[tt];
            stk[ ++ tt] = i;
        }
        
        ll res = 0;
        for (int i = 1; i <= n; i ++ ) 
            res = max(res, (ll) h[i] * (r[i] - l[i] - 1));
        printf("%lld\n", res);
    }
    return 0;
}
