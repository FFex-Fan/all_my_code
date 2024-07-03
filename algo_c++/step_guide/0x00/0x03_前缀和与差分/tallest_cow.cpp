#include <cstdio>
#include <iostream>
#include <algorithm>
#include <set>

using namespace std;

const int N = 100010;

int n, p, h, m;
int b[N];
set<pair<int, int>> st;

void calc(int l, int r) {
    if (l > r) swap(l, r);
    if (st.count({l, r})) return;  //  判重
    st.insert({l, r});
    b[l + 1] --, b[r] ++ ;
}

int main() {
    cin >> n >> p >> h >> m;
    while (m -- ) {
        int a, b;
        cin >> a >> b;
        if (abs(a - b) == 1) continue;  // 无意义数据
        calc(a, b);
    }
    
    for (int i = 1; i <= n; i ++ ) {
        b[i] += b[i - 1];
        cout << h + b[i] << endl;
    }
    
    return 0;
}
