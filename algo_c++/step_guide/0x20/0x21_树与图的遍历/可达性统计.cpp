#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <queue>
#include <bitset>

using namespace std;

const int N = 30010;

int n, m;
int h[N], ne[N], to[N], tot;
int d[N], seq[N], cnt;

bitset<N> f[N];

void add(int a, int b) {
    to[ ++ tot] = b, ne[tot] = h[a], h[a] = tot;
    d[b] ++ ;
}

void topsort() {
    queue<int> q;
    
    for (int i = 1; i <= n; i ++ ) 
        if (d[i] == 0) q.push(i);
        
    while (q.size()) {
        int a = q.front(); q.pop();
        seq[ ++ cnt] = a;
        for (int i = h[a]; i; i = ne[i]) {
            int j = to[i];
            if ( -- d[j] == 0) 
                q.push(j);
        }
    }
}

int main() {
    cin >> n >> m;
    for (int i = 0; i < m; i ++ ) {
        int a, b;
        cin >> a >> b;
        add(a, b);
    }
    
    topsort();
    
    for (int i = n; i; i -- ) {
        int j = seq[i];
        f[j][j] = 1;
        
        for (int k = h[j]; k; k = ne[k]) f[j] |= f[to[k]];
    }
    
    for (int i = 1; i <= n; i ++ ) cout << f[i].count() << endl;
    return 0;
}
