#include <cstdio>
#include <cstring>
#include <iostream>

using namespace std;

const int N = 100100;

int n, m;
int q[N];
int d[N];

int main() {
    cin >> n >> m;
    for (int i = 1; i <= n; i ++ ) {
        cin >> q[i];
        d[i] = q[i] - q[i - 1];
    }
    while (m -- ) {
        int a, b, c;
        cin >> a >> b >> c;
        d[a] += c;
        d[b + 1] -= c;
    }
    for (int i = 1; i <= n; i ++ ) {
        d[i] += d[i - 1];
        cout << d[i] << ' ';
    }
    return 0;
}
