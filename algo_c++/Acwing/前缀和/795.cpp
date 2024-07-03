#include <cstdio>
#include <iostream>
#include <cstring>

using namespace std;

const int N = 100010;

int n, m;
int q[N];

int main() {
    ios::sync_with_stdio(false);
    cin.tie(0), cout.tie(0);
    cin >> n >> m;
    for (int i = 1; i <= n; i ++ ) {
        cin >> q[i];
        q[i] = q[i] + q[i - 1];
    }
    while (m -- ) {
        int a, b;
        cin >> a >> b;
        cout << q[b] - q[a - 1] << endl;
    }
    return 0;
}
