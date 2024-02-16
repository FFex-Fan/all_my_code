#include <cstdio>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 100010;

int q[N];
int n, a;

int main() {
    cin >> n >> a;
    for (int i = 0; i < n; i ++ ) cin >> q[i];
    sort(q, q + n);
    cout << q[a - 1] << endl;
    return 0;
}
