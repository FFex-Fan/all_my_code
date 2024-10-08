#include <cstdio>
#include <iostream>
#include <cstring>
#include <ctime>
#include <cmath>
#include <map>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <sstream>
#include <algorithm>
#include <bitset>
#include <vector>
#include <deque>

#define pb push_back
#define ppb pop_back
#define yes puts("YES")
#define no puts("NO")
#define all(a) a.begin(), a.end()
#define show(x) cout << x << '\n'

#define rep2(i, a, b) for (int i = a; i <= b; i ++ )
#define rep1(i, a, b) for (int i = a; i < b; i ++ )
#define per2(i, a, b) for (int i = a; i >= b; i -- )
#define per1(i, a, b) for (int i = a; i > b; i -- )
#define fio ios::sync_with_stdio(false), cout.tie(0), cin.tie(0)

#define ff first
#define ss second

using namespace std;

typedef unsigned long long ull;
typedef pair<int, int> pii;
typedef pair<string, int> psi;
typedef pair<double, double> pdd;
typedef long long ll;

const int N = 100010, M = N * 2;
const int mod = 1000000007;
const int inf = 0x3f3f3f3f;

int q[N], tmp[N];

ll merge(int l, int r) {
	if (l >= r) return 0;
	int mid = l + r >> 1;
	ll res = merge(l, mid) + merge(mid + 1, r);

	int idx = 0, i = l, j = mid + 1;
	while (i <= mid && j <= r) 
		if (q[i] <= q[j]) tmp[idx ++ ] = q[i ++ ];
		else {
			res += mid - i + 1;
			tmp[idx ++ ] = q[j ++ ];
		}
	while (i <= mid) tmp[idx ++ ] = q[i ++ ];
	while (j <= r) tmp[idx ++ ] = q[j ++ ];

	for (i = l, j = 0; i <= r; i ++, j ++ ) q[i] = tmp[j];
	return res;
}

int main() {
	int n;
	scanf("%d", &n);
	for (int i = 0; i < n; i ++ ) scanf("%d", &q[i]);
	printf("%lld\n", merge(0, n - 1));
	return 0;
}
