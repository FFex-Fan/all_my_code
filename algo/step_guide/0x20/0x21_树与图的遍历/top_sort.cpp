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
#include <queue>

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

int n, m;
int d[N], a[N], cnt;
int to[N], ne[N], h[N], tot;

void add(int x, int y) {
	to[ ++ tot] = y, ne[tot] = h[x], h[x] = tot;
	d[y] ++ ;
}

void topsort() {
	queue<int> q;
	for (int i = 1; i <= n; i ++ ) 
		if (!d[i]) q.push(i);

	while (q.size()) {
		int x = q.front(); q.pop();
		for (int i = h[x]; i; i = ne[i]) {
			int y = to[i];
			a[ ++ cnt] = x;  // 记录拓扑序列
			if (-- d[y] == 0) q.push(y);
		}
	}
}

int main() {
	cin >> n >> m;
	for (int i = 1; i <= m; i ++ ) {
		int x, y;
		scanf("%d%d", &x, &y);
		add(x, y);
	}

	topsort();
	for (int i = 1; i <= cnt; i ++ ) printf("%d ", a[i]);
	return 0;
}
