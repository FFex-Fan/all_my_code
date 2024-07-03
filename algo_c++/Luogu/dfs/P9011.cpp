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

const int N = 1010, M = N * 2;
const int mod = 1000000007;
const int inf = 0x3f3f3f3f;

int n, m;
int st[N], ed[N], tpr[N], v[N];
int q[N];
int backup[N];
int res = 1e8, cnt;
bool vis[N];

bool check() {
	bool flag = true;
	for (int i = 0; i < 105; i ++ )
		if (q[i] > 0) {
			flag = false;
			break;
		}
	return flag;
}

void edit(int l, int r, int v) {
	for (int i = l; i <= r; i ++ ) 
		q[i] += v;
}

void dfs(int price) {
	if (cnt > m) return;

	if (check()) res = min(res, price);

	for (int i = 0; i < m; i ++ ) {
		if (vis[i]) continue;
		cnt ++ ;
		vis[i] = true;
		edit(st[i], ed[i], -tpr[i]);
		dfs(price + v[i]);
		vis[i] = false;
		cnt --;
		edit(st[i], ed[i], tpr[i]);
	}
}

int main() {
	fio;
	cin >> n >> m;
	for (int i = 0; i < n; i ++ ) {
		int a, b, c;
		cin >> a >> b >> c;
		for (int j = a; j <= b; j ++ ) 
			q[j] = c;
	}
	for (int i = 0; i < m; i ++ ) 
		cin >> st[i] >> ed[i] >> tpr[i] >> v[i];

	dfs(0);

	show(res);
	
	return 0;
}
