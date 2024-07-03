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

const int N = 110, M = N * 2;
const int mod = 1000000007;
const int inf = 0x3f3f3f3f;

int n, m, r;
bool a[N][N];

bool calc(int i, int j, int x, int y) {
	int dx = x - i, dy = y - j;
	return r >= sqrt(dx * dx + dy * dy);
}

void get_area(int x, int y) {
	for (int i = x - r; i <= x + r; i ++ )
		for (int j = y - r; j <= y + r; j ++ ) {
			if (a[i][j]) continue;
			if (calc(i, j, x, y)) a[i][j] = true;
		}
}

int main() {
	fio;
	cin >> n >> m >> r;
	while (m -- ) {
		int x, y;
		cin >> x >> y;
		get_area(x, y);
	}
	int res = 0;
	for (int i = 1; i <= n; i ++ )
		for (int j = 1; j <= n; j ++ )
			if (a[i][j]) res ++ ;
	show(res);
	return 0;
}
