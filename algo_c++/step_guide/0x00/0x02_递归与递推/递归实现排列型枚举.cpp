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

int n;
bool vis[10];
vector<int> vi;

void dfs(int x) {
    if (vi.size() == n) {
        for (int i = 0; i < n; i ++ )
            cout << vi[i] << ' ';
        cout << endl;
        return ;
    }

    for (int i = 1; i <= n; i ++ ) {
        if (vis[i]) continue;
        vis[i] = true;
        vi.pb(i);
        dfs(i);
        vi.ppb();
        vis[i] = false;
    }
}

int main() {
    fio;
    cin >> n;
    for (int i = 1; i <= n; i ++ ) {
        vis[i] = true;
        vi.pb(i);
        dfs(i);
        vi.ppb();
        vis[i] = false;
    }
    return 0;
}
