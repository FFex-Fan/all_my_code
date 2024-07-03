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

int m;
char g[10][10];
int dx[5] = {0, -1, 1, 0, 0}, dy[5] = {0, 0, 0, 1, -1};

void turn(int x, int y) {
    for (int i = 0; i < 5; i ++ ) {
        int a = x + dx[i], b = y + dy[i];
        if (a >= 0 && a < 5 && b >= 0 && b < 5) 
            g[a][b] ^= 1;
    }
}

int solve() {
    int ans = 100;
    // 第一行不同的按法会导致不同的结果（第一行确定唯一结果）
    // 但若有最小值则一定存在于32种情况中
    for (int k = 0; k < 1 << 5; k ++ ) { 
        int res = 0;
        char back[10][10];
        memcpy(back, g, sizeof g);

        for (int i = 0; i < 5; i ++) 
            if (k >> i & 1) {
                res ++;
                turn(0, i);
            }

        for (int i = 0; i < 4; i ++ ) 
            for (int j = 0; j < 5; j ++ )
                if (g[i][j] == '0') {
                    res ++ ;
                    turn(i + 1, j);
                }

        bool flag = true;
        for (int i = 0; i < 5; i ++ )
            if (g[4][i] == '0') {
                flag = false;
                break;
            }

        if (flag) ans = min(ans, res);
        memcpy(g, back, sizeof back);
    }
    if (ans > 6) return -1;
    return ans;
}

int main() {
    fio;
    cin >> m;
    while (m -- ) {
        for (int i = 0; i < 5; i ++ ) cin >> g[i];

        cout << solve() << endl;
    }
    return 0;
}
