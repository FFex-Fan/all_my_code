#include <cstdio>
#include <iostream>
#include <algorithm>
#include <unordered_map>

using namespace std;

const int N = 3000;

int n, m;

struct cow {
    int min, max;
} cows[N];

unordered_map<int, int> uii;

int main() {
    cin >> n >> m;
    for (int i = 0; i < n; i ++ ) cin >> cows[i].min >> cows[i].max;
    for (int i = 0; i < m; i ++ ) {
        int a, b;
        cin >> a >> b;
        uii[a] += b;
    }
    
    sort(cows, cows + n, [](cow a, cow b) {
        return a.max < b.max;
    });
    
    int res = 0;
    for (int i = 0; i < n; i ++ ) {
        for (int j = cows[i].min; j <= cows[i].max; j ++ ) {
            if (uii[j]) {
                uii[j] -- , res ++ ;
                break;
            }
        }
    }
    cout << res << endl;
    
    return 0;
}
