#include <cstdio>
#include <iostream>

using namespace std;

typedef long long ll;

int a, b, p;

/**
 * 3 ^ 11 = ?
 * 15 = 1111
 * 3 ^ 1 = 3
 * 3 ^ 2 = 9
 * 3 ^ 4 = 81
 * 3 ^ 8 = 6561
 * 
 * 3 ^ 15 = 3 × 9 × 81 × 6561 = 14348907
 */

 int power(int a, int b, int p) {
     int ans = 1 % p;
    for ( ; b; b >>= 1) {
        if (b & 1) ans = (ll) ans * a % p;
        a = (ll) a * a % p;
    }
    return ans;
 }

int main() {
    scanf("%d%d%d", &a, &b, &p);
    int ans = power(a, b, p);
    printf("%d", ans);
    return 0;
}
