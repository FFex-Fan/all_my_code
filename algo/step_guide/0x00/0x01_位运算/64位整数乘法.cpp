/*
a = 3, b = 15 = 1111;
b = 2 ^ 0 + 2 ^ 1 + 2 ^ 2 + 2 ^ 3 = 15
a * b = a * (2 ^ 0 + 2 ^ 1 + 2 ^ 2 + 2 ^ 3)
      = 1 * (a * 2 ^ 0) + 1 * (a * 2 ^ 1) + 1 * (a * 2 ^ 2) + 1 * (a * 2 ^ 3)
*/

#include <iostream>

using namespace std;

typedef long long ll;

ll a, b, p;

int main() {
    cin >> a >> b >> p;
    ll ans = 0;
    for ( ; b; b >>= 1) {
        if (b & 1) ans = (ans + a) % p;
        a = (a * 2) % p;
    }
    cout << ans << endl;
    return 0;
}
