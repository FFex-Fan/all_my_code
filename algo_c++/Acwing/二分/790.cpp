#include <iostream>
#include <cstring>

using namespace std;

double l = -100.0, r = 100.0;

int main() {
    double n;
    cin >> n;
    for (int i = 0; i < 200; i ++ ) {
        double mid = (l + r) / 2;
        if (mid * mid * mid > n) r = mid;
        else l = mid;
    }
    printf("%.6f", l);
    return 0;
}
