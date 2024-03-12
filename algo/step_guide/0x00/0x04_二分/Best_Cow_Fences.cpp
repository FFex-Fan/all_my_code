#include <cstdio>
#include <iostream>
#include <cmath>

using namespace std;

const int N = 100001;

double a[N], b[N], s[N];
int n, L;

int main() {
	cin >> n >> L;
	for (int i = 1; i <= n; i ++ ) scanf("%lf", &a[i]);

	double eps = 1e-5, l = -1e6, r = 1e6;
	while (r - l > eps) {
		double mid = (l + r) / 2;
		for (int i = 1; i <= n; i ++ ) {
			b[i] = a[i] - mid;  // 处理平均值先减去mid，转化为子段非正或非负
			s[i] = s[i - 1] + b[i];
		}

		double ans = -1e10, min_val = 1e10;
		for (int i = L; i <= n; i ++ ) {
			min_val = min(min_val, s[i - L]);
			ans = max(ans, s[i] - min_val);
		}

		if (ans >= 0) l = mid; else r = mid;
	}
	cout << int(r * 1000) << endl;
	return 0;
}
