#include <cstdio>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 100010;

int n;
int a[N];

int main() {
	ios::sync_with_stdio(false);
	cin.tie(0), cout.tie(0);

	cin >> n;
	for (int i = 0; i < n; i ++ ) 
		cin >> a[i];
	sort(a, a + n);
	int pos = a[n / 2], res = 0;
	for (int i = 0; i < n; i ++ ) res += abs(pos - a[i]);
	cout << res << endl;
	return 0;
}
