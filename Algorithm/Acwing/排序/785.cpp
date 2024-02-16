#include <cstdio>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 100010;

int n;
int q[N];

void q_sort(int l, int r) {
	if (l >= r) return;
	int i = l - 1, j = r + 1;
	int x = q[l + r >> 1];
	while (i < j) {
		while (q[ ++ i] < x);
		while (q[ -- j] > x);
		if (i < j) swap(q[i], q[j]);
	}
	q_sort(l, j);
	q_sort(j + 1, r);
}

int main() {
	cin >> n;
	for (int i = 0; i < n; i ++ ) cin >> q[i];
	q_sort(0, n - 1);
	for (int i = 0; i < n; i ++ ) cout << q[i] << ' ';
	return 0;
}
