#include <cstdio>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 5e5 + 10;

typedef long long ll;

int a[N], tmp[N];

ll merge(int l, int r) {
	if (l >= r) return 0;
	
	int mid = l + r >> 1;
	ll res = merge(l, mid) + merge(mid + 1, r);
	
	int idx = 0, i = l, j = mid + 1;
	while (i <= mid && j <= r) {
		if (a[i] <= a[j]) tmp[idx ++ ] = a[i ++ ];
		else tmp[idx ++ ] = a[j ++ ], res += mid - i + 1;
	}
	
	while (i <= mid) tmp[idx ++ ] = a[i ++ ];
	while (j <= r) tmp[idx ++ ] = a[j ++ ];

	for (int i = l, j = 0; i <= r; i ++ , j ++ ) a[i] = tmp[j];
	return res;
}

int main() {
	while (1) {
	    int n; scanf("%d", &n);
		if (!n) break;
		for (int i = 0; i < n; i ++ ) scanf("%d", &a[i]);

		printf("%lld\n", merge(0, n - 1));
	}
	return 0;
}

