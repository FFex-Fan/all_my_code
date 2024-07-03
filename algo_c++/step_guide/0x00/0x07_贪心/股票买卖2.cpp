#include <cstdio>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 100010;

int n;
int a[N];

int main() {
	cin >> n;
	for (int i = 0; i < n; i ++ ) cin >> a[i];
	
	int res = 0;
	for (int i = 0; i < n - 1; i ++ ) 
		if (a[i + 1] > a[i]) res += a[i + 1] - a[i];

	cout << res << endl;
	return 0;
}
