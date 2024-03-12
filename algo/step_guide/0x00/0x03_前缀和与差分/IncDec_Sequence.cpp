#include <cstdio>
#include <iostream>
#include <cmath>

using namespace std;

typedef long long ll;

const int N = 100010;

int a[N];
int n;

int main() {
	scanf("%d", &n);
	for (int i = 1; i <= n; i ++ ) scanf("%d", &a[i]);
	for (int i = n; i > 1; i -- ) a[i] -= a[i - 1]; 
	ll pos = 0, neg = 0;
	for (int i = 2; i <= n; i ++ ) {  // 从 2 开始
		if (a[i] > 0) pos += a[i] ;
		else neg -= a[i] ;
	}
	
	printf("%lld\n%lld", min(pos, neg) + abs(pos - neg), abs(pos - neg) + 1);
	return 0;
}
