#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>
#include <algorithm>

#define pb push_back

using namespace std;

const int N = 2e5 + 10;

int n, m;
int a[N], audio[N], word[N];
int cnt[N * 3];
vector<int> vi;

int query(int x) {
	return lower_bound(vi.begin(), vi.end(), x) - vi.begin();
}

int main() {
	ios::sync_with_stdio(false);
	cin.tie(0), cout.tie(0);
	
	cin >> n;
	for (int i = 1; i <= n; i ++ ) {
		cin >> a[i]; 
		vi.pb(a[i]);
	}

	cin >> m;
	for (int i = 1; i <= m; i ++ ) {
		cin >> audio[i];
		vi.pb(audio[i]);
	}

	for (int i = 1; i <= m; i ++ ) {
		cin >> word[i];
		vi.pb(word[i]);
	}
	
	// 离散化
	sort(vi.begin(), vi.end());
	vi.erase(unique(vi.begin(), vi.end()), vi.end());

	// 统计不同语言有多少科学家会
	for (int i = 1; i <= n; i ++ ) cnt[query(a[i])] ++ ;
	
	int res = 1, max_b = 0, max_c = 0;
	for (int i = 1; i <= m; i ++ ) {
		 // b: 会语音对应语言的人数
		 // c: 会字幕对应语言的人数
		int b = cnt[query(audio[i])], c = cnt[query(word[i])];
        
		if (b > max_b) res = i, max_b = b, max_c = c;
		else if (b == max_b && c > max_c) res = i, max_c = c;
	}
	
	cout << res << endl;
	return 0;
}

