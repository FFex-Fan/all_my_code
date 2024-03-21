#include <cstdio>
#include <iostream>
#include <algorithm>
#include <cstring>

using namespace std;

const int N = 1000010;

int stk_l[N], l;
int stk_r[N], r;

int sum[N];
int f[N];
int m;

/* 
 * 在序列中间某个指定位置进行修改，可以通过两个线性结构来维护该位置
 * 光标左侧的存储在 stk_l 栈中， 右侧存储在栈 stk_r 中
 * sum 记录前缀和
 * f 记录最大值
 */

int main() {
    f[0] = -1e9;  // 设置哨兵
	cin >> m;
	string op = "";
	int x;
	while (m -- ) {
		cin >> op;
		if (op == "I") {
			cin >> x;
			stk_l[ ++ l] = x;
			sum[l] = sum[l - 1] + stk_l[l];
			f[l] = max(f[l - 1], sum[l]);
		}
		else if (op == "D") {
			if (l) l -- ;
		}
		else if (op == "L") {
			if (l) stk_r[ ++ r] = stk_l[l -- ];
		}
		else if (op == "R") {
			if (r) {
				stk_l[ ++ l] = stk_r[r -- ];
				sum[l] = sum[l - 1] + stk_l[l];
				f[l] = max(f[l - 1], sum[l]);
			}
		}
		else {
			cin >> x;
			cout << f[x] << endl;
		}
	}
	return 0;
}
