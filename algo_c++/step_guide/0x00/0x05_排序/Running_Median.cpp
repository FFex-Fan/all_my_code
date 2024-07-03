/**
 * 通过一个 大根堆 + 小根堆 来维护中位数
 */
#include <cstdio>
#include <algorithm>
#include <vector>
#include <queue>

using namespace std;

int n, m;
int no;


int main() {
	scanf("%d", &m);
	while (m -- ) {
		
		scanf("%d%d", &no, &n);
		printf("%d %d\n", no, n + 1 >> 1);
		
		priority_queue<int> down;
        priority_queue<int, vector<int>, greater<int> >  up; 
		
		int cnt = 0;
		for (int i = 0; i < n; i ++ ) {
			int c; scanf("%d", &c);
			
			
			// 比大根堆的最大值小就插入大根堆
			if (down.empty() || c <= down.top()) down.push(c);
			else up.push(c);  // 比大根堆的最大值大就插入小根堆中


            // 保证大根堆最多比小根堆多一个数据
			if (down.size() - 1 > up.size()) up.push(down.top()), down.pop();
			if (up.size() > down.size()) down.push(up.top()), up.pop();

			if (!(i & 1)) {
				printf("%d ", down.top());
				if ( ++ cnt % 10 == 0) puts("");
			}
		}
		if (cnt % 10) puts("");
	}
	return 0;
}
