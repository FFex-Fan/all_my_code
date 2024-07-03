#include <cstdio>
#include <iostream>
#include <algorithm>
#include <cstring>

using namespace std;

string s;
char a[3];

int main() {
	while (getline(cin, s)) {
		a[0] = s[0], a[1] = s[1], a[2] = s[2];
		sort(a, a + 3);
		for (int i = 0; i < 3; i ++ )
		cout << a[i] << ' ';
		cout << endl;
	}
	return 0;
}
