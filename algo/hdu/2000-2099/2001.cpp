#include <cstdio>
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace std;


double calc(double x1, double y1, double x2, double y2) {
	double dx = x2 - x1;
	double dy = y2 - y1;
	return sqrt(dx * dx + dy * dy);
}

int main() {
	double x1, x2, y1, y2;
	while (cin >> x1 >> y1 >> x2 >> y2) {
		cout << calc(x1, y1, x2, y2) << endl;
	}
	return 0;
}
