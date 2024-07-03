#include <cstdio>
#include <algorithm>

using namespace std;

const int N = 5010;

int r, t, m, n;
int g[N][N];

int main() {
    scanf("%d%d", &t, &r);
    r = min(5001, r);  // r <= 5000
    m = n = r;
    while (t -- ) {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        n = max(n, a + 1), m = max(m, b + 1);
        g[a + 1][b + 1] += c;
    }
    
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= m; j ++ )
            g[i][j] += g[i - 1][j] + g[i][j - 1] - g[i - 1][j - 1];
    
    int res = 0;
    for (int i = r; i <= n; i ++ )
        for (int j = r; j <= m; j ++ )
            res = max(res, g[i][j] - g[i - r][j] - g[i][j - r] + g[i - r][j - r]);

    printf("%d", res);
    return 0;
}
