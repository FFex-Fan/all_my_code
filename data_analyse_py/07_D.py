def floyd(n):
    for i in range(20):
        for j in range(20):
            print(val[i][j], end='\t')
        print()
    dis = [[inf for i in range(maxn + 1)] for j in range(maxn + 1)]  # 最短路矩阵
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            dis[i][j] = val[i][j]  # 初始化最短路矩阵
    ans = inf
    for k in range(1, n + 1):
        for i in range(1, k):
            for j in range(1, i):
                ans = min(ans, dis[i][j] + val[i][k] + val[k][j])  # 更新答案
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                dis[i][j] = min(dis[i][j], dis[i][k] + dis[k][j])
    return ans


if __name__ == '__main__':
    n = 20
    maxn = 50
    inf = 10000000
    val = [[inf for i in range(maxn + 1)] for j in range(maxn + 1)]  # 原图的邻接矩阵
    for i in range(28):
        inp = input().split(' ')
        a = int(inp[0])
        b = int(inp[1])
        c = int(inp[2])
        val[a][b] = c
        val[b][a] = c

    print(floyd(n))
