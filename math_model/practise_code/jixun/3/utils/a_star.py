# coding=utf-8
import heapq


def read_file(file):
    with open(file, 'r') as f:  # 打开文件
        lines = f.readlines()  # 读取所有行

    # 获取地图尺寸
    n, m = lines[1].split(" ")
    n, m = int(n), int(m)

    g = []
    for i in range(2, n + 2):  # 读取地图
        g.append(lines[i].strip().split())

    num_task = int(lines[n + 3])
    tasks = []
    for i in range(n + 4, n + 4 + num_task):  # 读取任务
        sx, sy, ex, ey = map(int, lines[i].split())
        tasks.append((sx, sy, ex, ey))

    return g, tasks


g, tasks = read_file(
    '/Users/ffex/Developer/all_code/math_model/practise_code/jixun/3/data/附件1/16x16map.txt')  # 读取地图和任务


def is_valid(pos):
    return 0 <= pos[0] < len(g) and 0 <= pos[1] < len(g[0]) and g[pos[0]][pos[1]] != '@'


def manhattan_distance(pos1, pos2):  # 启发函数
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


# 定义节点类
class Node:
    def __init__(self, position, g, h, parent=None):
        self.position = position  # 节点的位置，元组形式 (x, y)
        self.g = g  # 从起点到当前节点的代价
        self.h = h  # 从当前节点到终点的估算代价（启发式函数）
        self.f = g + h  # 总代价
        self.parent = parent  # 父节点，用于回溯路径

    def __lt__(self, other):
        return self.f < other.f


def a_star(start, end):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 定义四个方向
    heap = []
    vis = set()

    start_node = Node(start, 0, manhattan_distance(start, end))
    end_node = Node(end, 0, 0)

    heapq.heappush(heap, start_node)

    while heap:
        cur_node = heapq.heappop(heap)

        if cur_node.position == end_node.position:
            path = []
            while cur_node:
                path.append(cur_node.position)
                cur_node = cur_node.parent  # 回溯路径
            return path[::-1]

        vis.add(cur_node.position)

        for dx, dy in directions:
            neighbour_pos = (cur_node.position[0] + dx, cur_node.position[1] + dy)
            if is_valid(neighbour_pos) and neighbour_pos not in vis:
                if any(item.position == neighbour_pos and item.f <= neighbour_node.f for item in heap):
                    continue

                # 计算邻居的g值、h值和f值
                g = cur_node.g + 1
                h = manhattan_distance(neighbour_pos, end)
                neighbour_node = Node(neighbour_pos, g, h, cur_node)

                heapq.heappush(heap, neighbour_node)
    return None


paths = [a_star((task[0], task[1]), (task[2], task[3])) for task in tasks]


print(g)
print(tasks)

for i in paths:
    print(i)

'''
[(2, 7), (1, 7), (1, 6)]
[(10, 1), (9, 1), (8, 1), (7, 1), (7, 2), (7, 3), (7, 4), (6, 4), (5, 4), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (5, 8), (6, 8), (7, 8), (7, 9)]
[(1, 4), (0, 4), (0, 3), (0, 2), (0, 1), (1, 1)]
[(7, 6), (6, 6), (6, 5), (6, 4), (5, 4)]
[(7, 2), (7, 3), (7, 4), (6, 4), (6, 5)]
[(6, 6), (6, 5), (6, 4), (5, 4), (4, 4), (4, 5), (4, 6), (4, 7), (3, 7), (3, 8), (3, 9), (3, 10), (2, 10), (2, 11), (2, 12), (1, 12)]
[(12, 11), (12, 10), (12, 9), (11, 9), (10, 9), (10, 10), (9, 10), (8, 10), (7, 10), (7, 9), (7, 8), (6, 8), (5, 8), (4, 8), (4, 7), (4, 6)]
[(3, 11), (3, 10), (3, 9), (3, 8), (4, 8), (4, 7), (4, 6), (4, 5)]
[(14, 6), (15, 6), (15, 5), (15, 4), (14, 4), (13, 4), (13, 3), (13, 2), (12, 2), (11, 2), (10, 2), (10, 3), (9, 3), (8, 3), (7, 3), (7, 4), (7, 5), (7, 6)]
[(4, 5), (4, 4), (5, 4), (5, 3), (5, 2), (5, 1), (4, 1), (4, 0), (3, 0), (2, 0), (2, 1), (2, 2), (3, 2)]
[(3, 15), (4, 15), (4, 14), (4, 13), (4, 12), (5, 12), (5, 11), (6, 11), (7, 11), (8, 11), (8, 10), (9, 10), (10, 10), (10, 9), (11, 9), (12, 9), (12, 8), (12, 7), (11, 7), (10, 7), (9, 7), (9, 8)]
[(14, 14), (13, 14), (13, 13), (13, 12), (12, 12), (11, 12), (10, 12), (9, 12), (9, 11), (8, 11), (7, 11), (6, 11)]
'''
