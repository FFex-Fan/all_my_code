import heapq


def read_file(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    # 获取地图尺寸
    n, m = lines[1].split(" ")
    n, m = int(n), int(m)

    g = []
    for i in range(2, n + 2):
        g.append(lines[i].strip().split())

    num_task = int(lines[n + 3])
    tasks = []
    for i in range(n + 4, n + 4 + num_task):
        sx, sy, ex, ey = map(int, lines[i].split())
        tasks.append((sx, sy, ex, ey))

    return g, tasks

g, tasks = read_file('../data/附件1/8x8map.txt')

# 检查位置有效性
def is_valid(pos):
    return 0 <= pos[0] < len(g) and 0 <= pos[1] < len(g[0]) and g[pos[0]][pos[1]] != '@'

# 曼哈顿距离作为启发式函数
def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# 节点类
class Node:
    def __init__(self, position, g, h, parent=None, time=0):
        self.position = position
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent
        self.time = time

    def __lt__(self, other):
        return self.f < other.f

# A* 搜索算法
def a_star(start, end, constraints):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    heap = []
    vis = set()
    start_node = Node(start, 0, manhattan_distance(start, end))
    heapq.heappush(heap, start_node)

    while heap:
        cur_node = heapq.heappop(heap)

        if cur_node.position == end:
            path = []
            while cur_node:
                path.append(cur_node.position)
                cur_node = cur_node.parent
            return path[::-1]

        vis.add((cur_node.position, cur_node.time))

        for dx, dy in directions:
            next_time = cur_node.time + 1
            neighbour_pos = (cur_node.position[0] + dx, cur_node.position[1] + dy)
            if is_valid(neighbour_pos) and (neighbour_pos, next_time) not in vis:
                if (cur_node.position, neighbour_pos, next_time) in constraints or (neighbour_pos, cur_node.position, next_time) in constraints:
                    continue
                g = cur_node.g + 1
                h = manhattan_distance(neighbour_pos, end)
                neighbour_node = Node(neighbour_pos, g, h, cur_node, next_time)
                heapq.heappush(heap, neighbour_node)
    return None

# 约束基搜索 (CBS) 算法，用于解决问题1
def cbs_problem1(g, tasks):
    paths = []
    constraints = set()

    for task in tasks:
        start = (task[0], task[1])
        end = (task[2], task[3])
        path = a_star(start, end, constraints)
        if not path:
            print("无法找到初始路径")
            return None

        conflict_found = False
        for i, existing_path in enumerate(paths):
            for t in range(min(len(path), len(existing_path))):
                if path[t] == existing_path[t]:  # 位置冲突
                    conflict_found = True
                    constraints.add((path[t-1], path[t], t))  # 添加冲突的边约束
                    break
                if t > 0 and path[t] == existing_path[t-1] and path[t-1] == existing_path[t]:  # 边冲突
                    conflict_found = True
                    constraints.add((path[t], path[t-1], t))  # 添加相反方向的边约束
                    break
            if conflict_found:
                break

        if conflict_found:
            path = a_star(start, end, constraints)
            if not path:
                print("无法找到冲突解决后的路径")
                return None

        paths.append(path)

    return paths

paths = cbs_problem1(g, tasks)
if paths:
    for path in paths:
        print(path)


# 检验是否有冲突边
def extract_edges_with_time(path):
    edges = []  # 用于存储路径中的边和时间步
    for k in range(len(path) - 1):
        edge = (path[k], path[k + 1])
        time_step = k  # 使用索引作为时间步
        edges.append((edge, time_step))
    return edges

def check_conflict_edges(columns):
    conflicts = []  # 用于存储冲突边的信息
    # 遍历每一对路径，比较它们之间的边
    for i, path1 in enumerate(columns):
        edges_path1 = extract_edges_with_time(path1)

        for j, path2 in enumerate(columns):
            if i >= j:  # 跳过自身比较以及重复比较
                continue

            edges_path2 = extract_edges_with_time(path2)

            # 比较两条路径的所有边，检查是否有冲突边
            for edge1, time1 in edges_path1:
                for edge2, time2 in edges_path2:
                    # 检查是否有相同的边或反向的边，并且发生在相同时间步
                    if (edge1 == edge2 or edge1 == (edge2[1], edge2[0])) and time1 == time2:
                        conflicts.append((i + 1, j + 1, edge1, time1))
    return conflicts is not None, conflicts

# print()
# print()
#
# # 验证是否有冲突点
# columns = sorted(paths, key=lambda x: len(x))
# for i in columns:
#     print(i)
# flag = False
# point = {}
# for i, col in enumerate(columns):
#     if point.get(i) is None:
#         point[i] = []
#     if col not in point[i]:
#         point[i].append(col)
#     else:
#         flag = True
# print(flag)
#
# _, con = check_conflict_edges(paths)
# print("存在冲突边：", con)
