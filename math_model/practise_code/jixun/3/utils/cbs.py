import heapq
from collections import defaultdict

# 定义Node类，用于表示CBS算法中的节点
class Node:
    def __init__(self, constraints, paths, cost):
        self.constraints = constraints  # 该节点的约束条件
        self.paths = paths  # 当前所有机器人的路径集合
        self.cost = cost  # 该节点的总代价（所有路径的长度总和）

    # 定义小于操作符，用于优先队列中的节点排序
    def __lt__(self, other):
        return self.cost < other.cost

# 检查两个路径之间是否有冲突
def is_conflict(path1, path2):
    for t in range(max(len(path1), len(path2))):
        # 获取两个机器人在时间t的位置信息
        loc1 = path1[t] if t < len(path1) else path1[-1]  # 如果时间t超过路径长度，则保持在最后一个位置
        loc2 = path2[t] if t < len(path2) else path2[-1]
        # 如果两个机器人在同一时间点占据同一位置，则发生了冲突
        if loc1 == loc2:
            return True, (loc1, t)
        # 如果两个机器人在同一时间点交换了位置，也视为冲突
        if t > 0 and path1[t-1] == loc2 and path2[t-1] == loc1:
            return True, (loc1, loc2, t)
    return False, None

# A*算法，用于在指定的约束条件下为单个机器人找到从起点到终点的最优路径
def a_star(start, goal, grid, constraints):
    open_list = [(0, start, [])]  # 优先队列，存储待扩展节点（代价、当前位置、路径）
    closed_list = set()  # 存储已经访问过的节点，避免重复访问
    while open_list:
        cost, current, path = heapq.heappop(open_list)  # 从优先队列中取出代价最低的节点
        if current == goal:
            return path + [current]  # 如果到达目标，返回路径
        if (current, len(path)) in closed_list:
            continue  # 如果该状态已经访问过，跳过
        closed_list.add((current, len(path)))  # 标记该状态为已访问
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 检查四个相邻位置
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]) and grid[neighbor[0]][neighbor[1]] == 0:
                # 检查相邻位置是否在网格范围内且没有障碍物
                if len(path)+1 not in constraints or neighbor not in constraints[len(path)+1]:
                    # 检查该位置是否违反了约束条件
                    heapq.heappush(open_list, (cost + 1, neighbor, path + [current]))
                    # 将新状态加入优先队列
    return None  # 如果找不到路径，返回None

# CBS算法的主函数，用于找到所有机器人的无冲突路径
def find_solution(grid, starts, goals):
    root = Node([], [], 0)  # 创建根节点，无约束，初始代价为0
    for start, goal in zip(starts, goals):
        path = a_star(start, goal, grid, defaultdict(set))  # 为每个机器人计算初始路径
        if not path:
            return None  # 如果有任何一个机器人无法找到路径，返回None
        root.paths.append(path)  # 将路径加入根节点
        root.cost += len(path) - 1  # 更新总代价

    open_list = [root]  # 创建优先队列并加入根节点
    while open_list:
        node = heapq.heappop(open_list)  # 从优先队列中取出代价最低的节点
        conflict_found = False
        for i in range(len(node.paths)):
            for j in range(i+1, len(node.paths)):
                conflict, details = is_conflict(node.paths[i], node.paths[j])
                if conflict:
                    conflict_found = True
                    c1, c2 = details
                    # 生成两个子问题，分别为两个机器人增加冲突约束
                    constraints1 = node.constraints + [(i, c1)]
                    constraints2 = node.constraints + [(j, c2)]
                    paths1 = node.paths[:]
                    paths2 = node.paths[:]
                    paths1[i] = a_star(starts[i], goals[i], grid, defaultdict(set, {t: {loc} for _, (loc, t) in constraints1}))
                    paths2[j] = a_star(starts[j], goals[j], grid, defaultdict(set, {t: {loc} for _, (loc, t) in constraints2}))
                    # 检查是否可以为两个机器人找到新的无冲突路径
                    if paths1[i] and paths2[j]:
                        heapq.heappush(open_list, Node(constraints1, paths1, sum(len(p) for p in paths1)))
                        heapq.heappush(open_list, Node(constraints2, paths2, sum(len(p) for p in paths2)))
                    break
            if conflict_found:
                break
        if not conflict_found:
            return node.paths  # 如果没有冲突，返回所有机器人的路径
    return None  # 如果无法找到无冲突的解决方案，返回None

# 示例网格
grid = [
    [0, 1, 0, 0, 0],  # 0表示可行位置，1表示障碍物
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]
# 机器人起点和目标点
starts = [(0, 0), (4, 0)]
goals = [(4, 4), (0, 4)]

# 计算解决方案
solution = find_solution(grid, starts, goals)
if solution:
    for i, path in enumerate(solution):
        print(f"Agent {i+1}: {path}")  # 输出每个机器人的路径
else:
    print("No solution found")  # 如果无法找到解决方案，输出提示
