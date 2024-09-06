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


g, tasks = read_file('data/附件1/8x8map.txt')  # 读取地图和任务


def is_valid(pos):
    return 0 <= pos[0] < len(g) and 0 <= pos[1] < len(g[0]) and g[pos[0]][pos[1]] != '@'


def manhattan_distance(pos1, pos2):  # 启发函数
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


# 定义节点类
class ANode:
    def __init__(self, position, g, h, parent=None):
        self.position = position  # 节点的位置，元组形式 (x, y)
        self.g = g  # 从起点到当前节点的代价
        self.h = h  # 从当前节点到终点的估算代价（启发式函数）
        self.f = g + h  # 总代价
        self.parent = parent  # 父节点，用于回溯路径

    def __lt__(self, other):
        return self.f < other.f


def a_star(agent, start, end, constraints):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 定义四个方向
    heap = []
    vis = set()


    # 在所有冲突中过滤出和当前机器人有关的冲突
    tmp_constraints = [constraint for constraint in constraints if constraint[0] == agent]

    start_node = ANode(start, 0, manhattan_distance(start, end))
    end_node = ANode(end, 0, 0)

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
                neighbour_node = ANode(neighbour_pos, g, h, cur_node)

                heapq.heappush(heap, neighbour_node)
    return None


class Node:
    def __init__(self, constraints, solution, cost):
        self.constraints = constraints
        self.solution = solution
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost


def find_conflict(solution):
    pass


def cbs(mapf_instance):
    all_constraints = []
    all_solution = []
    all_cost = 0

    for i in range(len(mapf_instance["agents"])):
        start, end = mapf_instance["agents"][i]
        cur_path, cur_cost = a_star(i, start, end, all_constraints)

        all_solution.append(cur_cost)
        all_cost += cur_cost

    root = Node(all_constraints, all_solution, all_cost)

    open_list = [] # 定义一个 priority_queue
    heapq.heappush(open_list, root)

    while open_list is not Node:
        p = heapq.heappop(open_list)

        cur_conflict = find_conflict(p.solution)

        if cur_conflict is None:
            return p.solution

        (agent1, agent2, position, time) = cur_conflict
        for agent in [agent1, agent2]:
            new_constraints = p.constraints.copy()
            new_constraints.append((agent, position, time))

            new_solution = p.solution.copy()
            new_path, new_cost = a_star(agent, mapf_instance["agents"][agent][0], mapf_instance["agents"][agent][1],
                                        new_constraints)

            new_solution[agent] = new_path
            new_all_cost = sum(len(path) for path in new_solution)
            new_node = Node(new_constraints, new_solution, new_all_cost)
            heapq.heappush(open_list, new_node)
    return None



if __name__ == '__main__':
    mapf_instance = {
        "agents": [((0, 0), (4, 4)), ((1, 1), (5, 5))]
    }

    solution = cbs(mapf_instance)
