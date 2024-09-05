def read_file(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    n, m = map(int, lines[1].split())
    g = [lines[i + 2].strip().split() for i in range(n)]

    num_task = int(lines[n + 3])
    tasks = [tuple(map(int, lines[i].split())) for i in range(n + 4, n + 4 + num_task)]

    return g, tasks


grid, tasks = read_file('data/附件1/16x16map.txt')

directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]


class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent


def is_vaild(nx, ny):
    return 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == '.'


def bfs(task):
    start = (task[0], task[1])
    end = (task[2], task[3])
    q = [Node(start)]
    visited = set(start)
    while q:
        current = q.pop(0)
        if current.position == end:  # 回溯路径
            path = [current.position]
            while current.parent:
                path.append(current.parent.position)
                current = current.parent
            return path[::-1]
        for dx, dy in directions:
            nx, ny = current.position[0] + dx, current.position[1] + dy
            new_position = (nx, ny)
            if is_vaild(nx, ny) and new_position not in visited:
                visited.add(new_position)
                q.append(Node(new_position, current))
    return None


path = [bfs(task) for task in tasks]
for p in path:
    print(p)