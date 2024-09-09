import gurobipy as gp
from gurobipy import GRB
import numpy as np


file_path = "data/附件1/8x8map.txt"

with open(file_path, "r") as f:
    lines = f.readlines()

map_size = lines[1].strip().split()
map_width = int(map_size[1])
map_height = int(map_size[0])

print(f'Map size: {map_height} x {map_width}')

obstacles = set()
for i in range(map_height):
    row = lines[2 + i].strip().split()
    for j in range(map_width):
        if row[j] == '@':
            obstacles.add((i, j))
print(f'Obstacles: {obstacles}')

num_robots = int(lines[map_height + 3].strip())
start_position = []
goal_position = []


for i in range(num_robots):
    task = list(map(int, lines[map_height + 4 + i].strip().split()))
    start_position.append((task[0], task[1]))
    goal_position.append((task[2], task[3]))

print(f'Start position: {start_position}')
print(f'Goal position: {goal_position}')


model = gp.Model("Robot_Path_Planning")

T = 122

x = {}
for i in range(num_robots):
    for t in range(T):
        for j in range(map_height):
            for k in range(map_width):
                if (j, k) in obstacles:
                    continue
                x[i, j, k, t] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}_{k}_{t}')

T_max = model.addVar(vtype=GRB.CONTINUOUS, name='T_max')





