#---------------------#
# DO NOT MODIFY BEGIN #
#---------------------#

import logging

import util
from problems.q1c_problem import q1c_problem

#-------------------#
# DO NOT MODIFY END #
#-------------------#

from problems.q1a_problem import q1a_problem
from solvers.q1a_solver import q1a_solver
from game import Directions
import heapq as hq
import util
import time
from game import Grid


def q1c_solver(problem: q1c_problem):

    global shortest_pairs
    # shortest_pairs = allPairShortest(problem.walls)
    startState = problem.getStartState()

    # test = dfs(startState[0], problem.foods, problem.walls)
    visited_food_grid = dfs(startState[0], problem.foods, problem.walls)
    new_food_list = set(startState[1]) - set(visited_food_grid)
    startState = (startState[0], tuple(new_food_list))
    open_list = []
    closed_set = set()
    g_costs = {startState: 0}
    parent_map = {}

    hq.heappush(open_list, (heuristic(startState, problem), startState))
    
    while open_list:
    
        _, current = hq.heappop(open_list)
        
        if problem.isGoalState(current):
            return reconstruct_path(parent_map, current)
        
        if current in closed_set:
            continue
        
        closed_set.add(current)
        
        for successor, action, stepCost in problem.getSuccessors(current):
            tentative_g_cost = g_costs[current] + stepCost
            
            if successor not in g_costs or tentative_g_cost < g_costs[successor]:
                g_costs[successor] = tentative_g_cost
                priority = tentative_g_cost + heuristic(successor, problem)
                hq.heappush(open_list, (priority, successor))
                parent_map[successor] = (current, action)
    
    return []

total_time = 0

# def heuristic(state, problem: q1c_problem):

#     global total_time
#     global shortest_pairs
#     pacmanPosition, remaining_food = state
    
#     if not remaining_food:
#         return 0
    
#     x = time.time()
#     heuristic = 0

#     min_dist = float('inf')
#     start = cell_to_node(pacmanPosition[0], pacmanPosition[1], problem.walls.height)
#     for food_index in remaining_food:
#         end = cell_to_node(food_index[0], food_index[1], problem.walls.height)
#         x = util.manhattanDistance(pacmanPosition, food_index)
#         min_dist = min(min_dist, x)
   
#     return + min_dist + len(remaining_food) * 5

def cluster_food(remaining_food):
    # Simple clustering by proximity
    clusters = []
    for food in remaining_food:
        added = False
        for cluster in clusters:
            if any(util.manhattanDistance(food, c) < 3 for c in cluster):
                cluster.append(food)
                added = True
                break
        if not added:
            clusters.append([food])
    return clusters

def heuristic(state, problem):
    pacmanPosition, remaining_food = state

    if not remaining_food:
        return 0

    clusters = cluster_food(remaining_food)
    cluster_dists = [min(util.manhattanDistance(pacmanPosition, food) for food in cluster) for cluster in clusters]

    nearest_cluster_dist = min(cluster_dists)
    return nearest_cluster_dist + len(remaining_food) * 4

def reconstruct_path(parent_map, current):
    path = []
    while current in parent_map:
        current, action = parent_map[current]
        path.append(action)
    path.reverse()
    return path

def cell_to_node(x, y, height):
        return x * height + y

def dfs(start_pos: tuple[int, int], foodGrid: Grid, wallGrid):
    height = foodGrid.height
    width = foodGrid.width

    visited = [[False for _ in range(height)] for _ in range(width)]
    stack = [start_pos]

    while stack and len(foodGrid.asList()) != 0:
        x, y = stack.pop(0)

        if visited[x][y]:
            continue

        # Mark the current node as visited
        foodGrid[x][y] = False
        visited[x][y] = True

        all_directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]

        for direction in all_directions:
            next_x = x + direction[0]
            next_y = y + direction[1]

            if 0 <= next_x < width and 0 <= next_y < height and not visited[next_x][next_y] and not wallGrid[next_x][next_y]:
                stack.append((next_x, next_y))
    
    return foodGrid.asList()
