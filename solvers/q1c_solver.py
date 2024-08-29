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


def q1c_solver(problem: q1c_problem):

    global shortest_pairs
    shortest_pairs = allPairShortest(problem.walls)
    startState = problem.getStartState()

    # check of uneatable dot
    food_pacman = list(startState[1])
    food_pacman.insert(0,startState[0])
    print(food_pacman)
    _, visited_food = mst(food_pacman, problem.walls.height)

    new_food_remaining = tuple([startState[1][i] for i in range(len(startState[1])) if visited_food[i+1]])
    print(startState[1])
    startState = (startState[0], new_food_remaining)
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

def heuristic(state, problem: q1c_problem):

    global total_time
    global shortest_pairs
    pacmanPosition, remaining_food = state
    
    if not remaining_food:
        return 0
    
    x = time.time()
    heuristic = 0
    # construct mst of the remaining dots
    # heuristic = mst(remaining_food, problem.walls.height)

    # Calculate the minimum distance to the closest dot
    min_dist = float('inf')
    start = cell_to_node(pacmanPosition[0]-1, pacmanPosition[1]-1, problem.walls.height-2)
    for food_index in remaining_food:
        end = cell_to_node(food_index[0]-1, food_index[1]-1, problem.walls.height-2)
        min_dist = min(min_dist, shortest_pairs[start][end])
   
    y = time.time()
    print(y-x)
    heuristic += min_dist  
    # return  heuristic + len(remaining_food) * ((problem.walls.height + problem.walls.width) / 2)
    return heuristic + len(remaining_food) * 5


def reconstruct_path(parent_map, current):
    path = []
    while current in parent_map:
        current, action = parent_map[current]
        path.append(action)
    path.reverse()
    return path

def cell_to_node(x, y, height):
        return x * height + y

def allPairShortest(wall_grid):
    
    x = time.time()
    INF = float('inf')
    n = wall_grid.width-2
    m = wall_grid.height-2

    # Initialize the distance matrix
    dist = [[INF] * (n * m) for _ in range(n * m)]

    for i in range(n):
        for j in range(m):
            if wall_grid[i+1][j+1]:
                continue
            node = cell_to_node(i,j, m)
            dist[node][node] = 0. # distance to itself is 0

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + dx, j + dy
                if 0 <= ni < n and 0 <= nj < m and not wall_grid[ni+1][nj+1]:
                    neighbor_node = cell_to_node(ni, nj, m)
                    dist[node][neighbor_node] = 1  # Distance between adjacent cells is 1

    # Floyd-Warshall algorithm
    total_nodes = n * m
    for k in range(total_nodes):
        for i in range(total_nodes):
            for j in range(total_nodes):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    y = time.time()
    print(y-x)
    return dist

def mst(food_list: list[tuple[int]], height):

    global shortest_pairs

    visited = [False for _ in range(len(food_list))]
    total_node = len(food_list)
    num_visited_node = 0
    cost = 0

    while (num_visited_node < total_node):
        # For every vertex in the set S, find the all adjacent vertices
        #, calculate the distance from the vertex selected at step 1.
        # if the vertex is already in the set S, discard it otherwise
        # choose another vertex nearest to selected vertex  at step 1.
        minimum = float('inf')
        x = 0
        y = 0
        for i in range(total_node):
            if visited[i]:
                for j in range(total_node):
                    start = cell_to_node(food_list[i][0]-1, food_list[i][1]-1, height-2)
                    end = cell_to_node(food_list[j][0]-1, food_list[j][1]-1, height-2)
                    if ((not visited[j]) and shortest_pairs[start][end] and shortest_pairs[start][end] != float("inf")): 
                        # not in selected and there is an edge
                        if minimum > shortest_pairs[start][end]:
                            minimum = shortest_pairs[start][end]
                            x = i
                            y = j

        start = cell_to_node(food_list[x][0]-1, food_list[x][1]-1, height-2)
        end = cell_to_node(food_list[y][0]-1, food_list[y][1]-1, height-2)
        cost += shortest_pairs[start][end]
        visited[y] = True
        num_visited_node += 1

    return cost, visited
