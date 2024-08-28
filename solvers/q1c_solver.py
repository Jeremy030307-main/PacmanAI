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

def q1c_solver(problem: q1c_problem):
    
    shortest_pairs = allPairShortest(problem.walls)

    start_pos = problem.getStartState()
    keyPoint = [start_pos] + problem.food 
    dist = [[0 for _ in range(len(keyPoint))] for _ in range(len(keyPoint))]
    visited = [False for _ in range(len(keyPoint))]
    total_node = len(keyPoint)
    num_visited_node = 0

    for i in range(len(keyPoint)):
        for j in range(len(keyPoint)):
            dist[i][j] = shortest_pairs[cell_to_node(keyPoint[i][0],keyPoint[i][1],problem.walls.height)][cell_to_node(keyPoint[j][0],keyPoint[j][1],problem.walls.height)]

    path = []
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
                    if ((not visited[j]) and dist[i][j]):  
                        # not in selected and there is an edge
                        if minimum > dist[i][j]:
                            minimum = dist[i][j]
                            x = i
                            y = j
        
        if x not in path:
            path.append(x)
        
        if y not in path:
            path.append(y)

        visited[y] = True
        num_visited_node += 1

    print(path, total_node)

    # get the actions
    whole_actions = []
    for i in range(len(path[1:])):
        position_prob = q1a_problem(problem.startingGameState)
        position_prob.start_pos = keyPoint[path[i]]
        position_prob.goalPoint = keyPoint[path[i+1]]
        whole_actions += q1a_solver(position_prob)
    
    print(whole_actions)

    return whole_actions

def cell_to_node(x, y, height):
        return x * height + y

def allPairShortest(wall_grid):
     
    INF = float('inf')

    n = wall_grid.width
    m = wall_grid.height

    # Initialize the distance matrix
    dist = [[INF] * (n * m) for _ in range(n * m)]

    for i in range(n):
        for j in range(m):
            if wall_grid[i][j]:
                continue
            node = cell_to_node(i,j, m)
            dist[node][node] = 0. # distance to itself is 0

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + dx, j + dy
                if 0 <= ni < n and 0 <= nj < m and not wall_grid[ni][nj]:
                    neighbor_node = cell_to_node(ni, nj, m)
                    dist[node][neighbor_node] = 1  # Distance between adjacent cells is 1

    # Floyd-Warshall algorithm
    total_nodes = n * m
    for k in range(total_nodes):
        for i in range(total_nodes):
            for j in range(total_nodes):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    return dist


# # Displaying the shortest distance between the start 'S' and finish 'F'
# start = (0, 0)
# finish = (2, 2)
# start_node = start[0] * len(maze[0]) + start[1]
# finish_node = finish[0] * len(maze[0]) + finish[1]

# print(f"Shortest distance from 'S' to 'F': {distances[start_node][finish_node]}")
# for i in distances:
#     print(i)