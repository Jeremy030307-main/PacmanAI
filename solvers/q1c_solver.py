#---------------------#
# DO NOT MODIFY BEGIN #
#---------------------#

import logging

import util
from problems.q1c_problem import q1c_problem

#-------------------#
# DO NOT MODIFY END #
#-------------------#

from problems.q1b_problem import q1b_problem
from solvers.q1b_solver import q1b_solver
from game import Directions
import heapq as hq
import util
import time
from game import Grid

# def q1c_solver(problem: q1c_problem):
#     timeout = 10
#     astarData = astar_initialise(problem)
#     num_expansions = 0
#     terminate = False
#     start_time = time.time()
#     while not terminate:
#         num_expansions += 1
#         terminate, result = astar_loop_body(problem, astarData, timeout, start_time)
#     print(f'Number of node expansions: {num_expansions}')
#     return result

# class Node:

#     def __init__(self, state: tuple[tuple[int, int],tuple[tuple[int, int]]]) -> None:

#         self.state = state
#         self.position = state[0]
#         self.food_remaining = state[1]
#         self.g: int = float('inf')
#         self.f: int = float('inf')
#         self.parent: Node = None
#         self.actionTaken: Directions = None
#         self.visited = False

#     def __eq__(self, value: 'Node') -> bool:
#         return (self.f, self.state) == (value.f, value.state)

#     def __ne__(self, value: 'Node') -> bool:
#         return (self.f, self.state) != (value.f, value.state)

#     def __lt__(self, value: 'Node'): 
#         return (self.f, self.state) < (value.f, value.state)
    
#     def __gt__(self, value: 'Node'): 
#         return (self.f, self.state) > (value.f, value.state)
    
#     def __ge__(self, value: 'Node'): 
#         return (self.f, self.state) >= (value.f, value.state)
    
#     def __le__(self, value: 'Node'): 
#         return (self.f, self.state) <= (value.f, value.state)

# class AStarData:
#     # YOUR CODE HERE
#     def __init__(self):
#         self.total_food = 0
#         self.open_list: list[Node] = []
#         self.nodes: dict[tuple, Node] = {}
#         self.terminate = False
#         self.treshold = None
#         self.visited:list[Node] = []

# def astar_initialise(problem: q1c_problem):

#     # YOUR CODE HERE
#     astarData = AStarData()

#     # get the starting position of the pacman, and initialise the starting node for the position
#     start_state = problem.getStartState()

#     # run a dfs on the entire maze to identify unreachable food dot
#     visited_food_grid = dfs(start_state[0], problem.foods.deepCopy(), problem.walls)
#     valid_food_list = set(start_state[1]) - set(visited_food_grid)
#     start_state = (start_state[0], tuple(valid_food_list))

#     # update the initial food dot available on the maze
#     astarData.total_food = len(valid_food_list)

#     # create the start node
#     start_node = Node(start_state)
#     start_node.g = 0
#     start_node.f = heuristic(start_state, problem)
#     start_node.parent = start_node
#     astarData.nodes[start_state] = start_node

#     # set the initial threshold value as 0
#     astarData.treshold = 0

#     hq.heappush(astarData.open_list, start_node)
#     hq.heappush(astarData.visited, start_node)

#     return astarData    

# def astar_loop_body(problem: q1c_problem, astarData: AStarData, timeout ,start_time):

#     lowest_visited_node: Node = astarData.visited.pop(0)
#     astarData.treshold = (astarData.total_food - len(lowest_visited_node.food_remaining)) * 10 - lowest_visited_node.g
#     hq.heappush(astarData.open_list, lowest_visited_node)

#     while len(astarData.open_list) > 0:
#         elapsed_time = time.time() - start_time

#         # get the node with the lower f_value                
#         hq.heapify(astarData.open_list)
#         current_node = astarData.open_list.pop(0)

#         # check if the current position is the goal state
#         if problem.isGoalState((current_node.position, current_node.food_remaining)) or elapsed_time > timeout - 0.1:
#             actions = action_reconstruct(astarData, current_node)
#             astarData.terminate = True
#             return astarData.terminate, actions
        
#         if current_node.visited:
#             continue

#         current_node.visited = True

#         for next_state, action, cost in problem.getSuccessors((current_node.position, current_node.food_remaining)):

#             # computer the new g_value, h_value and f_value
#             new_g = current_node.g + cost
#             new_h = heuristic(next_state, problem)
#             new_f = new_g + new_h

#             # First check: is the successor already have a node
#             next_state_node = astarData.nodes.get(next_state)

#             # update the h_value, it parent node and the action from parent node to this node (also known as node redirection)
#             if (next_state_node is None) or astarData.nodes[next_state].f > new_f:

#                 next_state_node = Node(next_state)
#                 astarData.nodes[next_state] = next_state_node

#                 next_state_node.f = new_f
#                 next_state_node.g = new_g
#                 next_state_node.actionTaken = action
#                 next_state_node.parent = current_node

#                 if next_state_node.f <= astarData.treshold:
#                     hq.heappush(astarData.open_list, next_state_node)
#                 else:
#                     hq.heappush(astarData.visited, next_state_node)

#     return astarData.terminate, []

# def heuristic(state, problem: q1c_problem):

#     pacmanPosition, remaining_food = state
    
#     if not remaining_food:
#         return 0
    
#     min_dist = float('inf')
#     for food_index in remaining_food:
#         x = util.manhattanDistance(pacmanPosition, food_index)
#         min_dist = min(min_dist, x)
   
#     return + min_dist + len(remaining_food) * 5

# def action_reconstruct(astarData: AStarData, destination_node: Node):
#     action = []

#     state_node: Node = destination_node
#     while state_node.parent != state_node:
#         action.append(state_node.actionTaken)
#         state_node = state_node.parent

#     action.reverse()
#     return action

def dfs(start_pos: tuple[int, int], foodGrid: Grid, wallGrid):
    height = foodGrid.height
    width = foodGrid.width

    visited = [[False for _ in range(height)] for _ in range(width)]
    stack = [start_pos]

    while stack and len(foodGrid.asList()) != 0:
        print("fsdfdsf")
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

def q1c_solver(problem: q1c_problem):

    start_state = problem.getStartState()   
    visited_food_grid = dfs(start_state[0], problem.foods.deepCopy(), problem.walls)
    valid_food_list = set(start_state[1]) - set(visited_food_grid)

    total_path = []
    problem_b = q1b_problem(problem.startingGameState)
    problem_b.goalPoints = valid_food_list

    while problem_b.goalPoints:
        total_path += q1b_solver(problem_b)

    return total_path