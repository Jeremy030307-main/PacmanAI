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
from problems.q1a_problem import q1a_problem
from solvers.q1b_solver import q1b_solver
from solvers.q1a_solver import q1a_solver

from game import Directions, Actions
import util
from game import Grid
import time

def greedy_path(problem: q1c_problem):

    start_state = problem.getStartState()   

    action_paths = []
    paths: list[tuple[tuple[int,int], int]] = []
    problem_b = q1b_problem(problem.startingGameState)
    problem_b.goalPoints = start_state[1]
    
    while problem_b.goalPoints:

        start_point = problem_b.start_pos
        actions = q1b_solver(problem_b)
        paths.append((start_point, len(actions)))
        action_paths += actions

    paths.append((problem_b.start_pos, 0))
    return action_paths, paths

def two_opt_swap(action_path, path, v1, v2, gs):

    new_path = path[:]
    for i in range(v1+2,v2+1):
        new_path[i] = (new_path[i][0], path[i-1][1])
    new_path[v1+1:v2+1] = new_path[v2:v1:-1]

    break_point1 = [0,0]
    break_point2 = [0,0]
    for i in range(v2+1):

        if i < v1:
            break_point1[0] += path[i][1]
        elif i == v1:
            break_point1[1] = break_point1[0] + path[i][1]
        
        if i < v2:
            break_point2[0] += path[i][1]
        elif i == v2:
            break_point2[1] = break_point2[0] + path[i][1]

    distance_prob = q1a_problem(gs)
    distance_prob.start_pos = path[v1][0]
    distance_prob.goalPoint = path[v2][0]
    first_interval = q1a_solver(distance_prob)

    if v2 >= len(path)-1:
        second_interval = []
    else:
        distance_prob.start_pos = path[v1+1][0]
        distance_prob.goalPoint = path[v2+1][0]
        second_interval = q1a_solver(distance_prob)

    first_part = action_path[:break_point1[0]]
    middle_part: list = action_path[break_point2[0]-1: break_point1[1]-1:-1]
    for i in range(len(middle_part)):
        middle_part[i] = Directions.REVERSE[middle_part[i]]
    last_part = action_path[break_point2[1]:]

    new_action_path = first_part + first_interval + middle_part + second_interval + last_part
    new_path[v1] = (new_path[v1][0], len(first_interval))
    new_path[v2] = (new_path[v2][0], len(second_interval))
    
    return new_action_path, new_path

def walk_path(action_path, problem: q1c_problem):

    pacman_position, food_remaining = problem.getStartState()
    x,y = pacman_position
    distance = 0
    
    for action in action_path:
        dx,dy = Actions.directionToVector(action)
        x,y = int(x + dx), int(y + dy)
        distance += 1

        if (x,y) in food_remaining:
            food_remaining.remove((x,y))

            if len(food_remaining) <= 0:
                break
    
    return action_path[:distance]

def q1c_solver(problem: q1c_problem):

    start = time.time()

    action_paths, paths = greedy_path(problem)
    # return action_paths
    best_distance = len(action_paths)
    improvement = True

    while improvement:
        improvement = False
        for i in range(len(paths)):

            new_action, new_path = remove_a_dot(action_paths, paths, i, problem.startingGameState)
            walk_route = walk_path(new_action, problem)
            if (problem.unreachable and (best_distance - len(walk_route) >= 8)) or (not problem.unreachable and (best_distance - len(walk_route) > 100)):
                    best_distance = len(walk_route)
                    action_paths = walk_route
                    paths = new_path
                    improvement = True
                    break  

            for j in range(i + 1, len(paths)):
                if i < len(paths)-1:
                    new_action, new_path = two_opt_swap(action_paths, paths, i, j, problem.startingGameState)
                    walk_route = walk_path(new_action, problem)
                    if len(walk_route) < best_distance:
                        best_distance = len(walk_route)
                        action_paths = walk_route
                        paths = new_path
                        improvement = True
                        break  # Exit the inner loop

                    if time.time() - start > 9.88:
                        return action_paths
                if improvement:
                    break  # Exit the outer loop

    return action_paths

def remove_a_dot(action_path, path, index, gs):

    new_actions = action_path[:]
    new_path: list = path[:]

    start_action = 0
    for i in range(index-1):
        start_action += path[i][1]
    end_action = start_action + path[index-1][1] + path[index][1]

    if index < len(path)-1:
        distance_prob = q1a_problem(gs)
        distance_prob.start_pos = path[index-1][0]
        distance_prob.goalPoint = path[index+1][0]
        new_interval_path = q1a_solver(distance_prob)
    else: # if the point to remove is the at the end of path
        new_interval_path = []

    new_path[index-1] = (new_path[index-1][0], len(new_interval_path))
    new_path.pop(index)
    new_actions[start_action:end_action] = new_interval_path
    print(new_actions)
    return new_actions, new_path


    

