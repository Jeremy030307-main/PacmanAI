#---------------------#
# DO NOT MODIFY BEGIN #
#---------------------#

import logging

import util
from problems.q1a_problem import q1a_problem
from game import Directions

def q1a_solver(problem: q1a_problem):
    astarData = astar_initialise(problem)
    num_expansions = 0
    terminate = False
    while not terminate:
        num_expansions += 1
        terminate, result = astar_loop_body(problem, astarData)
    print(f'Number of node expansions: {num_expansions}')
    return result

#-------------------#
# DO NOT MODIFY END #
#-------------------#

class AStarData:
    # YOUR CODE HERE
    def __init__(self):
        self.open_list = util.PriorityQueue()
        self.close_list = util.Counter()
        self.g_value = util.Counter()
        self.parents = {}
        self.terminate = False

def astar_initialise(problem: q1a_problem):

    # YOUR CODE HERE
    astarData = AStarData()
    start_pos = problem.getStartState()

    # initiliase the starting position into the astarData
    astarData.open_list.push(start_pos, 0)
    astarData.g_value[start_pos] = 0
    astarData.close_list[start_pos] = 0
    astarData.parents[start_pos] = Directions.STOP

    # Check if pacman already at destination
    if problem.isGoalState(start_pos):
        astarData.terminate = True

    return astarData

def astar_loop_body(problem: q1a_problem, astarData: AStarData):
    # YOUR CODE HERE
    
    while (not astarData.open_list.isEmpty()) or (not astarData.terminate):

        # get the node with the lower f_value
        current_state = astarData.open_list.pop()

        # check if the current position is the goal state
        if problem.isGoalState(current_state):
            actions = action_reconstruct(astarData, problem.goalPoint)
            astarData.terminate = True
            return actions

        for next_state, action, cost in problem.getSuccessors(current_state):

            # computer the new g_value, h_value and f_value
            new_g = astarData.g_value[current_state] + cost
            new_h = astar_heuristic(next_state, problem.goalPoint)
            new_f = new_g + new_h

            # First Check: if next state does not present in both open list and close list, means it is newly visited node
            # Action: Add next state to open list
            if next_state not in astarData.open_list and next_state not in astarData.close_list:
                astarData.open_list.push(next_state, new_f)
                astarData.parents[next_state] = (action, current_state)
                astarData.g_value[next_state] = new_g 

            elif astarData.g_value[next_state] > new_g:

                if astarData.close_list[next_state] > 0:
                    # Second Check: if next state has a larger g_value compared to current state, means that there is a shorter path from start to this node
                    # Action: Set the current state as the parent of the next state
                    # Third check: if next state already visited (if next state in closed list)
                    # Action: Removed it from closed list and add in to open list
                    astarData.parents[next_state] = (action, current_state)
                    astarData.g_value[next_state] = new_g 
                    astarData.close_list[next_state] = 0
                    astarData.open_list.push(next_state, new_f)
            
                elif next_state in astarData.open_list:
                    astarData.open_list.update(next_state, new_f)
                    astarData.parents[next_state] = (action, current_state)

def astar_heuristic(current, goal):
    # YOUR CODE HERE

    # current is the position of pacman in (x,y), where goal is the position of goal state
    # in this heuristic, the h_value is the manhattan distance between this two point 
    return util.manhattanDistance(current, goal)

def action_reconstruct(astarData: AStarData, destination):
    action = []

    state = destination
    while astarData.parents[state][0] == Directions.STOP:
        action.append[astarData.parents[state][0]]
        state = astarData.parents[state][1]

    action.append(astarData.parents[state][0])
    action.reverse()

    return action

