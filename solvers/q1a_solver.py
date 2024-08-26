#---------------------#
# DO NOT MODIFY BEGIN #
#---------------------#

import logging

import util
from problems.q1a_problem import q1a_problem

def q1a_solver(problem: q1a_problem):
    astarData = astar_initialise(problem)
    num_expansions = 0
    terminate = False
    while not terminate:
        num_expansions += 1
        terminate, result = astar_loop_body(problem, astarData)
    print(f'Number of node expansions: {num_expansions}')
    print(result)
    return result

#-------------------#
# DO NOT MODIFY END #
#-------------------#

from game import Directions
import heapq as hq

class Node:

    def __init__(self, state) -> None:
        self.state = state
        self.g: int = float('inf')
        self.f: int = float('inf')
        self.parent: Node = None
        self.actionTaken: Directions = None
        self.visited: bool = False

    def __eq__(self, value: 'Node') -> bool:
        return self.f == value.f

    def __ne__(self, value: 'Node') -> bool:
        return self.f != value.f

    def __lt__(self, value: 'Node'): 
        return self.f < value.f
    
    def __gt__(self, value: 'Node'): 
        return self.f > value.f
    
    def __ge__(self, value: 'Node'): 
        return self.f >= value.f
    
    def __le__(self, value: 'Node'): 
        return self.f <= value.f

class AStarData:
    # YOUR CODE HERE
    def __init__(self):
        self.open_list: list[Node] = []
        self.nodes: dict[tuple, Node] = {}
        self.terminate = False

def astar_initialise(problem: q1a_problem):

    # YOUR CODE HERE
    astarData = AStarData()

    # get the starting position of the pacman, and initialise the starting node for the position
    start_pos = problem.getStartState()
    start_node = Node(state=start_pos)
    start_node.g = 0
    start_node.f = 0
    start_node.parent = start_node
    astarData.nodes[start_pos] = start_node

    # Check if pacman already at destination, if yes terminate the program, otherwise push the node into queue
    if problem.isGoalState(start_pos):
        astarData.terminate = True
    else:
        # push it into queue
        hq.heappush(astarData.open_list, start_node)

    return astarData

def astar_loop_body(problem: q1a_problem, astarData: AStarData):
    # YOUR CODE HERE
    count = 0
    while len(astarData.open_list) > 0:
        print(count)
        # get the node with the lower f_value
        current_node = astarData.open_list.pop()

        if current_node.visited:
            continue

        # check if the current position is the goal state
        if problem.isGoalState(current_node.state):
            actions = action_reconstruct(astarData, current_node)
            astarData.terminate = True
            return astarData.terminate, actions

        for next_state, action, cost in problem.getSuccessors(current_node.state):

            # computer the new g_value, h_value and f_value
            new_g = current_node.g + cost
            new_h = astar_heuristic(next_state, problem.goalPoint)
            new_f = new_g + new_h

            # First check: is the successor already crated a node
            next_state_node = astarData.nodes.get(next_state)

            if next_state_node is None:  # if there are no node associate with this state, created one 
                next_state_node = Node(state=next_state)
                astarData.nodes[next_state] = next_state_node

            # update the h_value, it parent node and the action from parent node to this node (also known as noe redirection)
            if astarData.nodes[next_state].f > new_f:
                next_state_node.f = new_f
                next_state_node.g = new_g
                next_state_node.actionTaken = action
                next_state_node.parent = current_node

                if next_state_node.visited == True:
                    next_state_node.visited == False

                hq.heappush(astarData.open_list, next_state_node)
        count += 1
        print(astarData.nodes)

    return astarData.terminate, []

def astar_heuristic(current, goal):
    # YOUR CODE HERE

    # current is the position of pacman in (x,y), where goal is the position of goal state
    # in this heuristic, the h_value is the manhattan distance between this two point 
    return util.manhattanDistance(current, goal)

def action_reconstruct(astarData: AStarData, destination_node: Node):
    action = []

    state_node: Node = destination_node
    while state_node.parent.state != state_node.state:
        action.append(state_node.actionTaken)
        state_node = state_node.parent

    action.reverse()
    return action

