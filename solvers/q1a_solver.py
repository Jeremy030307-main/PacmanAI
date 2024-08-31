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
        return (self.f,self.g) == (value.f,value.g)

    def __ne__(self, value: 'Node') -> bool:
        return (self.f,self.g) != (value.f,value.g)

    def __lt__(self, value: 'Node'): 
        return (self.f,self.g) < (value.f,value.g)
    
    def __gt__(self, value: 'Node'): 
        return (self.f,self.g) > (value.f,value.g)
    
    def __ge__(self, value: 'Node'): 
        return (self.f,self.g) >= (value.f,value.g)
    
    def __le__(self, value: 'Node'): 
        return (self.f,self.g) <=(value.f,value.g)

class AStarData:
    # YOUR CODE HERE
    def __init__(self):
        self.open_list: list[Node] = []
        self.nodes: dict[tuple, Node] = {}
        self.closed_set: set[tuple] = set()
        self.terminate = False

def astar_initialise(problem: q1a_problem):

    # YOUR CODE HERE
    astarData = AStarData()

    # get the starting position of the pacman, and initialise the starting node for the position
    problem.getStartState()
    start_pos = problem.start_pos
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
    while len(astarData.open_list) > 0:
        
        # get the node with the lower f_value                
        hq.heapify(astarData.open_list)
        current_node = astarData.open_list.pop(0)

        if current_node.state in astarData.closed_set:
            continue
        astarData.closed_set.add(current_node.state)

        # check if the current position is the goal state
        if problem.isGoalState(current_node.state):
            actions = action_reconstruct(astarData, current_node)
            astarData.terminate = True
            return astarData.terminate, actions

        for next_state, action, cost in problem.getSuccessors(current_node.state):
            if next_state in astarData.closed_set:
                continue
            
            new_g = current_node.g + cost
            
            if next_state not in astarData.nodes or new_g < astarData.nodes[next_state].g:
                next_state_node = astarData.nodes.get(next_state, Node(next_state))
                next_state_node.g = new_g
                next_state_node.f = new_g + astar_heuristic(next_state, problem.goalPoint)
                next_state_node.parent = current_node
                next_state_node.actionTaken = action
                
                astarData.nodes[next_state] = next_state_node
                hq.heappush(astarData.open_list, next_state_node)

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

