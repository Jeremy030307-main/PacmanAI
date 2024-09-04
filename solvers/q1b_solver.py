#---------------------#
# DO NOT MODIFY BEGIN #
#---------------------#

import logging

import util
from problems.q1b_problem import q1b_problem

def q1b_solver(problem: q1b_problem):
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
        self.treshold = None
        self.terminate = False
        self.visited = []

def astar_initialise(problem: q1b_problem):

    # YOUR CODE HERE
    astarData = AStarData()

    # get the starting position of the pacman, and initialise the starting node for the position
    start_pos = problem.getStartState()
    start_node = Node(state=start_pos)
    start_node.g = 0
    start_node.f = astar_heuristic(start_pos, problem.goalPoints)
    start_node.parent = start_node
    astarData.nodes[start_pos] = start_node

    # set the initial threshold value as the f_value of start node
    astarData.treshold = start_node.f

    # Check if pacman already at destination, if yes terminate the program, otherwise push the node into queue
    if problem.isGoalState(start_pos):
        astarData.terminate = True
    else:
        # push it into queue
        hq.heappush(astarData.open_list, start_node)
        hq.heappush(astarData.visited, start_node)

    return astarData    

def astar_loop_body(problem: q1b_problem, astarData: AStarData):
    # YOUR CODE HERE
    
    # get the node with the lower f_value, and set it as threshold
    if len(astarData.open_list) <= 0:
        lower_visited_node = astarData.visited.pop(0)
        astarData.treshold = lower_visited_node.f
        hq.heappush(astarData.open_list, lower_visited_node)
        
    # get the node with the lower f_value                
    hq.heapify(astarData.open_list)
    current_node = astarData.open_list.pop(0)

    while current_node.visited and len(astarData.open_list) > 0:
        current_node = astarData.open_list.pop(0)
        if len(astarData.open_list) <= 0:
            lower_visited_node = astarData.visited.pop(0)
            astarData.treshold = lower_visited_node.f
            hq.heappush(astarData.open_list, lower_visited_node)

    current_node.visited = True

    # check if the current position is the goal state
    if problem.isGoalState(current_node.state):
        actions = action_reconstruct(astarData, current_node)
        astarData.terminate = True
        return astarData.terminate, actions

    for next_state, action, cost in problem.getSuccessors(current_node.state):

        # computer the new g_value, h_value and f_value
        new_g = current_node.g + cost
        new_h = astar_heuristic(next_state, problem.goalPoints)
        new_f = new_g + new_h

        # First check: is the successor already have a node
        next_state_node = astarData.nodes.get(next_state)

        if next_state_node is None:  # if there are no node associate with this state, created one 
            next_state_node = Node(state=next_state)
            astarData.nodes[next_state] = next_state_node

        # update the h_value, it parent node and the action from parent node to this node (also known as node redirection)
        if astarData.nodes[next_state].f > new_f:
            next_state_node.f = new_f
            next_state_node.g = new_g
            next_state_node.actionTaken = action
            next_state_node.parent = current_node

            if next_state_node.f <= astarData.treshold:
                hq.heappush(astarData.open_list, next_state_node)
            else:
                hq.heappush(astarData.visited, next_state_node)

    return astarData.terminate, []

def astar_heuristic(current, goal):
    # YOUR CODE HERE

    # current is the position of pacman in (x,y), where goal is the position of goal state
    # in this heuristic, the h_value is the manhattan distance between this two point 
    
    return min([util.manhattanDistance(current, goal_point) for goal_point in goal])

def action_reconstruct(astarData: AStarData, destination_node: Node):
    action = []

    state_node: Node = destination_node
    while state_node.parent.state != state_node.state:
        action.append(state_node.actionTaken)
        state_node = state_node.parent

    action.reverse()
    return action
