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
from game import Directions
import heapq as hq

def q1c_solver(problem: q1c_problem):
    
    start_pos = problem.getStartState()

    keyPoint = [start_pos] + problem.food 
    dist = [[0 for _ in range(len(keyPoint)+1)] for _ in range(len(keyPoint)+1)]
    actions = [[0 for _ in range(len(keyPoint)+1)] for _ in range(len(keyPoint)+1)]

    for i in range(len(keyPoint)):
        for j in range(len(keyPoint)):
            position_problem = q1a_problem(problem.startingGameState)
            actions_taken = q1a_solver(position_problem, keyPoint[i], keyPoint[j])
            dist[i+1][j+1] = len(actions_taken)
            actions[i+1][j+1] = actions_taken

    n = len(keyPoint)  # there are four nodes in example graph (graph is 1-based)
    
    # Memoization for top-down recursion
    memo = [[-1] * (1 << (n+1)) for _ in range(n+1)]
    parent = [[-1] * (1 << (n+1)) for _ in range(n+1)]  # To store the backtracking info

    def fun(i, mask):
        # Base case
        if mask == ((1 << i) | 3):
            return dist[1][i]

        # Memoization check
        if memo[i][mask] != -1:
            return memo[i][mask]

        res = 10**9  # Initialize result to a large value

        # Try every possible previous city j
        for j in range(1, n+1):
            if (mask & (1 << j)) != 0 and j != i and j != 1:
                cost = fun(j, mask & (~(1 << i))) + dist[j][i]
                if cost < res:
                    res = cost
                    parent[i][mask] = j  # Store the city j that leads to the minimum cost

        memo[i][mask] = res  # Store the minimum value
        return res

    # Driver code to compute the minimum cost
    ans = 10**9
    last_city = -1
    final_mask = (1 << (n+1)) - 1  # All cities visited

    for i in range(1, n+1):
        cost = fun(i, final_mask) + dist[i][1]
        if cost < ans:
            ans = cost
            last_city = i  # Store the last city in the optimal path

    print("The cost of the most efficient tour = " + str(ans))

    # Step 2: Backtrack to find the path
    path = []
    mask = final_mask

    while last_city != -1:
        path.append(last_city)
        next_city = parent[last_city][mask]
        mask = mask & (~(1 << last_city))
        last_city = next_city

    path.append(1)  # Add the starting city at the end of the path

    # Since we backtracked, the path is in reverse order, so we need to reverse it
    path.reverse()

    # get the actions
    whole_actions = []
    for i in range(len(path[1:])):
        print(path[i],path[i+1])
        print(actions[path[i]][path[i+1]])
        whole_actions += actions[path[i]][path[i+1]]
    
    print(whole_actions)

    print("The most efficient path is:", path)

    return whole_actions

def q1a_solver(problem: q1a_problem, start, end):
    astarData = astar_initialise(problem, start, end)
    num_expansions = 0
    terminate = False
    while not terminate:
        num_expansions += 1
        terminate, result = astar_loop_body(problem, astarData)
    # print(f'Number of node expansions: {num_expansions}')
    return result

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

def astar_initialise(problem: q1a_problem, start, end):

    # YOUR CODE HERE
    astarData = AStarData()
    problem.getStartState()
    problem.start_pos = start
    problem.goalPoint = end

    # get the starting position of the pacman, and initialise the starting node for the position
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
        if current_node.visited:
            continue

        current_node.visited = True

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

            # update the h_value, it parent node and the action from parent node to this node (also known as node redirection)
            if astarData.nodes[next_state].f > new_f:
                next_state_node.f = new_f
                next_state_node.g = new_g
                next_state_node.actionTaken = action
                next_state_node.parent = current_node

                if next_state_node.visited == True:
                    next_state_node.visited == False

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

