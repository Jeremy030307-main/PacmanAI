import logging
import random

import util 
from game import Actions, Agent, Directions, Grid, AgentState
from logs.search_logger import log_function
from pacman import GameState
from util import manhattanDistance
	 
def scoreEvaluationFunction(currentGameState: GameState, searchData: 'SearchData', in_dead_end, in_tunnel):

    print(in_dead_end, in_tunnel)
    # initial score 
    score = currentGameState.getScore()

    # Get Pacman position and relevant game state information
    pacman_pos = currentGameState.getPacmanPosition()

    # ----------------------------------- Reward Section (Food) -----------------------------------
    remaining_food = currentGameState.getFood().asList()

    # Manhattan distance from pacman to each food
    food_dist = [manhattanDistance(pacman_pos, foodPos) for foodPos in remaining_food]

    # evaluation of score for the curren state
    number_of_non_food_pos = len(currentGameState.getFood().asList(False))  

    reciprocalfoodDistance = 0
    if sum(food_dist) > 0:
        reciprocalfoodDistance = 1.0 / sum(food_dist)
        
    score += reciprocalfoodDistance + number_of_non_food_pos

    # ----------------------------------- Reward and Penalty Section (Ghost) -----------------------------------
    ghost_state: list[AgentState] = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    # Manhattan distance from pacman to each ghost, and also keep track of the closest ghost
    nearest_ghost_dist = float('inf')
    ghost_dist = [manhattanDistance(pacman_pos, ghost.getPosition()) for ghost in ghost_state]

    for ghost in ghost_state:
        curr_dist = manhattanDistance(pacman_pos, ghost.getPosition())
        ghost_dist.append(curr_dist)
        if curr_dist < nearest_ghost_dist:
            nearest_ghost_dist = curr_dist

    # Distance to nearest capsule (power pellet)
    capsuleDistances = [manhattanDistance(pacman_pos, capsule) for capsule in capsules]
    nearestCapsuleDist = min(capsuleDistances) if capsuleDistances else float('inf')
    
    # check is pacman already in a dead end path
    if in_dead_end:  
        print("Enter")
        dead_end_start_pos, dist_walk = in_dead_end
        move_needed = len(searchData.dead_end[dead_end_start_pos])*2 - dist_walk + 2  # this is the move needed by pacman to reach the dead end exit
        priority = 8

    # check is pacman already in a tunnel
    elif in_tunnel:  
        tunnel_start, tunnel_end, dist_walk = in_tunnel
        move_needed = max(len(searchData.tunnel[tunnel_end]) - dist_walk+1, dist_walk+1) # the maximmum distance to exit the tunnel between both opening
        priority = 6

    # check is pacman on the start of the dead end path
    elif searchData.dead_end.get(pacman_pos):  
        dead_end_path = searchData.dead_end.get(pacman_pos)
        move_needed = 2 * (len(dead_end_path)-1) + 1
        priority = 5
    
    # check is pacman on the start of the tunnel
    elif searchData.tunnel.get(pacman_pos):
        tunnel_path = searchData.tunnel.get(pacman_pos)
        move_needed = len(tunnel_path)
        priority = 3

    # check is pacman on a corner
    elif pacman_pos in searchData.corner:
        move_needed = 1
        priority = 2
   
    else:
        move_needed = 0
        priority = 1
    
    if nearest_ghost_dist > move_needed:
        score += priority * (nearest_ghost_dist-move_needed)
    else:
        score += (-priority) * move_needed * (move_needed-nearest_ghost_dist)

    newScaredTimes = [ghostState.scaredTimer for ghostState in ghost_state]
    sumScaredTimes = sum(newScaredTimes)
    sumGhostDistance = sum (ghost_dist)

    if sumScaredTimes > 0:    
        score +=   sumScaredTimes + (-1 * len(capsules)) + (-1 * sumGhostDistance)
    else :
        score +=  sumGhostDistance + len(capsules)

    return score

class SearchData:

    def __init__(self) -> None:
        self.first_time = False  # indicate that this is the first time the game call for agent action
        self.dead_end: dict = None
        self.tunnel: dict = None
        self.corner: set = None

        # this is the state on weather the pacman is now in a tunnel or in a path that lead to death_end
        self.in_dead_end = False
        self.in_tunnel = False

class Q2_Agent(Agent):

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '3'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.data = SearchData()

    @log_function
    def getAction(self, gameState: GameState):
        """
            Returns the minimax action from the current gameState using self.depth
            and self.evaluationFunction.

            Here are some method calls that might be useful when implementing minimax.
            gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

            gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

            gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        logger = logging.getLogger('root')
        logger.info('MinimaxAgent')

        if self.data.dead_end == None:
            self.data.dead_end = {} ;self.data.tunnel = {} ; self.data.corner = set()
            self.check_maze_info(gameState)
            self.data.first_time = True
        
        actions = gameState.getLegalActions(0)
        currentScore = float('-inf')
        returnAction = None
        alpha = float('-inf')
        beta = float('inf')

        for action in actions:
            nextState = gameState.generateSuccessor(0,action)

            new_in_dead_end = self.data.in_dead_end
            if new_in_dead_end:
                start, walk_dist = new_in_dead_end
                path = self.data.dead_end[start]
                if len(path) > 0:
                    if walk_dist + 1 < len(path) and path[walk_dist+1] == nextState.getPacmanPosition():
                        new_in_dead_end = (start, walk_dist+1)
                    elif walk_dist - 1 >= 0 and path[walk_dist-1] == nextState.getPacmanPosition():
                        new_in_dead_end = (start, walk_dist-1)
                    else:
                        new_in_dead_end = False
            else:
                path = self.data.dead_end.get(nextState.getPacmanPosition())

                if path is not None:
                    new_in_dead_end = (nextState.getPacmanPosition(), 0)
            
            new_in_tunnel = self.data.in_tunnel
            if new_in_tunnel:
                start,end, walk_dist = new_in_tunnel
                path = self.data.tunnel[start]
                if len(path) > 0:
                    if walk_dist + 1 < len(path) and path[walk_dist+1] == nextState.getPacmanPosition():
                        new_in_tunnel = (start,end, walk_dist+1)
                    elif walk_dist - 1 >= 0 and path[walk_dist-1] == nextState.getPacmanPosition():
                        new_in_tunnel = (start,end, walk_dist-1)
                    else:
                        new_in_tunnel = False
            else:
                path = self.data.tunnel.get(nextState.getPacmanPosition())

                if path is not None:
                    new_in_tunnel = (nextState.getPacmanPosition(), path[-1], 0)
                    
            # Next level is a min level. Hence calling min for successors of the root.
            score, updated_in_dead_end, updated_in_tunnel = \
                self.min_value(nextState,0,alpha,beta, new_in_dead_end, new_in_tunnel)

            # Choosing the action which is Maximum of the successors.
            if score > currentScore:
                returnAction = action
                currentScore = score
                self.data.in_dead_end = updated_in_dead_end
                self.data.in_tunnel = updated_in_tunnel

            # Updating alpha value at root.    
            if score > beta:
                return returnAction
            
            alpha = max(alpha,score)

        return returnAction

    def max_value(self, game_state: GameState, depth: int, alpha: int, beta:int, in_dead_end, in_tunnel):

        currDepth = depth + 1

        # Check for termina state
        if game_state.isWin() or game_state.isLose() or currDepth >= self.depth:  
            return self.evaluationFunction(game_state, self.data, in_dead_end, in_tunnel), in_dead_end, in_tunnel
    
        max_value = float('-inf')
        actions = game_state.getLegalActions(0)
        for action in actions:

            # generate the successor game state after taking this action
            successor= game_state.generateSuccessor(0,action)

            new_in_dead_end = in_dead_end
            if in_dead_end:
                start, walk_dist = in_dead_end
                path = self.data.dead_end[start]
                if len(path) > 0:
                    if walk_dist + 1 < len(path) and path[walk_dist+1] == successor.getPacmanPosition():
                        new_in_dead_end = (start, walk_dist+1)
                    elif walk_dist - 1 >= 0 and path[walk_dist-1] == successor.getPacmanPosition():
                        new_in_dead_end = (start, walk_dist-1)
                    else:
                        new_in_dead_end = False
            else:
                path = self.data.dead_end.get(successor.getPacmanPosition())

                if path is not None:
                    new_in_dead_end = (successor.getPacmanPosition(), 0)
            
            new_in_tunnel = in_tunnel
            if in_tunnel:
                start,end, walk_dist = in_tunnel
                path = self.data.tunnel[start]
                if len(path) > 0:
                    if walk_dist + 1 < len(path) and path[walk_dist+1] == successor.getPacmanPosition():
                        new_in_tunnel = (start,end, walk_dist+1)
                    elif walk_dist - 1 >= 0 and path[walk_dist-1] == successor.getPacmanPosition():
                        new_in_tunnel = (start,end, walk_dist-1)
                    else:
                        new_in_tunnel = False
            else:
                path = self.data.tunnel.get(successor.getPacmanPosition())

                if path is not None:
                    new_in_tunnel = (successor.getPacmanPosition(), path[-1], 0)


            updated_max, updated_in_dead_end, updated_in_tunnel = self.min_value(successor, currDepth, alpha, beta, new_in_dead_end, new_in_tunnel)

            if updated_max > max_value:
                max_value = updated_max
                new_in_dead_end = updated_in_dead_end
                new_in_tunnel = updated_in_tunnel

            if max_value > beta:
                return max_value, new_in_dead_end, new_in_tunnel
            
            alpha = max(alpha,max_value)

        return max_value, new_in_dead_end, new_in_tunnel

    def min_value(self, game_state: GameState, depth: int, alpha: int, beta:int, in_dead_end, in_tunnel, agent_index = 1):

        # Check for terminal state
        if game_state.isWin() or game_state.isLose():
            return self.evaluationFunction(game_state, self.data, in_dead_end, in_tunnel), in_dead_end, in_tunnel
        
        minvalue = float('inf')
        actions = game_state.getLegalActions(agent_index)
        for action in actions:
            successor= game_state.generateSuccessor(agent_index,action)

            new_in_dead_end = in_dead_end
            if in_dead_end:
                start, walk_dist = in_dead_end
                path = self.data.dead_end[start]
                if len(path) > 0:
                    if walk_dist + 1 < len(path) and path[walk_dist+1] == successor.getPacmanPosition():
                        new_in_dead_end = (start, walk_dist+1)
                    elif walk_dist - 1 >= 0 and path[walk_dist-1] == successor.getPacmanPosition():
                        new_in_dead_end = (start, walk_dist-1)
                    else:
                        new_in_dead_end = False
            else:
                path = self.data.dead_end.get(successor.getPacmanPosition())

                if path is not None:
                    new_in_dead_end = (successor.getPacmanPosition(), 0)
            
            new_in_tunnel = in_tunnel
            if in_tunnel:
                start,end, walk_dist = in_tunnel
                path = self.data.tunnel[start]
                if len(path) > 0:
                    if walk_dist + 1 < len(path) and path[walk_dist+1] == successor.getPacmanPosition():
                        new_in_tunnel = (start,end, walk_dist+1)
                    elif walk_dist - 1 >= 0 and path[walk_dist-1] == successor.getPacmanPosition():
                        new_in_tunnel = (start,end, walk_dist-1)
                    else:
                        new_in_tunnel = False
            else:
                path = self.data.tunnel.get(successor.getPacmanPosition())

                if path is not None:
                    new_in_tunnel = (successor.getPacmanPosition(), path[-1], 0)

            if agent_index == (game_state.getNumAgents()-1):
                updated_minvalue, updated_in_dead_end, updated_in_tunnel = self.max_value(successor,depth,alpha,beta, new_in_dead_end, new_in_tunnel)

                if updated_minvalue < minvalue:
                    minvalue = updated_minvalue
                    new_in_dead_end = updated_in_dead_end
                    new_in_tunnel = updated_in_tunnel
            else:
                updated_minvalue, updated_in_dead_end, updated_in_tunnel = \
                    self.min_value(successor,depth,alpha,beta,new_in_dead_end, new_in_tunnel, agent_index+1)
                
                if updated_minvalue < minvalue:
                    minvalue = updated_minvalue
                    new_in_dead_end = updated_in_dead_end
                    new_in_tunnel = updated_in_tunnel
            
            if minvalue <= alpha:
                return minvalue, new_in_dead_end, new_in_tunnel
            
            beta = min(beta,minvalue)

        return minvalue, new_in_dead_end, new_in_tunnel
    
    def check_maze_info(self, game_state: GameState):
        
        walls: Grid = game_state.getWalls()
        height = walls.height
        width = walls.width

        all_directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        visited = [[False for _ in range(height)] for _ in range(width)]
        stack = [game_state.getPacmanPosition()]

        while stack:

            x, y = stack.pop()

            if visited[x][y]:
                continue

            # Mark the current node as visited
            visited[x][y] = True

            is_corner, is_dead_end, is_corridor = self.position_check((x,y), game_state)

            # Check weather the current position is a dead end or corner
            if is_dead_end:
                self.data.dead_end[(x,y)] = []
            elif is_corner:
                self.data.corner.add((x,y))
            elif is_corridor:
                self.follow_corridor((x,y), visited, game_state, stack)

            for direction in all_directions:
                next_x = x + direction[0]
                next_y = y + direction[1]

                if 0 <= next_x < width and 0 <= next_y < height and not visited[next_x][next_y] and not game_state.hasWall(next_x, next_y):
                    stack.append((next_x, next_y))

    def position_check(self, position, game_state: GameState):
        """ 
        Check weather the given position is a dead end. A dead end is a 
        position in the maze where there are only one valid action.
        """
        x, y = position
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        open_paths = []

        is_corner = False
        is_dead_end = False
        is_corridor = False
        
        for dx, dy in directions:
            if not game_state.hasWall(x+dx, y+dy):
                open_paths.append((dx, dy))

        if len(open_paths) == 1:
            is_dead_end = True
        elif len(open_paths) == 2:
            (dx1, dy1), (dx2, dy2) = open_paths

            # Check if they are perpendicular ( if it's a corner )
            is_corner =  abs(dx1) != abs(dx2) and abs(dy1) != abs(dy2)

            # Horizontal path: both moves are left-right
            if abs(dx1) == 1 and dy1 == 0 and abs(dx2) == 1 and dy2 == 0:
                is_corridor = True
            
            # Vertical path: both moves are up-down
            if abs(dy1) == 1 and dx1 == 0 and abs(dy2) == 1 and dx2 == 0:
                is_corridor = True

        return (is_corner, is_dead_end, is_corridor)
    
    def follow_corridor(self, start, visited, gs: GameState, ori_stack: list):

        height = gs.getWalls().height
        width = gs.getWalls().width     
        all_directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]

        # if met a corridor, then it has potential to lead to a dead end, or a long corridor with several corner
        stack = [(start, [])]

        while stack:

            (x,y), path = stack.pop()
            path.append((x,y))

            visited[x][y] = True

            for direction in all_directions:
                next_x = x + direction[0]
                next_y = y + direction[1]

                if (0 <= next_x < width) and (0 <= next_y < height) and (not gs.hasWall(next_x, next_y)) and ((next_x, next_y) not in path):
                    is_corner, is_dead_end, is_corridor = self.position_check((next_x,next_y), gs)
                    if is_corner or is_corridor:
                        stack.append(((next_x, next_y), path[:]))
                    elif is_dead_end:
                        new_path = path[:]
                        new_path.append((next_x, next_y))
                        self.data.dead_end[new_path[0]] = new_path
                        if gs.getPacmanPosition() in path:
                            self.data.in_dead_end = (new_path[0], new_path.index(gs.getPacmanPosition()))
                        
                    else:
                        ori_stack.append((next_x, next_y))
                        if len(path) > 1:
                            new_path = path[:]

                            if self.data.tunnel.get(path[0]):
                                new_path = path[::-1] + self.data.tunnel[path[0]][1:]
                                self.data.tunnel.pop(path[0])
                                self.data.tunnel[new_path[0]] = new_path[:]
                            else:
                                self.data.tunnel[new_path[0]] = new_path[:]

                            self.data.tunnel[new_path[-1]] = new_path[:]

                            if gs.getPacmanPosition() in new_path:
                                self.data.in_tunnel = (new_path[0],new_path[-1], new_path.index(gs.getPacmanPosition()))
