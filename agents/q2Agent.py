import logging
import random

import util 
from game import Actions, Agent, Directions, Grid, AgentState
from logs.search_logger import log_function
from pacman import GameState
from util import manhattanDistance
from enum import Enum
import random

class MazeState(Enum):
    CORNER = 0.2
    TUNNEL = 0.5
    DEAD_END = 0.9

    def __call__(self, index=None, path: 'PathInfo' = None):
        return MazeStateInstance(self, index, path)

def scoreEvaluationFunction(currentGameState: GameState, maze_info: list[list['MazeStateInstance']]):

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

    farthest_food = max(food_dist) if len(food_dist) > 0 else 0
    
    reciprocalfoodDistance = 0
    if sum(food_dist) > 0:
        reciprocalfoodDistance = 1.0 / sum(food_dist)
    
    reciprocalFathestFood = 0
    if farthest_food > 0:
        reciprocalFathestFood = 1/farthest_food
    
    if len(remaining_food) < 0.2 * len(remaining_food):
        food_reward = reciprocalFathestFood
    else:
        food_reward = reciprocalfoodDistance
        
    score += currentGameState.getScore() + number_of_non_food_pos + food_reward

    # ----------------------------------- Reward and Penalty Section (Ghost) -----------------------------------
    ghost_state: list[AgentState] = currentGameState.getGhostStates()
    ghost_dist = [manhattanDistance(pacman_pos, ghost.getPosition()) for ghost in ghost_state]
    nearest_ghost_dist =  min(ghost_dist) if ghost_dist else float('inf')

    capsules = currentGameState.getCapsules()
    # Distance to nearest capsule (power pellet)
    capsuleDistances = [manhattanDistance(pacman_pos, capsule) for capsule in capsules]
    nearestCapsuleDist = min(capsuleDistances) if capsuleDistances else float('inf')
    
    # Ghost handling based on scared state and proximity
    scared_times = [ghost.scaredTimer for ghost in ghost_state]
    total_scared_time = sum(scared_times)

    if total_scared_time > 0:
        # If ghosts are scared, focus on chasing them
        score += total_scared_time - sum(ghost_dist)
    else:
        # Otherwise, focus on avoiding them
        score += sum(ghost_dist)
    
    if sum(scared_times) > 0:    
        score +=   sum(scared_times) + (-1 * len(capsules)) + (-1 * sum(ghost_dist))
    else :
        score +=  sum(ghost_dist) + len(capsules)
    return score

    # # ----------------------------------- Reward and Penalty Section (Maze State) -----------------------------------

    # # get the state of the pacman position on the maze
    # current_pos_state: MazeStateInstance = maze_info[pacman_pos[0]][pacman_pos[1]]
    # escape_move = 0
    # eat_move = 0
    # priority = current_pos_state.status.value if current_pos_state is not None else 0

    # # check is pacman in a dead end path
    # if current_pos_state == MazeState.DEAD_END:
    #     print("Dead End")
    #     escape_move = current_pos_state.index + 2

    #     dead_end_pos = current_pos_state.path_info.end
    #     middle_pos = current_pos_state.path_info.path[len(current_pos_state.path_info.path)//2]
    #     if currentGameState.hasFood(dead_end_pos[0], dead_end_pos[1]) == True:
    #         eat_move = (current_pos_state.index + 1) + (current_pos_state.path_info.length - (current_pos_state.index + 1)) * 2
    #     elif currentGameState.hasFood(middle_pos[0], middle_pos[1]) == True:
    #         eat_move = ((current_pos_state.index + 1) + (current_pos_state.path_info.length - (current_pos_state.index + 1)) * 2)/2
    #     else: 
    #         eat_move = 0

    # # check is pacman in a tunnel
    # elif current_pos_state == MazeState.TUNNEL:
    #     print("Tunnel")
    #     # check if the tunnel start and end still have food
    #     tunnel_start = current_pos_state.path_info.start
    #     tunnel_end = current_pos_state.path_info.end

    #     has_start_food = currentGameState.hasFood(tunnel_start[0], tunnel_start[1])
    #     has_end_food = currentGameState.hasFood(tunnel_end[0], tunnel_end[1])

    #     if has_start_food and has_end_food:
    #         # Incentivize clearing the tunnel by prioritizing the closer food
    #         if current_pos_state.index <= current_pos_state.path_info.length / 2:
    #             escape_move = current_pos_state.index + 1  # Move towards the tunnel end
    #         else:
    #             escape_move = current_pos_state.path_info.length - current_pos_state.index  # Move towards the tunnel start
        
    #     elif has_start_food:
    #         escape_move = current_pos_state.path_info.length - current_pos_state.index  # Move towards the tunnel start
        
    #     elif has_end_food:
    #         escape_move = current_pos_state.index + 1  # Move towards the tunnel end
        
    #     else:
    #         # If no food is left in the tunnel, focus on ghost avoidance
    #         first_opening_nearby_ghost = min([manhattanDistance(tunnel_start, ghost.getPosition()) for ghost in ghost_state])
    #         second_opening_nearby_ghost = min([manhattanDistance(tunnel_end, ghost.getPosition()) for ghost in ghost_state])
    #         nearest_ghost_dist = min(first_opening_nearby_ghost, second_opening_nearby_ghost)

    #     # check which opening has a ghost nearby 
    #     first_opening_nearby_ghost = min([manhattanDistance(current_pos_state.path_info.start, ghost.getPosition()) for ghost in ghost_state])
    #     second_opening_nearby_ghost = min([manhattanDistance(current_pos_state.path_info.start, ghost.getPosition()) for ghost in ghost_state])
    #     nearest_ghost_dist = min(first_opening_nearby_ghost, second_opening_nearby_ghost)

    #     if first_opening_nearby_ghost > second_opening_nearby_ghost:
    #         escape_move = current_pos_state.index + 1 
    #         eat_move = current_pos_state.path_info.length - current_pos_state.index 
    #     else:
    #         escape_move = current_pos_state.path_info.length - current_pos_state.index 
    #         eat_move = current_pos_state.index + 1 

    # elif current_pos_state == MazeState.CORNER:
    #     print("Corner")
    #     escape_move = 1
    #     eat_move = 1

    # if nearest_ghost_dist > escape_move and total_scared_time > 0:
    #     print("Ghost Nearby")
    #     score += priority * (eat_move/escape_move) if eat_move != 0 and escape_move != 0 else 0
    # else:
    #     score -= priority * escape_move

    return score

class Q2_Agent(Agent):

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '3'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.maze_info: list[list[PathInfo]] = None

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

        if self.maze_info is None:
            self.check_maze_info(gameState)
        
        actions = gameState.getLegalActions(0)
        currentScore = float('-inf')
        returnAction = None
        alpha = float('-inf')
        beta = float('inf')

        for action in actions:
            nextState = gameState.generateSuccessor(0,action)
                    
            print("start")
            # Next level is a min level. Hence calling min for successors of the root.
            score = self.min_value(nextState,0,alpha,beta)

            # Choosing the action which is Maximum of the successors.
            if score > currentScore:
                returnAction = action
                currentScore = score
            
            alpha = max(alpha,score)

        return returnAction

    def max_value(self, game_state: GameState, depth: int, alpha: int, beta:int):

        currDepth = depth + 1

        # Check for termina state
        if game_state.isWin() or game_state.isLose() or currDepth >= self.depth:  
            return self.evaluationFunction(game_state, self.maze_info)
    
        max_value = float('-inf')
        actions = game_state.getLegalActions(0)
        for action in actions:

            # generate the successor game state after taking this action
            successor= game_state.generateSuccessor(0,action)

            max_value = max(max_value, self.min_value(successor, currDepth, alpha, beta))

            if max_value > beta:
                return max_value
            
            alpha = max(alpha,max_value)

        return max_value

    def min_value(self, game_state: GameState, depth: int, alpha: int, beta:int, agent_index = 1):

        # Check for terminal state
        if game_state.isWin() or game_state.isLose():
            return self.evaluationFunction(game_state, self.maze_info)
        
        minvalue = float('inf')
        actions = game_state.getLegalActions(agent_index)
        for action in actions:
            successor= game_state.generateSuccessor(agent_index,action)

            if agent_index == (game_state.getNumAgents()-1):
                minvalue = min(minvalue, self.max_value(successor,depth,alpha,beta))

            else:
                minvalue = min(minvalue,self.min_value(successor,depth,alpha,beta, agent_index+1))
                
            if minvalue <= alpha:
                return minvalue
            
            beta = min(beta,minvalue)

        return minvalue
    
    def check_maze_info(self, game_state: GameState):
        
        walls: Grid = game_state.getWalls()
        height = walls.height
        width = walls.width

        self.maze_info = [[None for _ in range(height)] for _ in range(width)]

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
                self.maze_info[x][y] =  MazeState.DEAD_END(0, PathInfo([(x,y)]))
            elif is_corner:
                self.maze_info[x][y] = MazeState.CORNER()
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
                        new_path_info = PathInfo(new_path)
                        for index, (x,y) in enumerate(new_path):
                            self.maze_info[x][y] = MazeState.DEAD_END(index, new_path_info)

                    else:
                        ori_stack.append((next_x, next_y))

                        if len(path) > 1:
                            new_path = path[:]
                            new_path_info = PathInfo(new_path)
                            if self.maze_info[new_path[0][0]][new_path[0][1]] is not None:
                                # get the path info of the exiting path 
                                exist_index = self.maze_info[new_path[0][0]][new_path[0][1]].index
                                existing_path_info: PathInfo = self.maze_info[new_path[0][0]][new_path[0][1]].path_info
                                new_path = existing_path_info.path + new_path[1:]
                                existing_path_info.update(new_path)
                                
                                for index, (x,y) in enumerate(new_path):
                                    self.maze_info[x][y] = MazeState.TUNNEL(exist_index+index, existing_path_info)
                            else:
                                new_path.reverse()
                                for index, (x,y) in enumerate(new_path):
                                    self.maze_info[x][y] = MazeState.TUNNEL(index, new_path_info)

class PathInfo:

    def __init__(self, path: list[tuple[int, int]]) -> None:
        self.path = path
        self.start = path[0]
        self.end = path[-1]
        self.length = len(path)
    
    def update(self, path: list[tuple[int, int]]):
        self.path = path
        self.start = path[0]
        self.end = path[-1]
        self.length = len(path)

class MazeStateInstance:
    def __init__(self, status, index, path_info):
        self.status: MazeState = status
        self.index: int = index
        self.path_info: 'PathInfo' = path_info

    def __eq__(self, other):
        # Compare to another StatusInstance or a Status enum member
        if isinstance(other, MazeStateInstance):
            return self.status == other.status and self.path_info == other.path_info and self.index == other.index
        elif isinstance(other, MazeState):
            return self.status == other
        return False
