import logging
import random

import util 
from game import Actions, Agent, Directions, Grid, AgentState
from logs.search_logger import log_function
from pacman import GameState
from util import manhattanDistance
from enum import Enum
import random, time, math

class MazeState(Enum):
    CORNER = 0.2
    TUNNEL = 0.5
    DEAD_END = 0.9

    def __call__(self, index=None, path: 'PathInfo' = None):
        return MazeStateInstance(self, index, path)
    
def score_evaluation_food(currentGameState: GameState, scared_time):
    
    score = 0

    pacman_pos = currentGameState.getPacmanPosition()
    remaining_food = currentGameState.getFood().asList()    

    if not remaining_food:
        return 0

    # Manhattan distance from Pacman to each food
    food_dist = [manhattanDistance(pacman_pos, foodPos) for foodPos in remaining_food] 

    # Calculate reciprocal of the sum of food distances
    reciprocal_food_distance = 0
    if sum(food_dist) > 0:
        reciprocal_food_distance = 10/ sum(food_dist)  # Adjust the scaling factor as needed

    # Incorporate the distance to the closest food
    closest_food_distance = min(food_dist) if food_dist else 1  # Avoid division by zero

    # Score calculation
    food_score = (5 / closest_food_distance)  # Reward for being closer to food

    # Apply a penalty if food is too far away
    distance_penalty = -0.1 * sum(food_dist) if sum(food_dist) > max(currentGameState.getWalls().width, currentGameState.getWalls().height) else 0  # Adjust the threshold

    # Combine everything
    if scared_time > 0:
        return 0
    
    score += + food_score + distance_penalty + reciprocal_food_distance

    return score

def score_evaluation_ghost(currentGameState: GameState):

    pacman_pos = currentGameState.getPacmanPosition()
    score = 0

    # get the distance from pacman to each ghost, apply penalty, the nearest the ghost, the higher the penalty
    ghost_state: list[AgentState] = currentGameState.getGhostStates()
    total_ghost_dist = 0
    ghost_dist_penalty = 0
    nearest_ghost_dist = float('inf')
    for ghost in ghost_state:
        distance = manhattanDistance(pacman_pos, ghost.getPosition())
        nearest_ghost_dist = min(nearest_ghost_dist, distance)
        total_ghost_dist += distance
        ghost_dist_penalty -= 5/ (distance + 1)

    # Get the scared timer of the ghost, and take the sum of it 
    scared_times = [ghost.scaredTimer for ghost in ghost_state]
    total_scared_time = sum(scared_times)

    if total_scared_time > 0 :  # means that the ghost can be eaten
        score += total_scared_time + abs(ghost_dist_penalty*100)
    else:
        score += (ghost_dist_penalty)

    return score, total_scared_time, nearest_ghost_dist

def score_evaluation_capsule(currentGameState: GameState, ghost_scared_timer):
    
    pacman_pos = currentGameState.getPacmanPosition()
    score = 0

    # get the distance from pacman to each ghost, apply penalty, the nearest the ghost, the higher the penalty
    total_capsule_dist = 0
    capsule_dist_reward = 0
    for capsule in currentGameState.getCapsules():
        distance = manhattanDistance(pacman_pos, capsule)
        total_capsule_dist += distance
        capsule_dist_reward += 1/ (distance + 1)

    if ghost_scared_timer > 0 :  # means that the ghost can be eaten
        score -= capsule_dist_reward 
    else:
        score += (capsule_dist_reward)

    return score

def penalty_frequency_visit (currentGameState: GameState, visit_freq):
    
    pacman_pos = currentGameState.getPacmanPosition()

    visit_count = visit_freq[pacman_pos[0]][pacman_pos[1]]
    return min(10 * (visit_count ** 2), 490)

def score_evaluation_dead_end(currentGameState: GameState, maze_info: list[list['MazeStateInstance']], nearest_ghost, scared_time):

    pacman_pos = currentGameState.getPacmanPosition()

    # get the state of the pacman position on the maze, (help to determine weather it is a dead end or tunnel or corner)
    pos_maze_state: MazeStateInstance = maze_info[pacman_pos[0]][pacman_pos[1]]
    maze_state_score = 0    

    # check is pacman in a dead end path
    if pos_maze_state == MazeState.DEAD_END:

        if pos_maze_state.path_info.total_food == 0:

            # apply a harsh penalty to avoid pacman from entering the dead end, the deeper the end, the higher the penalty, 
            # can assume the opening of the dead end is a wall, so that pacman cannot go in
            if pos_maze_state.path_info.length == 1:
                maze_state_score -= 500
                return maze_state_score
            
            maze_state_score -= pos_maze_state.index * 200 # similar penalty of eaten by ghost
            return maze_state_score

        # if there are only food left in dead end, just move in the dead end
        if pos_maze_state.path_info.total_food == len(currentGameState.getFood().asList()):
            maze_state_score += 500

        # below the section is the condition for the is food in the dead end path
        escape_move = pos_maze_state.index + 1
        eat_move = (pos_maze_state.index + 1) + (pos_maze_state.path_info.length - (pos_maze_state.index + 1)) * 2

        # if there is enough move to comsume the food in-and-out, give a high reward to complete it, like real high
        if nearest_ghost > eat_move or scared_time > 0: 
            maze_state_score += 500
        elif nearest_ghost > escape_move + 3:
            maze_state_score += (nearest_ghost - escape_move) * 10
        else:
            maze_state_score -= 500

    return maze_state_score

def scoreEvaluationFunction(currentGameState: GameState, maze_info: list[list['MazeStateInstance']], visit_freq):

    # initial score 
    score = currentGameState.getScore()

    # Get Pacman position and relevant game state information
    pacman_pos = currentGameState.getPacmanPosition()

    ghost_score, ghost_scared, nearest_ghost = score_evaluation_ghost(currentGameState)
    food_score = score_evaluation_food(currentGameState, ghost_scared)
    capsule_score = score_evaluation_capsule(currentGameState, ghost_scared)
    freq_visit_penalty = penalty_frequency_visit(currentGameState, visit_freq)
    dead_end_score = score_evaluation_dead_end(currentGameState, maze_info, nearest_ghost, ghost_scared)
    # tunnel_score = score_evaluation_tunnel(currentGameState, maze_info, visit_freq, nearest_ghost, ghost_scared)

    score += food_score + ghost_score + capsule_score - freq_visit_penalty + dead_end_score

    if currentGameState.isWin():
        score = 99999
    elif currentGameState.isLose():
        score = -99999

    return score

class Q2_Agent(Agent):

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '3'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.maze_info: list[list[MazeStateInstance]] = None
        self.time_limit = 29
    
    @log_function
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action with alpha-beta pruning from the current gameState
        using self.depth and self.evaluationFunction.
        """

        def alphaBeta(state, depth, agentIndex, alpha, beta):
            if time.time() - self.start_time > self.time_limit - 1:
                return self.evaluationFunction(state, self.maze_info, self.visit_freq)

            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state, self.maze_info, self.visit_freq)

            if agentIndex == 0:  # Pacman's turn
                return maxValue(state, depth, alpha, beta)
            else:  # Ghosts' turn
                return minValue(state, depth, agentIndex, alpha, beta)

        def maxValue(state, depth, alpha, beta):
            best_value = -math.inf
            actions = state.getLegalActions(0)
            for action in actions:
                successor = state.generateSuccessor(0, action)
                value = alphaBeta(successor, depth, 1, alpha, beta)
                best_value = max(best_value, value)
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break  # Beta cutoff
            return best_value

        def minValue(state, depth, agentIndex, alpha, beta):
            best_value = math.inf
            next_agent = agentIndex + 1
            if agentIndex == state.getNumAgents() - 1:  # Last ghost, then Pacman's turn
                next_agent = 0
                depth -= 1
            actions = state.getLegalActions(agentIndex)
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                value = alphaBeta(successor, depth, next_agent, alpha, beta)
                best_value = min(best_value, value)
                beta = min(beta, best_value)
                if beta <= alpha:
                    break  # Alpha cutoff
            return best_value

        if self.maze_info is None:
            self.check_maze_info(gameState)
            self.visit_freq = [[0 for _ in range(gameState.getWalls().height)] for _ in range(gameState.getWalls().width)]
            
        pacman_pos = gameState.getPacmanPosition()
        self.start_time = time.time()

        best_action = None
        best_value = -math.inf
        alpha = -math.inf
        beta = math.inf
        legal_actions = gameState.getLegalActions(0)
            
        for action in legal_actions:
            value = alphaBeta(gameState.generateSuccessor(0, action), self.depth - 1, 1, alpha, beta)
            if value > best_value:
                best_value = value
                best_action = action

        # update the maze information for the action choosen
        dx, dy = Actions.directionToVector(best_action)
        next_x = int(pacman_pos[0]+dx)
        next_y = int(pacman_pos[1]+dy)
        
        self.visit_freq[next_x][next_y] += 1
        if gameState.hasFood(next_x, next_y):
            maze_state = self.maze_info[next_x][next_y]
            if maze_state is not None and (maze_state == MazeState.DEAD_END or maze_state == MazeState.TUNNEL):
                maze_state.path_info.total_food -= 1

        return best_action

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
                path_info = PathInfo([(x,y)])
                if game_state.hasFood(x,y):
                    path_info.total_food += 1
                self.maze_info[x][y] =  MazeState.DEAD_END(0, path_info)
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
                        visited[next_x][next_y] = True
                        new_path = path[:]
                        new_path.append((next_x, next_y))
                        new_path_info = PathInfo(new_path)
                        for index, (x,y) in enumerate(new_path):
                            self.maze_info[x][y] = MazeState.DEAD_END(index, new_path_info)
                            if gs.hasFood(x,y):
                                new_path_info.total_food += 1

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
                                    self.maze_info[x][y] = MazeState.TUNNEL(index, existing_path_info)
                                    if gs.hasFood(x,y):
                                        existing_path_info.total_food += 1
                            else:
                                new_path.reverse()
                                for index, (x,y) in enumerate(new_path):
                                    self.maze_info[x][y] = MazeState.TUNNEL(index, new_path_info)
                                    if gs.hasFood(x,y):
                                        new_path_info.total_food += 1

class PathInfo:

    def __init__(self, path: list[tuple[int, int]]) -> None:
        self.path = path
        self.start = path[0]
        self.end = path[-1]
        self.length = len(path) 
        self.total_food = 0
    
    def update(self, path: list[tuple[int, int]]):
        self.path = path
        self.start = path[0]
        self.end = path[-1]
        self.length = len(path)
        self.total_food = 0

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
