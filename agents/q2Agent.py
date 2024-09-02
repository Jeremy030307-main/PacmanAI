import logging
import random

import util 
from game import Actions, Agent, Directions, Grid
from logs.search_logger import log_function
from pacman import GameState
from util import manhattanDistance
from problems.q1b_problem import q1b_problem
from solvers.q1c_solver import q1b_solver


def scoreEvaluationFunction(currentGameState: GameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """

    # nearest_prob = q1b_problem(currentGameState)

    # # distance to nearest food
    # min_dist_food = len(q1b_solver(nearest_prob))

    # # distance to nearest ghost
    # nearest_prob.start_pos = currentGameState.getPacmanPosition()
    # nearest_prob.goalPoints = currentGameState.getGhostPositions()
    # min_dist_ghost = len(q1b_solver(nearest_prob))

    # # number of fod remaining
    # food_remaining = len(currentGameState.getFood().asList())

    # # distance to nearest capsule
    # nearest_prob.start_pos = currentGameState.getPacmanPosition()
    # nearest_prob.goalPoints = currentGameState.getCapsules()
    # min_dist_ghost = len(q1b_solver(nearest_prob))

    # ghost status


    # reward to eat a ghost


    # corner, dead end, tunnel


    # return currentGameState.getScore()
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
    """ Manhattan distance to the foods from the current state """
    foodList = newFood.asList()
    from util import manhattanDistance
    foodDistance = [0]
    for pos in foodList:
        foodDistance.append(manhattanDistance(newPos,pos))

    """ Manhattan distance to each ghost from the current state"""
    ghostPos = []
    for ghost in newGhostStates:
        ghostPos.append(ghost.getPosition())
    ghostDistance = [0]
    for pos in ghostPos:
        ghostDistance.append(manhattanDistance(newPos,pos))

    numberofPowerPellets = len(currentGameState.getCapsules())

    score = 0
    numberOfNoFoods = len(newFood.asList(False))           
    sumScaredTimes = sum(newScaredTimes)
    sumGhostDistance = sum (ghostDistance)
    reciprocalfoodDistance = 0
    if sum(foodDistance) > 0:
        reciprocalfoodDistance = 1.0 / sum(foodDistance)
        
    score += currentGameState.getScore()  + reciprocalfoodDistance + numberOfNoFoods

    if sumScaredTimes > 0:    
        score +=   sumScaredTimes + (-1 * numberofPowerPellets) + (-1 * sumGhostDistance)
    else :
        score +=  sumGhostDistance + numberofPowerPellets
    return score

class SearchData:

    def __init__(self) -> None:
        self.first_time = False  # indicate that this is the first time the game call for agent action
        self.dead_end = None
        self.tunnel = None
        self.corner = None

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
            print(self.data.dead_end, self.data.in_tunnel)
            self.data.first_time = True
        
        actions = gameState.getLegalActions(0)
        currentScore = float('-inf')
        returnAction = None
        alpha = float('-inf')
        beta = float('inf')

        for action in actions:
            nextState = gameState.generateSuccessor(0,action)
            # Next level is a min level. Hence calling min for successors of the root.
            score = self.min_value(nextState,0,alpha,beta, 1)

            # Choosing the action which is Maximum of the successors.
            if score > currentScore:
                returnAction = action
                currentScore = score
            # Updating alpha value at root.    
            if score > beta:
                return returnAction
            alpha = max(alpha,score)

        return returnAction

    def max_value(self, game_state: GameState, depth: int, alpha: int, beta:int):

        currDepth = depth + 1

        # Check for termina state
        if game_state.isWin() or game_state.isLose() or currDepth == self.depth:  
            return self.evaluationFunction(game_state)
    
        max_value = float('-inf')
        actions = game_state.getLegalActions(0)
        for action in actions:

            # generate the successor game state after taking this action
            successor= game_state.generateSuccessor(0,action)

            max_value = max (max_value, self.min_value(successor, currDepth, alpha, beta))
            if max_value > beta:
                return max_value
            alpha = max(alpha,max_value)
        return max_value

    def min_value(self, game_state: GameState, depth: int, alpha: int, beta:int, agent_index = 1):

        # Check for terminal state
        if game_state.isWin() or game_state.isLose(): 
            return self.evaluationFunction(game_state)
        
        minvalue = float('inf')
        actions = game_state.getLegalActions(agent_index)
        for action in actions:
            successor= game_state.generateSuccessor(agent_index,action)

            if agent_index == (game_state.getNumAgents()-1):
                minvalue = min(minvalue,self.max_value(successor,depth,alpha,beta))
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
                            if self.data.tunnel.get(path[0]):
                                self.data.tunnel[path[0]].append(path[:])
                            else:
                                self.data.tunnel[path[0]] = [path[:]]

                            self.data.tunnel[path[-1]] = path[:]

                            if gs.getPacmanPosition() in path:
                                self.data.in_tunnel = (path[0], path.index(gs.getPacmanPosition()))
