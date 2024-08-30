import logging
import time
from typing import Tuple

import util
from game import Actions, Agent, Directions, Grid
from logs.search_logger import log_function
from pacman import GameState


class q1c_problem:
    """
    A search problem associated with finding a path that collects all of the
    food (dots) in a Pacman game.
    Some useful data has been included here for you
    """
    def __str__(self):
        return str(self.__class__.__module__)

    def __init__(self, gameState: GameState):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.startingGameState: GameState = gameState
        self.walls = gameState.getWalls()
        self.foods = gameState.getFood()

        # unvisited_food = self.dfs(gameState.getFood().deepCopy())
        self.unreachable = False
        # if len(unvisited_food) > 0:
        #     self.unreachable = True

        valid_food = set(self.foods.asList())
        self.startState = (gameState.getPacmanPosition(), list(valid_food))
        

    @log_function
    def getStartState(self):
        "* YOUR CODE HERE *"
        return self.startState

    @log_function
    def isGoalState(self, state):
        "* YOUR CODE HERE *"
        pacmanPosition, remainingDots = state
        return len(remainingDots) == 0

    @log_function
    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """
        "* YOUR CODE HERE *"
        successors = []
        pacmanPosition, remainingDots = state
        
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = pacmanPosition
            dx, dy = Actions.directionToVector(action)
            next_x, next_y = int(x + dx), int(y + dy)
            
            if not self.walls[next_x][next_y]:
                nextPosition = (next_x, next_y)
                nextRemainingDots = tuple(dot for dot in remainingDots if dot != nextPosition)
                nextState = (nextPosition, nextRemainingDots)
                cost = 1
                successors.append((nextState, action, cost))
        
        return successors

    def dfs(self, food_grid : Grid):
        height = food_grid.height
        width = food_grid.width

        visited = [[False for _ in range(height)] for _ in range(width)]
        stack = [self.startingGameState.getPacmanPosition()]

        while stack and len(food_grid.asList()) != 0:
            x, y = stack.pop(0)

            if visited[x][y]:
                continue

            # Mark the current node as visited
            food_grid[x][y] = False
            visited[x][y] = True

            all_directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]

            for direction in all_directions:
                next_x = x + direction[0]
                next_y = y + direction[1]

                if 0 <= next_x < width and 0 <= next_y < height and not visited[next_x][next_y] and not self.walls[next_x][next_y]:
                    stack.append((next_x, next_y))
        
        return food_grid.asList()
