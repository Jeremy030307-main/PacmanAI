import logging
import time
from typing import Tuple

import util
from game import Actions, Agent, Directions
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

    @log_function
    def getStartState(self):

        self.walls = self.startingGameState.getWalls()
        
        # set the goal position for the pacman be the position of the single food
        self.food = self.startingGameState.getFood().asList()

        # for x, food_list in enumerate(food):
        #     for y, food_status in enumerate(food_list):
        #         if food_status == True:
        #             self.foods.append((x,y))

        return self.startingGameState.getPacmanPosition()

    @log_function
    def isGoalState(self, state):
        # state is the a list of index that represent food in the map
        return len(state) == 0

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
        # ------------------------------------------
        
        successors = []

        for action in [Directions.EAST, Directions.WEST, Directions.SOUTH, Directions.NORTH, Directions.STOP]:
            x,y = state
            dx, dy = Actions.directionToVector(action)  # this return (-1,0,1) which sum to the (x,y) which can determine the direction each action lead
            next_state = (int(x+dx), int(y+dy))
            cost = 1
            if action == Directions.STOP:
                cost = 0
            
            if not self.walls[next_state[0]][next_state[1]]:
                successors.append( (next_state, action, cost) )
        
        return successors

