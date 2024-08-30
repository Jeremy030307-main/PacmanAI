import logging
import time
from typing import Tuple

import util
from game import Actions, Agent, Directions
from logs.search_logger import log_function
from pacman import GameState


class q1b_problem:
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
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
        self.goalPoints = gameState.getFood().asList()
        self.start_pos = gameState.getPacmanPosition()


    @log_function
    def getStartState(self):
        return self.start_pos

    @log_function
    def isGoalState(self, state):
        # state is the position (x,y) of pacman 
        if state in self.goalPoints:
            self.goalPoints.remove(state)
            self.start_pos = state
            return True
        else:
            return False

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

