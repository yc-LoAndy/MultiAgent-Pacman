# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from util import Counter
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        oldFood = currentGameState.getFood()
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newGhostPos = successorGameState.getGhostPositions()

        "*** YOUR CODE HERE ***"
        # The evaluation should consider the following 2 factors:
        # 1. The distance between pacmen and the ghost (+)
        # 2. The distance (-) and the number (-) of food
        # Food
        canEat = oldFood[newPos[0]][newPos[1]]
        foodDis = [ manhattanDistance(newPos, food) for food in newFood.asList() ]
        foodNum = successorGameState.getNumFood()
        minFoodDis = min(foodDis) if foodNum != 0 else 0
        maxFoodDis = max(foodDis) if foodNum != 0 else 0

        # Ghost
        ghostDis = min( [ manhattanDistance(newPos, ghost) for ghost in newGhostPos ] )
        isScared = newScaredTimes[0] > 6

        if not isScared:
            value = ghostDis + 40*canEat  + 30/(minFoodDis + 0.5) + 20/(maxFoodDis + 0.5) + 30/(foodNum + 0.5)
        else:
            value = 40*canEat - minFoodDis - maxFoodDis - foodNum
        return value

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        counter = Counter()
        value = self.value(gameState, counter, level=1)
        nextMove = counter.argMax()
        return nextMove
    
    def value(self, gameState: GameState, counter: Counter, level: int):
        "Return the utility value based on it is a min-node or a max-node."
        agentNum = gameState.getNumAgents()
        if (gameState.isWin()) or (gameState.isLose()):
            return self.evaluationFunction(gameState)

        if (level - 1) / agentNum == self.depth:
            return self.evaluationFunction(gameState)

        if (level - 1) % agentNum == 0:  # It's pacmen's move
            return self.max_value(gameState, counter, level)
        else:                            # It's ghost move
            return self.min_value(gameState, counter, level)
    
    def max_value(self, gameState: GameState, counter: Counter, level: int):
        "Return max value among all the successors node's value"
        value = -float("INF")
        agentIndex = 0
        legalActions = gameState.getLegalActions(agentIndex=agentIndex)
        for action in legalActions:
            successorGameState = gameState.generateSuccessor(agentIndex=agentIndex, action=action)
            value = max(value, self.value(successorGameState, counter, level + 1))
            if level == 1:
                counter[action] += value
        return value

    def min_value(self, gameState: GameState, counter: Counter, level: int):
        "Return min value among all the successros node's value"
        value = float("INF")
        agentIndex = (level - 1) % gameState.getNumAgents()
        legalActions = gameState.getLegalActions(agentIndex=agentIndex)
        for action in legalActions:
            successorGameState = gameState.generateSuccessor(agentIndex=agentIndex, action=action)
            value = min(value, self.value(successorGameState, counter, level + 1))
        return value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        counter = Counter()
        alpha = -float("INF")
        beta = float("INF")
        value = self.value(gameState, alpha, beta, counter, level=1)
        nextMove = counter.argMax()
        return nextMove

    def value(self, gameState: GameState, alpha, beta, counter: Counter, level: int):
        "Return the utility value based on it is a min-node or a max-node."
        agentNum = gameState.getNumAgents()
        if (gameState.isWin()) or (gameState.isLose()):
            return self.evaluationFunction(gameState)

        if (level - 1) / agentNum == self.depth:
            return self.evaluationFunction(gameState)

        if (level - 1) % agentNum == 0:  # It's pacmen's move
            return self.max_value(gameState, alpha, beta, counter, level)
        else:                            # It's ghost move
            return self.min_value(gameState, alpha, beta, counter, level)
    
    def max_value(self, gameState: GameState, alpha, beta, counter: Counter, level: int):
        "Return max value among all the successors node's value"
        value = -float("INF")
        agentIndex = 0
        legalActions = gameState.getLegalActions(agentIndex=agentIndex)
        for action in legalActions:
            successorGameState = gameState.generateSuccessor(agentIndex=agentIndex, action=action)
            value = max(value, self.value(successorGameState, alpha, beta, counter, level + 1))
            if level == 1:
                counter[action] += value
            if value > beta:
                return value
            alpha = max(alpha, value)
        return value

    def min_value(self, gameState: GameState, alpha, beta, counter: Counter, level: int):
        "Return min value among all the successros node's value"
        value = float("INF")
        agentIndex = (level - 1) % gameState.getNumAgents()
        legalActions = gameState.getLegalActions(agentIndex=agentIndex)
        for action in legalActions:
            successorGameState = gameState.generateSuccessor(agentIndex=agentIndex, action=action)
            value = min(value, self.value(successorGameState, alpha, beta, counter, level + 1))
            if value < alpha:   # Smaller then current best-MAX
                return value
            beta = min(beta, value)
        return value


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        counter = Counter()
        value = self.value(gameState, counter, level=1)
        nextMove = counter.argMax()
        return nextMove
    
    def value(self, gameState: GameState, counter: Counter, level: int):
        "Return the utility value based on it is a min-node or a max-node."
        agentNum = gameState.getNumAgents()
        if (gameState.isWin()) or (gameState.isLose()):
            return self.evaluationFunction(gameState)

        if (level - 1) / agentNum == self.depth:
            return self.evaluationFunction(gameState)

        if (level - 1) % agentNum == 0:  # It's pacmen's move
            return self.max_value(gameState, counter, level)
        else:                            # It's ghost move
            return self.exp_value(gameState, counter, level)
    
    def max_value(self, gameState: GameState, counter: Counter, level: int):
        "Return max value among all the successors node's value"
        value = -float("INF")
        agentIndex = 0
        legalActions = gameState.getLegalActions(agentIndex=agentIndex)
        for action in legalActions:
            successorGameState = gameState.generateSuccessor(agentIndex=agentIndex, action=action)
            value = max(value, self.value(successorGameState, counter, level + 1))
            if level == 1:
                counter[action] += value
        return value

    def exp_value(self, gameState: GameState, counter: Counter, level: int):
        "Return min value among all the successros node's value"
        value = 0
        agentIndex = (level - 1) % gameState.getNumAgents()
        legalActions = gameState.getLegalActions(agentIndex=agentIndex)
        for action in legalActions:
            successorGameState = gameState.generateSuccessor(agentIndex=agentIndex, action=action)
            value += (self.value(successorGameState, counter, level + 1)) / len(legalActions)
        return value


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Basic status
    pos = currentGameState.getPacmanPosition()
    foodnum = currentGameState.getNumFood()
    foodPos = currentGameState.getFood().asList()
    ghostPos = currentGameState.getGhostPositions()
    currentScore = currentGameState.getScore()
    ghostStates = currentGameState.getGhostStates()
    capsulePos = currentGameState.getCapsules()
    capsuleNum = len(capsulePos)

    # Distance between the pacmen and ghosts
    ghostDistances = [ manhattanDistance(pos, ghost) for ghost in ghostPos]
    ghostDistance = sum(ghostDistances)

    # Food Distances
    foodDistances = [ manhattanDistance(pos, food) for food in foodPos ]
    minFoodDis = min(foodDistances) if len(foodDistances) != 0 else 0
    avgFoodDis = sum(foodDistances) / foodnum if len(foodDistances) != 0 else 0

    # Capsule distances
    newScaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    isScared = newScaredTimes[0] >= 3

    # Evaluation
    if not isScared:
        value = (-1 / (ghostDistance + 1)) + (1 / (avgFoodDis + 0.01)) - 0.5*foodnum - 1500*capsuleNum
    else:
        value = 0.01*currentScore - 2*minFoodDis - 50*foodnum - 60*ghostDistance

    # print(value)
    return value

# Abbreviation
better = betterEvaluationFunction
