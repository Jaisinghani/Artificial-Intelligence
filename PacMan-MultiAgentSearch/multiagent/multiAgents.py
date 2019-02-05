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
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) #Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #assigning values to food and ghosts
        foodWeight = 10.0
        ghostWeight = 10.0


        score = successorGameState.getScore()

        # Calculating distance to food for all foods and assigning score based on weight
        foodDistance = [manhattanDistance(newPos, food) for food in newFood.asList()]
        if len(foodDistance):
            score += foodWeight / min(foodDistance)

        # Calculating distance to ghost for all foods and assigning score based on weight
        ghostDistance = manhattanDistance(newPos, newGhostStates[0].getPosition())
        if ghostDistance > 0:
            score -= ghostWeight / ghostDistance


        return score
        #return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
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

        # is this a terminal state
    def isTerminal(self, state, depth, agent):
        return depth == self.depth or \
                state.isWin() or \
                state.isLose() or \
                state.getLegalActions(agent) == 0

    # is this agent pacman
    def isPacman(self, state, agent):
        return agent % state.getNumAgents() == 0

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
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
        "*** YOUR CODE HERE ***"

        best_score, best_move = self.maxFunction(gameState, self.depth)

        return best_move

    #function to choose the move based on max score
    def maxFunction(self, gameState, depth):
        #no moves
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), "noMove"

        moves = gameState.getLegalActions()
        #min score of all the moves
        scores = [self.minFunction(gameState.generateSuccessor(self.index, move), 1, depth) for move in moves]

        #choosing the max score
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = bestIndices[0]
        return bestScore, moves[chosenIndex]

    # function to return  min score and moves
    def minFunction(self, gameState, agent, depth):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), "noMove"
        moves = gameState.getLegalActions(agent)
        if (agent != gameState.getNumAgents() - 1):
            scores = [self.minFunction(gameState.generateSuccessor(agent, move), agent + 1, depth) for move in moves]
        else:
            scores = [self.maxFunction(gameState.generateSuccessor(agent, move), (depth - 1))[0] for move in moves]
        minScore = min(scores)
        worstIndices = [index for index in range(len(scores)) if scores[index] == minScore]
        chosenIndex = worstIndices[0]
        return minScore, moves[chosenIndex]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        def abPruning(state, depth, agent, alpha=float("-inf"), beta=float("inf")):
            if agent == state.getNumAgents():
                depth += 1
                agent = 0

            if self.isTerminal(state, depth, agent):
                return self.evaluationFunction(state), None

            # if the agent is pacman then initialize max value to - infinity else min value to + infinity
            if self.isPacman(state, agent):
                return getValue(state, depth, agent, alpha, beta, float('-inf'), max)
            else:
                return getValue(state, depth, agent, alpha, beta, float('inf'), min)

        def getValue(state, depth, agent, alpha, beta, max_min_score, max_min_function):
            bestScore = max_min_score
            bestAction = None

            for action in state.getLegalActions(agent):
                successor = state.generateSuccessor(agent, action)
                score, _ = abPruning(successor, depth, agent + 1, alpha, beta)
                bestScore, bestAction = max_min_function((bestScore, bestAction), (score, action))

                if self.isPacman(state, agent):
                    if bestScore > beta:
                        return bestScore, bestAction
                    alpha = max_min_function(alpha, bestScore)
                else:
                    if bestScore < alpha:
                        return bestScore, bestAction
                    beta = max_min_function(beta, bestScore)

            return bestScore, bestAction

        _, action = abPruning(gameState, 0, 0)
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        def expectimax(state, depth, agent):
            if agent == state.getNumAgents():
                return expectimax(state, depth + 1, 0)

            if self.isTerminal(state, depth, agent):
                return self.evaluationFunction(state)

            scores = [
                expectimax(state.generateSuccessor(agent, action), depth, agent + 1)
                for action in state.getLegalActions(agent)
            ]

            #for pacman find the best move i.e. max score
            if self.isPacman(state, agent):
                return max(scores)

            #take average of all ghost move scores as we don't know how will the ghost perform
            else:
                return sum(scores) / len(scores)

        #return the pacman's move that has best score
        return max(gameState.getLegalActions(0),
                   key=lambda x: expectimax(gameState.generateSuccessor(0, x), 0, 1)
                   )

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    foodWeight = 10.0
    ghostWeight = 10.0
    scaredGhostWeight = 100.0


    score = currentGameState.getScore()

    #distance to ghosts
    ghostScore = 0
    for ghost in newGhostStates:
        distance = manhattanDistance(newPos, newGhostStates[0].getPosition())
        if distance > 0:
            #if the ghost is scared go towards it
            if ghost.scaredTimer > 0:
                ghostScore += scaredGhostWeight / distance
            #if the ghost is not scared run
            else:
                ghostScore -= ghostWeight / distance
    score += ghostScore

    #distance to food
    foodDistance = [manhattanDistance(newPos, food) for food in newFood.asList()]
    if len(foodDistance):
        score += foodWeight / min(foodDistance)

    return score

# Abbreviation
better = betterEvaluationFunction

