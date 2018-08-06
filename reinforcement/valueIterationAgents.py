# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            currentValues = {}
            for state in self.mdp.getStates():
                #print "inside "
                if not self.mdp.isTerminal(state):
                    maxValue = None
                    actions = self.mdp.getPossibleActions(state)
                    #print "actions: " , actions
                    for action in actions:
                        if maxValue < self.computeQValueFromValues(state, action): #TODO: check whether to handle Q values
                            maxValue = self.computeQValueFromValues(state, action)
                            #print "maxValue", maxValue
                    if maxValue != None:
                        currentValues[state] = maxValue
                    else: 
                        print "ERROR FIX REQUIRED WITH VALUE"
            for state in currentValues:
                self.values[state] = currentValues[state]

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        QValue = 0
        for state_, prob in self.mdp.getTransitionStatesAndProbs(state,action):
            QValue += prob * ( self.mdp.getReward(state,action,state_) + self.discount * self.values[state_])
        
        return QValue
        #util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        maxValue = None;
        optimalAction = None;
        for action in self.mdp.getPossibleActions(state):
            if maxValue < self.computeQValueFromValues(state,action):    #TODO: check whether to handle Q values
                maxValue = self.computeQValueFromValues(state,action)
                optimalAction = action

        return optimalAction
        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for i in range(self.iterations):
            state = states[i%len(states)]
            if not self.mdp.isTerminal(state):
                maxValue = None
                for action in self.mdp.getPossibleActions(state):
                    if maxValue < self.computeQValueFromValues(state, action): #TODO: check whether to handle Q values
                        maxValue = self.computeQValueFromValues(state, action)
                self.values[state] = maxValue

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        #initialize the dictionary with key as states and values as predecessors states
        predecessors = {}
        for state in self.mdp.getStates():
            predecessors[state] = set([])
 
        #fill in the predecessors values
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    for state_, prob in self.mdp.getTransitionStatesAndProbs(state,action):
                        predecessors[state_].add(state)

        #initializing a priority queue
        priorityQueue = util.PriorityQueue()
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                maxQValue = None;
                for action in self.mdp.getPossibleActions(state):
                    if maxQValue < self.computeQValueFromValues(state,action):    #TODO: check whether to handle Q values
                        maxQValue = self.computeQValueFromValues(state,action)
                diff = abs(maxQValue - self.values[state])
                priorityQueue.push(state, -diff)

        for i in range(self.iterations):
            if not priorityQueue.isEmpty():
                state = priorityQueue.pop()
                if not self.mdp.isTerminal(state):
                    maxQValue = None
                    for action in self.mdp.getPossibleActions(state):
                        if maxQValue < self.computeQValueFromValues(state, action):    #TODO: check whether to handle Q values
                            maxQValue = self.computeQValueFromValues(state, action)
                    self.values[state] = maxQValue
                for p in predecessors[state]:
                    maxQValue = None;
                    for action in self.mdp.getPossibleActions(p):
                        if maxQValue < self.computeQValueFromValues(p,action):    #TODO: check whether to handle Q values
                            maxQValue = self.computeQValueFromValues(p,action)
                    diff = abs(maxQValue - self.values[p])
                    priorityQueue.update(p, -diff)

