"""
Agent that learns using n-step SARSA, 1-step SARSA and 1-step SARSA with temporal regularization. 

Pseudo code: Pg. 147 on http://incompleteideas.net/book/RLbook2018.pdf

To do:
- Include epsilon decay policy
- Change the size of memory to n to make it more efficient

"""
import numpy as np

class AgentNstepSARSA():
    """
    Agent that does n-step SARSA. See the associated notebook for ___. 
    """

    # Setting things up
    def __init__(self, env, no_steps=3, alpha=0.05, eps=0.5, gamma=0.9, epsDecay=True, epsFinal=0.05):

        # Initializing parameters
        self.no_steps = no_steps
        self.noStates = env.observation_space.n
        self.noActions = env.action_space.n
        self.eps = eps
        self.epst = eps
        self.gamma = gamma
        self.alpha = alpha
        self.updates = 0
        self.horizon = env.horizon
        self.epsDecay = epsDecay
        self.epsFinal = epsFinal
        self.decayRate = 0.0001

        # Creating memory
        self.state_memory = np.nan * np.ones(self.horizon + 1)
        self.action_memory = np.nan * np.ones(self.horizon + 1)
        self.reward_memory = np.nan * np.ones(self.horizon + 1)

        # Initializing Q values
        self.Q = np.random.rand(self.noStates, self.noActions)
        
    def reset(self):
        self.updates = 0
        self.Q = np.random.rand(self.noStates, self.noActions)

    def clear_memory(self):
        self.state_memory = np.nan * np.ones(self.horizon + 1)
        self.action_memory = np.nan * np.ones(self.horizon + 1)
        self.reward_memory = np.nan * np.ones(self.horizon + 1)
    
    # Function to do eps-greedy exploration
    def chooseActionEps(self, presentState):
        """
        Finds the action based on epsilon greedy policy.
        """
        self.updates = self.updates + 1
        if(self.epsDecay):
            self.epst = self.epsFinal + (self.eps-self.epsFinal) * np.exp(-self.decayRate * self.updates)
        if(np.random.rand(1) < self.epst): # Explore
            return np.random.randint(self.noActions)
        else: # Exploit
            return np.argmax(self.Q[presentState, :])
    
    # Function to exploit
    def chooseActionExploit(self, presentState):
        return np.argmax(self.Q[presentState, :])
        
    # Function to do softmax exploration
    def chooseActionSoftmax(self, presentState, temp=0.1):
        pi = np.exp(self.Q[presentState, :]/temp)/np.sum(np.exp(self.Q[presentState, :]/temp))
        V = np.cumsum(pi)
        c, = np.where(V-np.random.rand(1)>0)
        return c[0]

class AgentSARSA():
    """
    Agent that does one step SARSA with temporal regularization. 
    """

    # Setting things up
    def __init__(self, env, alpha=0.05, eps=0.5, beta=0.2, lambd=0.2, gamma=0.9, epsDecay=True):

        # Initializing parameters
        self.noStates = env.observation_space.n
        self.noActions = env.action_space.n
        self.gamma = gamma
        self.alpha = alpha
        self.lambd = lambd
        self.beta = beta
        self.updates = 0
        self.eps = eps
        self.epst = eps
        self.epsDecay = epsDecay
        self.epsFinal = 0.05
        self.decayRate = 0.0001
        self.p = np.zeros(self.noActions)

        # Initializing Q values
        self.Q = np.random.rand(self.noStates, self.noActions)

    def reset(self):
        self.updates = 0
        self.Q = np.random.rand(self.noStates, self.noActions)

    def initializeP(self, state):
        self.p = self.Q[state, :]

    def intializeQ(self, init):
        self.Q = init

    # Function to do eps-greedy exploration
    def chooseActionEps(self, presentState):
        """
        Finds the action based on epsilon greedy policy.
        """
        self.updates = self.updates + 1
        if(self.epsDecay):
            self.epst = self.epsFinal + (self.eps-self.epsFinal) * np.exp(-self.decayRate * self.updates)
        if(np.random.rand(1) < self.epst): # Explore
            return np.random.randint(self.noActions)
        else: # Exploit
            return np.argmax(self.Q[presentState, :])
    
    # Function to exploit
    def chooseActionExploit(self, presentState):
        return np.argmax(self.Q[presentState, :])
        
    # Function to do softmax exploration
    def chooseActionSoftmax(self, presentState, temp=0.1):
        pi = np.exp(self.Q[presentState, :]/temp)/np.sum(np.exp(self.Q[presentState, :]/temp))
        V = np.cumsum(pi)
        c, = np.where(V-np.random.rand(1)>0)
        return c[0]

    # Function for SARSA policy updates
    def updatePolicySARSA(self, S, A, R, S2, A2, updateStates=None):
        if(updateStates==None):
            self.Q[S,A] = self.Q[S,A] + self.alpha * (R + self.gamma * ((1 - self.beta) * self.Q[S2, A2] + self.beta * self.p[A2]) - self.Q[S,A])
        else:
            if(S in updateStates):
                self.Q[S,A] = self.Q[S,A] + self.alpha * (R + self.gamma * ((1 - self.beta) * self.Q[S2, A2] + self.beta * self.p[A2]) - self.Q[S,A])
        self.p = (1 - self.lambd) * self.Q[S,:] + self.lambd * self.p