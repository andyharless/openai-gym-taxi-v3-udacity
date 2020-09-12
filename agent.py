import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, alpha=.75, gamma=1, beta=.8, c1=0, c2=0,
                 get_epsilon=lambda i: .8*i**.999,
                 get_alpha=None, get_gamma=None, get_beta=None,
                 get_c1=None, get_c2=None):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        - alpha: default learning rate
        - gamma: defuult discount rate
        - beta: default decay rate of path memory
        - c1: default weight of value function in stochastic action distribution
        - c2: default (inverse) weight of path memory in stochastic action distribution
        - get_epsilon: function to choose epsilon (action stocahsticity) given episode number
        - get_alpha: function to choose alpha (learning rate) given episode number
        - get_gamma: function to choose gamma (discount rate) given episode number
        - get_beta: function to choose beta (decay rate of path memory) given episode number
        - c1: function to choose weight of value function in stochastic action distribution given episode number
        - c2: function to choose (inverse) weight of path memory in stochastic action distribution given episode number
        
        This agent learns using a variation on standard Q-learning.  The policy is epsilon-greedy,
        but when the non-greedy action is chosen, instead of being sampled from a uniform distribution,
        it is sampled from a distribution that reflects two things:
           - a preference for actions with higher Q values (i.e. "greedy but flexible")
           - a preference for novel actions (those that have recently been less often chosen in the current state)
        The latter are tracked via a "path memory" table "self.recent" (same shape as the Q table),
          which counts how often each action is taken in each state.
        At the end of each episode, path memories from the previous episode decay acccording to parameter "beta".
        The sampling distribution for stochastic actions is the softmax of a linear combination
          of the Q values (with coefficient "c1") and the path memory values (with coefficient negative "c2")
        
        """
        
        # Parameter values if they are to be constant
        self.alpha_init = alpha
        self.gamma_init = gamma
        self.beta_init = beta
        self.c1_init = c1
        self.c2_init = c2

        # Parameter values if we allow them to change for each episode
        self.get_epsilon = get_epsilon
        self.get_alpha = (lambda i:self.alpha_init) if get_alpha is None else get_alpha
        self.get_gamma = (lambda i:self.gamma_init) if get_gamma is None else get_gamma
        self.get_beta = (lambda i:self.beta_init) if get_beta is None else get_beta
        self.get_c1 = (lambda i:self.c1_init) if get_c1 is None else get_c1
        self.get_c2 = (lambda i:self.c2_init) if get_c2 is None else get_c2
         
        # Size of action space
        self.nA = nA
        
        # Initialize action-value table and path memory table
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.recent = defaultdict(lambda: np.zeros(self.nA))
         
        # Starting paraemter values
        self.epsilon = self.get_epsilon(0)
        self.alpha = self.get_alpha(0)
        self.gamma = self.get_gamma(0)
        self.beta = self.get_beta(0)
        self.c1 = self.get_c1(0)
        self.c2 = self.get_c2(0)
        self.i_episode = 0

        
    def select_action(self, state):
        """ Given the state, select an action
            using epsilon-greedy selection method

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """

        if state not in self.Q:
            # Case where there are no action values entered yet for this state
            # Just choose randomly
            retrun np.random.choice(self.nA)
                        
        q = np.asarray(self.Q[state])  # Action values for this state
        r = np.asarray(self.recent[state])  # Path memory for this state
        
        # Calculate distribution from which to sample actions for the stochastic option
        p = self.softmax(q*self.c1 - r*self.c2)  # Distribution to sample

        # Choose actions for the stochastic and non-stochastic options
        greedy_action = np.asarray(self.Q[state]).argmax()
        random_action = np.random.choice(self.nA, p=p) 

        # Choose the final action using epsilon-greedy policy
        return np.random.choice([random_action, greedy_action], 
                                p=[self.epsilon, 1-self.epsilon])
 

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple,
            according to the standard Q-learning procedure
            
            Also pdate learning parameters for next episode if current one is done

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        greedy_action = np.asarray(self.Q[next_state]).argmax()
        self.Q[state][action] += self.alpha*(reward + 
                                             self.gamma*self.Q[next_state][greedy_action] - 
                                             self.Q[state][action])
        
        if done:
            # Update parameters for the next episode
            self.i_episode += 1
            self.epsilon = self.get_epsilon(self.i_episode)
            self.alpha = self.get_alpha(self.i_episode)
            self.gamma = self.get_gamma(self.i_episode)
            self.gamma = self.get_gamma(self.i_episode)
            self.beta = self.get_beta(self.i_episode)
            self.c1 = self.get_c1(self.i_episode)
            self.c2 = self.get_c2(self.i_episode)

                        
    @staticmethod
    def softmax(a):  # Cacluate softmax values from an array of real numbers
        e = np.exp(a)
        return e/e.sum()
