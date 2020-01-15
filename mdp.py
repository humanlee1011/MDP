import csv
import sys

epsilon = 0.001
gamma = 0.9

f = [0.3, 0.3, 0.4]
g = [0.2, 0.3, 0.5]   
m = 2 + 1
n = 2 + 1

def transition_prob():
    """
        transition probability is a dictionary with (x, v) key
        where x is the number of data packages in the server and v is the number of data packages in the queue
    """
    transition = dict()
    for x in range(0, m):
        for v in range(0, n):
            prob = dict()
            for u in range(0, m):
                # import pdb; pdb.set_trace()
                s = (x, v)
                # transition[(0,0)][u] = 1
                prob[u] = calculate_transition_prob(s, u)
            transition[s] = prob
    return transition

# 2 * 2
transition = [
    []
]
transition 

def calculate_transition_prob(s, u):
    """given state and control, calculate the corresponding out-state and probability"""
    probs = {}
    for x in range(0, m):
        for v in range(0, n):
            if (v - s[1] + u >= m or v - s[1] + u < 0) or (s[0] + u - x < 0 or s[0] + u - x >= m):
                probs[(x, v)] = 0
            elif v > u + s[0] and u + s[0] >= 0:
                probs[(x, v)] = 0
            elif x == 0 and u + s[0] == 0:
                probs[(x, v)] = g[v - s[1] + u]
            elif v == m - 1 and u + s[0] > 0:
                probs[(x, v)] = f[s[0] + u - x] * sum(g[i] for i in range(v - s[1] + u, m))
            elif u + s[0] > 0:
                probs[(x, v)] = f[s[0] + u - x] * g[v - s[1] + u]
            # print(u, s, x, v)
    return probs
            


class MarkovDecisionProcesses:
    """
        Markov decision processes consists of quadtuples:
        S: states sets
        U: control sets
        g: cost function
        p_ij(u): transition probablities for a given control
    """
    def __init__(self, transition={}, control={}, gamma=.9):
        # collect all nodes from the transition models
        self.states = transition.keys()
        # initialize transition
        self.transition = transition
        # initialize control sets
        self.controls = control
        
    def get_transition_prob(self, state, control):
        """for a in-state and action, return a list pair of (prob, out-state). This is for the sake of iteration"""
        return transition[state][control]
    

    def cost_function(self, state):
        (x, v) = state
        """given the number of waiting data packages and processing data packages, return the storage cost and processing cost of them"""
        return self.storage_cost(x) + self.processing_cost(v)
    
    def storage_cost(self, x):
        """return storage cost of data packages waiting in the queue"""
        return x
    
    def processing_cost(self, v):
        """return processing cost of data packages in the server"""
        return v
    
    # def control(self):
    #     return self.control


def value_iteration(mdp):
    """
    Solving the MDP by value iteration.
    returns utility values for states after convergence
    """
    # initialize value of all the states to 0 (this is k=0 case)
    J1 = {s: 1000000 for s in mdp.states}
    mu = {s: -1 for s in mdp.states}
    while True:
        J = J1.copy()
        delta = 0
        for s in mdp.states:
            # import pdb; pdb.set_trace()
            # Bellman update, update the minimum cost values
            # J1[s] = mdp.cost_function(s) + min([sum([p * J[s1] for (s1, p) in mdp.get_transition_prob(s, u1).items()]) for u1 in mdp.controls])
            for u1 in mdp.controls:
                temp = mdp.cost_function(s) + sum([p * J[s1] for (s1, p) in mdp.get_transition_prob(s, u1).items()])
                if temp < J1[s]:
                    mu[s] = u1
                    J1[s] = temp
            # calculate maximum difference in value
            delta = max(delta, abs(J1[s] - J[s]))

        #check for convergence, if values converged then return J
        if delta < epsilon:
            return J, mu


def optimal_policy(J):
    """
    Given an MDP and a utility values J, determine the optimal policy as a mapping from state to action.
    returns policies which is dictionary of the form {state1: action1, state2: action2}
    """
    mu = {}
    for s in mdp.states:
        mu[s] = min(mdp.controls, key=lambda u: expected_utility(mdp, u, s, J))
    return mu


def expected_utility(mdp, u, s, J):
    """returns the expected utility of doing u in state s, according to the MDP and J."""
    return mdp.cost_function(s) + sum([p * J[s1] for (s1, p) in mdp.get_transition_prob(s, u).items()])

transition = transition_prob()
mdp = MarkovDecisionProcesses(transition, control={0,1,2})
J, mu = value_iteration(mdp)
print("Value Iteration results:\nState\tValue")
for state, value in J.items():
    print(state, '\t', value)

# mu = optimal_policy(J)
print("Optimal Policies:\nState\tControl\n")
for i in mdp.states:
    print(i, '\t', mu[i]) 
