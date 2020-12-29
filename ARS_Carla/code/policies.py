'''
Policy class for computing action from weights and observation vector. 
Horia Mania --- hmania@berkeley.edu
Aurelia Guy
Benjamin Recht 

Modified by Nate Cibik to work with Carla application
'''


import numpy as np
from filter import get_filter

class Policy(object):

    def __init__(self, policy_params):

        self.ob_dim = policy_params['ob_dim']
        self.ac_dim = policy_params['ac_dim']
        self.weights = np.zeros((self.ac_dim, self.ob_dim), dtype=np.float64)
        #self.weights = np.zeros(self.ac_dim)

        # a filter for updating statistics of the observations and normalizing inputs to the policies
        self.observation_filter = get_filter(policy_params['ob_filter'], 
                                             shape=(self.ob_dim,),
                                             mean=policy_params['initial_mean'],
                                             std=policy_params['initial_std'])
        #self.observation_filter = None
        self.update_filter = True
        
    def update_weights(self, new_weights):
        self.weights = new_weights
        return

    def get_weights(self):
        return self.weights

    def get_observation_filter(self):
        return self.observation_filter

    def act(self, state):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

class LinearPolicy(Policy):
    """
    Linear policy class that computes action as <w, ob>. 
    """

    def __init__(self, policy_params, initial_weights=None):
        Policy.__init__(self, policy_params)
        if initial_weights is None:
            self.weights = np.zeros((self.ac_dim, self.ob_dim), dtype=np.float64)
        else:
            self.weights = policy_params['initial_weights']

    def act(self, state, state_filter=True):
        if filter:
            state = self.observation_filter(state, update=self.update_filter)
        return np.dot(self.weights, state)

    def get_weights_plus_stats(self):
        
        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux
        
