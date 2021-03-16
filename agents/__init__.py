from typing import Any, Callable, List
import numpy as np


class Agent:
    def __init__(self,
                 bandits: int,
                 arms: int,
                 time_steps: int,
                 hyperparameters: List[float],
                 q_true: np.numarray) -> None:
        '''Initialize a multi-armed bandit agent.

        Args:
            bandits (int): The number of bandits (independent experiments). Since each experiment is noisy, take the 
                average to evaluate performance.
            arms (int): The number of arms of each bandit.
            time_steps (int): The number of steps we run each bandit.
            hyperparameters (list of (float)): Different hyperparameters to test the algorithm.
            q_true (numpy array): The true average rewards for each arm. 
        '''
        self.bandits = bandits
        self.arms = arms
        self.time_steps = time_steps
        self.hyperparameters = hyperparameters
        self.q_true = q_true

    def _reset(self) -> None:
        '''Reset the action-value table and the count table.
        '''
        self.Q = np.zeros((self.bandits, self.arms))
        self.N = np.zeros((self.bandits, self.arms))

    def _learn(self, fn: Callable[[Any, float], np.numarray]) -> np.numarray:
        '''Perform reinforcement learning on the agent.

        Args:
            fn (function): The exploration algorithm to use for the learning.

        Returns:
            rewards (numpy array): The array of the expected rewards over the time steps for each hyperparameter.
        '''
        rewards = np.zeros((len(self.hyperparameters), self.time_steps + 1))
        for i, hyperparameter in enumerate(self.hyperparameters):
            self._reset()
            rewards[i, :] = fn(hyperparameter)
        return rewards
