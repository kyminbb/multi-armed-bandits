from typing import List
import numpy as np
from . import Agent


class OptimisticInitialization(Agent):
    def __init__(self, bandits: int, arms: int, time_steps: int, initial_values: List[float], q_true: np.numarray):
        '''Initialize a multi-armed bandit agent using optimistic initialization.

        Args:
            bandits (int): The number of bandits (independent experiments). Since each experiment is noisy, take the 
                average to evaluate performance.
            arms (int): The number of arms of each bandit.
            time_steps (int): The number of steps we run each bandit.
            initial_values (list of (float)): Different initial action-values to test the algorithm.
            q_true (numpy array): The true average rewards for each arm. 
        '''
        super().__init__(bandits, arms, time_steps, initial_values, q_true)
        self.hyperparameter_name = 'initial value'

    def learn(self):
        '''Perform reinforcement learning on the agent.

        Returns:
            rewards (numpy array): The array of the expected rewards over the time steps for each initial action-value.
        '''
        return super()._learn(self._learn_initial_value)

    def _learn_initial_value(self, initial_value: float) -> np.numarray:
        '''Perform optimistic initialization with a given initial action-value.

        Args:
            epsilon (float): The initial action-value.

        Returns:
            average_expected_reward (numpy array): The array of the expected rewards over the time steps.
        '''
        # Initialize Q values optimistically
        self.Q += initial_value

        average_expected_reward = np.zeros(self.time_steps + 1)
        for t in range(1, self.time_steps + 1):
            expected_rewards = np.zeros(self.bandits)
            for bandit in range(self.bandits):
                action = np.argmax(self.Q[bandit, :])
                
                # Use deterministic policy
                expected_rewards[bandit] = self.q_true[bandit, action]

                # Update the action-value table and the count table
                reward = np.random.normal(self.q_true[bandit, action], 1)
                self.N[bandit, action] += 1
                self.Q[bandit, action] += (reward - self.Q[bandit, action]) / self.N[bandit, action]
            average_expected_reward[t] = np.average(expected_rewards)
        return average_expected_reward

    def __repr__(self):
        return 'Optimistic Initialization'
