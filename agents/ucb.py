from typing import List
import numpy as np
from . import Agent


class UCB(Agent):
    def __init__(self, bandits: int, arms: int, time_steps: int, c_values: List[float], q_true: np.numarray):
        '''Initialize a multi-armed bandit agent using UCB exploration.

        Args:
            bandits (int): The number of bandits (independent experiments). Since each experiment is noisy, take the 
                average to evaluate performance.
            arms (int): The number of arms of each bandit.
            time_steps (int): The number of steps we run each bandit.
            c_values (list of (float)): Different c values to test the algorithm.
            q_true (numpy array): The true average rewards for each arm. 
        '''
        super().__init__(bandits, arms, time_steps, c_values, q_true)
        self.hyperparameter_name = 'c'

    def learn(self) -> np.numarray:
        '''Perform reinforcement learning on the agent.

        Returns:
            rewards (numpy array): The array of the expected rewards over the time steps for each c value.
        '''
        return super()._learn(self._learn_c)

    def _learn_c(self, c) -> np.numarray:
        '''Perform UCB exploration with a given c value.

        Args:
            c (float): The c value.

        Returns:
            average_expected_reward (numpy array): The array of the expected rewards over the time steps.
        '''
        # Pull every arm once to prevent divide by zero in calculating UCB
        self._pull_once()

        average_expected_reward = np.zeros(self.time_steps + 1)
        for t in range(1, self.time_steps + 1):
            expected_rewards = np.zeros(self.bandits)
            for bandit in range(self.bandits):
                ucb_estimates = c * np.sqrt(np.log(t) / self.N[bandit, :])
                action = np.argmax(self.Q[bandit, :] + ucb_estimates)
                
                # Use deterministic policy
                expected_rewards[bandit] = self.q_true[bandit, action]

                # Update the action-value table and the count table
                reward = np.random.normal(self.q_true[bandit, action], 1)
                self.N[bandit, action] += 1
                self.Q[bandit, action] += (reward - self.Q[bandit, action]) / self.N[bandit, action]
            average_expected_reward[t] = np.average(expected_rewards)
        return average_expected_reward

    def _pull_once(self):
        '''Pull every arm of each bandit once.
        '''
        for bandit in range(self.bandits):
            for action in range(self.arms):
                self.N[bandit, action] += 1
                self.Q[bandit, action] += np.random.normal(self.q_true[bandit, action], 1)

    def __repr__(self):
        return 'UCB Exploration'
