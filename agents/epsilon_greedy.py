from typing import List, Tuple
import numpy as np
from . import Agent


class EpsilonGreedy(Agent):
    def __init__(self, bandits: int, arms: int, time_steps: int, epsilons: List[float], q_true: np.numarray) -> None:
        '''Initialize a multi-armed bandit agent using epsilon-greedy exploration.

        Args:
            bandits (int): The number of bandits (independent experiments). Since each experiment is noisy, take the 
                average to evaluate performance.
            arms (int): The number of arms of each bandit.
            time_steps (int): The number of steps we run each bandit.
            epsilons (list of (float)): Different epsilon values to test the algorithm.
            q_true (numpy array): The true average rewards for each arm. 
        '''
        super().__init__(bandits, arms, time_steps, epsilons, q_true)
        self.hyperparameter_name = 'epsilon'

    def learn(self) -> np.numarray:
        '''Perform reinforcement learning on the agent.

        Returns:
            rewards (numpy array): The array of the expected rewards over the time steps for each epsilon value.
        '''
        return super()._learn(self._learn_epsilon)

    def _learn_epsilon(self, epsilon: float) -> np.numarray:
        '''Perform epsilon-greedy exploration with a given epsilon value.

        Args:
            epsilon (float): The epsilon value.

        Returns:
            average_expected_reward (numpy array): The array of the expected rewards over the time steps.
        '''
        average_expected_reward = np.zeros(self.time_steps + 1)
        for t in range(1, self.time_steps + 1):
            expected_rewards = np.zeros(self.bandits)
            for bandit in range(self.bandits):
                max_Q = np.max(self.Q[bandit, :])
                num_ties, action = self._get_action(bandit, max_Q, epsilon)

                expected_rewards[bandit] = self._get_expected_reward(bandit, max_Q, num_ties, epsilon)

                # Update the action-value table and the count table
                reward = np.random.normal(self.q_true[bandit, action], 1)
                self.N[bandit, action] += 1
                self.Q[bandit, action] += (reward - self.Q[bandit, action]) / self.N[bandit, action]
            average_expected_reward[t] = np.average(expected_rewards)
        return average_expected_reward

    def _get_action(self, bandit: int, max_Q: float, epsilon: float) -> Tuple[int, int]:
        '''Returns the next action.

        Args:
            bandit (int): The index of the current bandit.
            max_Q (float): The current maximum action-value among the arms of the bandit.
            epsilon (float): The epsilon value.

        Returns:
            i, action (tuple of (int, int)): The tuple of the number of actions that have the equal action-value 
                with the next action, and the index of the next action.
        '''
        if np.random.rand() < epsilon:
            return 1, np.random.randint(self.arms)  # Exploration

        # Exploitation (break ties randomly)
        actions_sorted = np.argsort(-self.Q[bandit, :])
        i = 0
        for action in actions_sorted:
            if self.Q[bandit, action] < max_Q:
                break
            i += 1
        return i, np.random.choice(actions_sorted[:i])

    def _get_expected_reward(self, bandit: int, max_Q: float, num_ties: int, epsilon: float) -> float:
        '''Calculate the expected reward at the current time step.

        Args:
            bandit (int): The index of the current bandit.
            max_Q (float): The current maximum action-value among the arms of the bandit.
            num_ties (int): The number of actions that have the equal action-value with the next action.
            epsilon (float): The epsilon value.

        Returns:
            expected_reward (float): The expected reward.
        '''
        expected_reward = 0.0
        for action in range(self.arms):
            true_reward = self.q_true[bandit, action]
            pi = epsilon / self.arms
            if self.Q[bandit, action] == max_Q:
                pi += (1 - epsilon) / num_ties
            expected_reward += true_reward * pi
        return expected_reward

    def __repr__(self):
        return 'Epsilon-Greedy Exploration'
