#!/usr/bin/env python

"""
Simulate the simplified Graph Coloring environment.

Each episode is coloring a complete graph.
"""

# Core Library
import logging.config
import math
import random
from typing import Any, Dict, List, Tuple

# Third party
import cfg_load
import gym
import networkx
from networkx.algorithms import clique
import numpy as np
import pkg_resources
from gym import spaces

path = "config.yaml"  # always use slash in packages
filepath = pkg_resources.resource_filename("gym_rags", path)
config = cfg_load.load(filepath)
logging.config.dictConfig(config["LOGGING"])


class RAGSEnv(gym.Env):
    """
    Define a graph coloring environment.

    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self, k=(5, 3, 3)) -> None:
        self.__version__ = "0.0.1"
        logging.info(f"RAGSEnv - Version {self.__version__}")

        # General variables defining the graphs
        self.TOTAL_TIME_STEPS = (k[0] * (k[0] - 1)) // 2
        self.curr_step = 0
        self.is_graph_colored = False
        self.is_clique_found = False

        # Every edge can be RED - 0 or GREEN - 1
        self.action_space = spaces.Discrete(2)

        space_dim = [3]*self.TOTAL_TIME_STEPS
        self.observation_space = spaces.MultiDiscrete(space_dim)
        self.state = np.zeros(self.TOTAL_TIME_STEPS)
        self.red_graph = networkx.convert_matrix.from_numpy_array(np.zeros((k[0], k[0])))
        self.green_graph = networkx.convert_matrix.from_numpy_array(np.zeros((k[0], k[0])))
        self.red_clique_size = k[1]
        self.green_clique_size = k[2]

        # Generate indices from state into graph matrix
        self.indices = {}
        counter = 0
        for k1 in range(0, k[0]):
            for k2 in range(0, k1+1):
                self.indices[counter] = (k1+1, k2)
                counter = counter + 1

        # Store what the agent tried
        self.curr_episode = -1
        self.action_episode_memory: List[Any] = []

    def step(self, action: int) -> Tuple[List[int], float, bool, Dict[Any, Any]]:
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : int

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob : List[int]
                an environment-specific object representing your observation of
                the environment.
            reward : float
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over : bool
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info : Dict
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        if self.is_graph_colored:
            raise RuntimeError("Episode is done")
        self.curr_step += 1
        self._take_action(action)
        reward = self._get_reward()
        return self.state.tolist(), reward, self.is_graph_colored | self.is_clique_found, {}

    def _take_action(self, action: int) -> None:
        self.action_episode_memory[self.curr_episode].append(action)
        self.state[self.curr_step - 1] = action + 1
        idx = self.indices[self.curr_step - 1]

        if action == 0:
            self.red_graph.add_edge(idx[0], idx[1], color=0)
        else:
            self.green_graph.add_edge(idx[0], idx[1], color=1)

        self.is_graph_colored = self.curr_step == self.TOTAL_TIME_STEPS
        rc = clique.graph_clique_number(self.red_graph)
        gc = clique.graph_clique_number(self.green_graph)
        self.is_clique_found = rc >= self.red_clique_size | gc >= self.green_clique_size

    def _get_reward(self) -> float:
        """Reward is given for a colored edge."""
        if self.is_graph_colored and not self.is_clique_found:
            return 1000
        elif self.is_clique_found:
            return 0
        else:
            return 1

    def reset(self) -> List[int]:
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation: List[int]
            The initial observation of the space.
        """
        self.curr_step = 0
        self.curr_episode += 1
        self.action_episode_memory.append([])
        self.is_graph_colored = False
        self.is_clique_found = False
        self.state = np.zeros(self.TOTAL_TIME_STEPS)
        return self.state.tolist()

    def render(self, mode="human"):
        g = networkx.compose(self.red_graph, self.green_graph)
        networkx.draw_circular(g)
