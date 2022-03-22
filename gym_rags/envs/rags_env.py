#!/usr/bin/env python

"""
Simulate the simplified Graph Coloring environment.

Each episode is coloring a complete graph.
"""

# Core Library
import logging.config
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

    def __init__(self, start_size, max_size, red_clique, blue_clique):
        self.__version__ = "0.0.1"
        logging.info(f"RAGSEnv - Version {self.__version__}")

        self.start_size_param = start_size
        self.max_size_param = max_size
        self.red_clique_param = red_clique
        self.blue_clique_param = blue_clique
        self.max_nodes = 0
        self.curr_size = 0
        self.num_success = 0
        self.is_red_clique_found = False
        self.is_blue_clique_found = False
        self.is_done = False
        self.CURRENT_EDGES = 0
        self.MAX_EDGES = 0
        self.curr_step = -1
        self.max_nodes = 0
        self.red_clique_size = 0
        self.blue_clique_size = 0
        self.state = None
        self.red_graph = None
        self.blue_graph = None
        self._init()

        # Generate indices from state into graph matrix
        self.indices = {}
        counter = 0
        for k1 in range(0, self.MAX_EDGES):
            for k2 in range(0, k1):
                self.indices[counter] = (k1, k2)
                counter = counter + 1

        # Store what the agent tried
        self.curr_episode = -1
        # self.action_episode_memory: List[Any] = []

    def _init(self):
        self.max_nodes = self.max_size_param
        self.curr_size = self.start_size_param
        self.num_success = 0
        self.is_red_clique_found = False
        self.is_blue_clique_found = False
        self.is_done = False
        self.CURRENT_EDGES = (self.curr_size * (self.curr_size - 1)) // 2
        self.MAX_EDGES = (self.max_nodes * (self.max_nodes - 1)) // 2

        # Every edge can be RED - 0 or BLUE - 1
        self.action_space = spaces.MultiDiscrete([self.MAX_EDGES, 2])
        self.observation_space = spaces.Dict({"current": spaces.Box(low=np.zeros(self.CURRENT_EDGES),
                                                                    high=np.ones(self.CURRENT_EDGES) * 2,
                                                                    dtype=np.int8),
                                              "universe": spaces.Box(low=np.zeros(self.MAX_EDGES),
                                                                     high=np.ones(self.MAX_EDGES) * 2,
                                                                     dtype=np.int8),
                                              "red_clique_found": spaces.MultiBinary(1),
                                              "blue_clique_found": spaces.MultiBinary(1),
                                              })

        self.state = np.zeros(self.MAX_EDGES)
        self.red_graph = networkx.convert_matrix.from_numpy_array(np.zeros((self.max_nodes, self.max_nodes)))
        self.blue_graph = networkx.convert_matrix.from_numpy_array(np.zeros((self.max_nodes, self.max_nodes)))
        self.red_clique_size = self.red_clique_param
        self.blue_clique_size = self.blue_clique_param

    def step(self, action: Tuple[int, int]):
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
        if self.is_done:
            raise RuntimeError("Episode is done")
        reward = self._take_action(action)
        obs = dict(current=self.state[:self.CURRENT_EDGES].tolist(),
                   universe=self.state.tolist(),
                   red_clique=self.is_red_clique_found,
                   blue_clique=self.is_blue_clique_found)
        return obs, reward, self.is_done, dict(red_graph=self.red_graph, blue_graph=self.blue_graph)

    def _take_action(self, action: Tuple[int, int]):
        cell_idx = action[0]
        action_idx = action[1] + 1
        reward = 0
        idx = None
        if cell_idx < self.MAX_EDGES:
            idx = self.indices[cell_idx]

        is_same = self.state[cell_idx] == action_idx
        recolored = self.state[cell_idx] > 0

        if cell_idx >= self.CURRENT_EDGES:
            reward = -10  # punish for coloring edge beyond current edge
        elif action_idx not in [1, 2]:
            reward = -10  # punish for wrong color (if that ever happens)
        elif is_same:
            reward = -10
        else:
            self.state[cell_idx] = action_idx
            self._color_edge(idx[0], idx[1], action_idx)
            if recolored:
                if self.is_red_clique_found or self.is_blue_clique_found:
                    reward = -100
                    self.done = True
            else:
                if not self.is_red_clique_found and not self.is_blue_clique_found:
                    if np.all(self.state[:self.CURRENT_EDGES].astype(bool)):
                        self.num_success = np.count_nonzero(self.state[:self.CURRENT_EDGES])

                        if self.curr_size < self.max_nodes:
                            reward = self.CURRENT_EDGES  # successfully colored all edges so far without clique
                            self.red_graph.add_node(self.curr_size)
                            self.blue_graph.add_node(self.curr_size)
                            self.curr_size += 1
                            self.CURRENT_EDGES = (self.curr_size * (self.curr_size - 1)) // 2
                        else:
                            reward = self.MAX_EDGES * 100  # successful reached goal
                            self.is_done = True
                    else:
                        reward = 1
                else:
                    reward = -100
                    self.done = True
        return reward

    def _color_edge(self, n1, n2, color):
        if color == 1:
            g = self.red_graph
            color_char = 'r'
        elif color == 2:
            g = self.blue_graph
            color_char = 'b'
        else:
            assert "Invalid color"

        if not g.has_edge(n1, n2):
            g.add_edge(n1, n2, color=color_char)
        else:
            networkx.set_edge_attributes(g, {(n1, n2): {"color": color_char}})
        self.is_red_clique_found = clique.graph_clique_number(self.red_graph) >= self.red_clique_size
        self.is_blue_clique_found = clique.graph_clique_number(self.blue_graph) >= self.blue_clique_size

    def reset(self, seed=42):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation: List[int]
            The initial observation of the space.
        """
        self._init()
        np.random.seed(seed)
        self.state[:] = 0
        self._build_graph()
        obs = dict(current=self.state[:self.CURRENT_EDGES].tolist(),
                   universe=self.state.tolist(),
                   red_clique=self.is_red_clique_found,
                   blue_clique=self.is_blue_clique_found)
        info = dict(red_graph=self.red_graph, blue_graph=self.blue_graph)
        return obs, info

    def _build_graph(self):
        counter = 0
        for idx in self.state[:self.CURRENT_EDGES]:
            n1, n2 = self.indices[counter]
            counter += 1
            if idx == 1:
                self.red_graph.add_edge(n1, n2, color='r')
            elif idx == 2:
                self.blue_graph.add_edge(n1, n2, color='b')
            else:
                pass

        self.is_red_clique_found = clique.graph_clique_number(self.red_graph) >= self.red_clique_size
        self.is_blue_clique_found = clique.graph_clique_number(self.blue_graph) >= self.blue_clique_size

    def render(self, mode="human"):
        g = networkx.compose(self.red_graph, self.blue_graph)
        edges = g.edges()
        colors = [g[u][v]['color'] for u, v in edges]
        networkx.draw_circular(g, edge_color=colors)
