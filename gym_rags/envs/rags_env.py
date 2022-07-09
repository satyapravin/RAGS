#!/usr/bin/env python

"""
RAGS a Ramsey Graph Coloring Environment
"""

# Core Library
import logging.config
from typing import Any, Dict, List, Tuple

# Third party
import cfg_load
import gym
from gym import spaces
import itertools
from scipy.special import comb
import networkx
from networkx.algorithms import clique
import numpy as np
import pkg_resources


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

    def __init__(self, max_nodes=83):
        self.__version__ = "0.0.1"
        logging.info(f"RAGSEnv - Version {self.__version__}")
        self.max_nodes = max_nodes
        self._init()

    def _init(self):
        self.edge_index = 0
        self.is_done = False
        self.KCompletes = [(3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (4, 4), (4, 5), (4, 6), (4, 7), (5, 5), (5, 6), (5, 7)]
        self.action_space = spaces.Discrete(2)
        self.observation_space = np.zeros(comb(self.max_nodes, 2, exact=True))

    def step(self, action: Tuple[int, int]):
        if self.is_done:
            raise RuntimeError("Episode is done")

        reward = self._take_action(action)
        return self.observation_space, reward, self.is_done, dict(red_graph=self.red_graph, blue_graph=self.blue_graph)

    def _take_action(self, action: int):
        assert(0 < action_idx < 3)
        self._color_edge(self, action)
        reward = _check_if_done()
        return reward

    def _check_if_done(self):
        reward = 0
        red_clique_number = clique.graph_clique_number(self.red_graph)
        blue_clique_number = clique.graph_clique_number(self.blue_graph)
        
        for tup in self.KCompletes:
            red_clique_size = tup[0]
            blue_clique_size = tup[1]

            if red_clique_number >= red_clique_size:
                reward -= 1
            if blue_clique_number >= blue_clique_size:
                reward -= 1
                
        self.edge_index += 1            
        self.is_done = self.edge_index > len(self.observation_space)
        return reward
    
    def _color_edge(self, color):
        g = None
        if color == 1:
            g = self.red_graph
            color_char = 'r'
        else:
            g = self.blue_graph
            color_char = 'b'
        self.observation_space[self.edge_index] = color
        vindices = np.tril_indices(self.max_nodes, -1)
        n1 = vindices[0][self.edge_index]
        n2 = vindices[1][self.edge_index]
        g.add_edge(n1, n2, color=color_char)
        networkx.set_edge_attributes(g, {(n1, n2): {"color": color_char}})

    
    def reset(self, seed=42):
        self._init()
        np.random.seed(seed)
        self.observation_space = np.zeros(combin(self.curr_size, 2))
        self.red_graph = networkx.empty_graph(self.curr_size)
        self.blue_graph = networkx.empty_graph(self.curr_size)
        info = dict(red_graph=self.red_graph, blue_graph=self.blue_graph)
        return self.observation_space, info

    def render(self, mode="human"):
        g = networkx.compose(self.red_graph, self.blue_graph)
        edges = g.edges()
        colors = [g[u][v]['color'] for u, v in edges]
        networkx.draw_circular(g, edge_color=colors)
#!/usr/bin/env python

"""
RAGS a Ramsey Graph Coloring Environment
"""

# Core Library
import logging.config
from typing import Any, Dict, List, Tuple

# Third party
import cfg_load
import gym
from gym import spaces
import itertools
from scipy.special import comb
import networkx
from networkx.algorithms import clique
import numpy as np
import pkg_resources


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

    def __init__(self, max_nodes=83):
        self.__version__ = "0.0.1"
        logging.info(f"RAGSEnv - Version {self.__version__}")
        self.max_nodes = max_nodes
        self._init()

    def _init(self):
        self.edge_index = 0
        self.is_done = False
        self.KCompletes = [(3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (4, 4), (4, 5), (4, 6), (4, 7), (5, 5), (5, 6), (5, 7)]
        self.action_space = spaces.Discrete(2)
        self.observation_space = np.zeros(comb(self.max_nodes, 2, exact=True))

    def step(self, action: Tuple[int, int]):
        if self.is_done:
            raise RuntimeError("Episode is done")

        reward = self._take_action(action)
        return self.observation_space, reward, self.is_done, dict(red_graph=self.red_graph, blue_graph=self.blue_graph)

    def _take_action(self, action: int):
        assert(0 < action_idx < 3)
        self._color_edge(self, action)
        reward = _check_if_done()
        return reward

    def _check_if_done(self):
        reward = 0
        red_clique_number = clique.graph_clique_number(self.red_graph)
        blue_clique_number = clique.graph_clique_number(self.blue_graph)
        
        for tup in self.KCompletes:
            red_clique_size = tup[0]
            blue_clique_size = tup[1]

            if red_clique_number >= red_clique_size:
                reward -= 1
            if blue_clique_number >= blue_clique_size:
                reward -= 1
                
        self.edge_index += 1            
        self.is_done = self.edge_index > len(self.observation_space)
        return reward
    
    def _color_edge(self, color):
        g = None
        if color == 1:
            g = self.red_graph
            color_char = 'r'
        else:
            g = self.blue_graph
            color_char = 'b'
        self.observation_space[self.edge_index] = color
        vindices = np.tril_indices(self.max_nodes, -1)
        n1 = vindices[0][self.edge_index]
        n2 = vindices[1][self.edge_index]
        g.add_edge(n1, n2, color=color_char)
        networkx.set_edge_attributes(g, {(n1, n2): {"color": color_char}})

    
    def reset(self, seed=42):
        self._init()
        np.random.seed(seed)
        self.observation_space = np.zeros(combin(self.curr_size, 2))
        self.red_graph = networkx.empty_graph(self.curr_size)
        self.blue_graph = networkx.empty_graph(self.curr_size)
        info = dict(red_graph=self.red_graph, blue_graph=self.blue_graph)
        return self.observation_space, info

    def render(self, mode="human"):
        g = networkx.compose(self.red_graph, self.blue_graph)
        edges = g.edges()
        colors = [g[u][v]['color'] for u, v in edges]
        networkx.draw_circular(g, edge_color=colors)
