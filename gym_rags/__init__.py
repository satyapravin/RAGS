# Core Library
import logging

# Third party
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(id="RAGS-v0", entry_point="gym_rags.envs:RAGSEnv",
         kwargs=dict(start_size=8, max_size=43, red_clique=5, blue_clique=5))
