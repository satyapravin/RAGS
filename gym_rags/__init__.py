# Core Library
import logging

# Third party
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(id="RAGS-v0", entry_point="gym_rags.envs:RAGSEnv", kwargs={"dim": (5, 3, 3)})
