#!/usr/bin/env python

# Core Library
import unittest

# Third party
import gym

# First party
import gym_rags  # noqa


class Environments(unittest.TestCase):
    def test_env(self):
        env = gym.make("RAGS-v0")
        env.seed(0)
        env.reset()
        env.step(0)
