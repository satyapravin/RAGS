This repository contains a PIP package which is an OpenAI environment for simulating an enironment in which graph edges are colored.


## Installation

Install the [OpenAI gym](https://gym.openai.com/docs/).

Then install this package via

```
pip install -e .
```

## Usage

```
import gym
import gym_rags

env = gym.make('RAGS-v0', start_size=8, max_size=43, red_clique=5, blue_clique=5) # Graph with 8 nodes and no 5 red or green cliques.
```

See https://github.com/matthiasplappert/keras-rl/tree/master/examples for some examples.


## The Environment

Imagine you are coloring edges of a complete Graph with Red or Green colors such that there are no K cliques.
