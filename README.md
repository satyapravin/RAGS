This repository contains a PIP package which is an OpenAI environment for computing R(5, 7).


## Installation

Install the [OpenAI gym](https://www.gymlibrary.ml/).

Then install this package via

```
pip install -e .
```

## Usage

```
import gym
import gym_rags

# Graph with 8 nodes and no K5 red or K5 green cliques.
env = gym.make('RAGS-v0') 
```

## The Environment

Imagine you are coloring edges of a complete Graph with Red or Green colors such that there are no K monochromatic cliques.
