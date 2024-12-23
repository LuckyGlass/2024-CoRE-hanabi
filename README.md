# Intro.

This repo is based on [Google Hanabi Learning Env.](https://github.com/google-deepmind/hanabi-learning-environment).

# Getting started
> From Google Hanabi Learning Env.

Install the learning environment:
```
sudo apt-get install g++            # if you don't already have a CXX compiler
sudo apt-get install python-pip     # if you don't already have pip
pip install git+https://github.com/google-deepmind/hanabi-learning-environment.git
```
Run the examples:
```
pip install numpy                   # game_example.py uses numpy
python examples/rl_env_example.py   # Runs RL episodes
python examples/game_example.py     # Plays a game using the lower level interface
```
