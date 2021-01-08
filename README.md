# Deep Q-Networks

Teach a reinforcement learning agent to play Lunar Lander with Deep-Q Learning.

## Setting up the environment

To create the environment

    conda env create -f environment.yml

To activate the environment

    conda activate dqn

## Running the code

To train the agent using default parameters

    python train.py config.yaml results
  
To run inference 

    python predict.py results scores.npy
