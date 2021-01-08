# Deep Q-Networks

Teach a reinforcement learning agent to play Lunar Lander with Deep-Q Learning.

## Setting up the environment

To create the environment

    conda env create -f environment.yml

To activate the environment

    conda activate dqn

## Running the code

Set training parameters (refer to `config.yaml`).

To train the agent

    python train.py path/to/config.yaml path/to/parent_dir
  
To generate results

    python predict.py path/to/parent_dir path/to/output_dir
