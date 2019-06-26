# -*- coding: utf-8 -*-

from collections import deque
import numpy as np
import os

__all__ = ['train']


def train(agent, num_episodes=1000, num_consecutive_episodes=100,
          max_steps=1000, min_score=-np.inf, max_average_score=np.inf,
          output_dir=None, save_weights_interval=5, render=False,
          verbose=True):

    rolling_scores = deque(maxlen=num_consecutive_episodes)
    scores = []
    average_scores = []

    if output_dir is None:
        write = False
    else:
        output_dir = os.path.abspath(output_dir)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        write = True

    if write:
        cwd = os.getcwd()
        os.chdir(output_dir)
        os.mkdir('weights')
        weights_file = os.path.join('weights', 'episode%05i_score%05i.h5')

    try:

        for episode in range(1, num_episodes + 1):

            state = agent.env.reset()
            score = 0.0

            for steps in range(1, max_steps + 1):

                if render:
                    agent.env.render()

                if np.random.rand() < agent.epsilon:
                    action = agent.env.action_space.sample()
                else:
                    action = agent.act(state)

                next_state, reward, done, _ = agent.env.step(action)

                agent.observe(state, reward, action, next_state, done)
                agent.replay()  # train

                state = next_state
                score += reward

                if done or score < min_score:
                    break

            rolling_scores.append(score)
            average_score = np.mean(rolling_scores)

            scores.append(score)
            average_scores.append(average_score)

            if verbose:
                print('Episode: %i, Steps: %i, Epsilon: %.2f, '
                      'Current score: %.2f, Average score: %.2f' %
                      (episode, steps, agent.epsilon, score, average_score))

            if write and episode % save_weights_interval == 0:
                agent.save_weights(weights_file % (episode, average_score))

            if average_score >= max_average_score:
                break

    except KeyboardInterrupt:
        # allow early termination
        pass

    if write:
        agent.save_weights(weights_file % (episode, average_score))
        agent.save_model('model.h5')
        np.save('scores.npy', scores)
        np.save('average_scores.npy', average_scores)
        os.chdir(cwd)
