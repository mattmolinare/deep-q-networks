#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import deque
import gym
import numpy as np
import os

# local imports
from . import models
from . import utils

__all__ = [
    'DQNAgent',
    'DuelingDQNAgent'
]


class DQNAgent:

    def __init__(self, env, fc_layers, learning_rate, replay_memory_size,
                 min_eps, max_eps, lmbda, batch_size, gamma,
                 target_update_interval, use_double_dqn):

        self.env = env
        self.fc_layers = fc_layers
        self.learning_rate = learning_rate
        self.replay_memory_size = replay_memory_size
        self.min_eps = min_eps
        self.max_eps = max_eps
        self.lmbda = lmbda
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_interval = target_update_interval
        self.use_double_dqn = use_double_dqn

        self.model = self.build_model()
        self.target_model = self.build_model()

        self._memory = np.ndarray(self.replay_memory_size,
                                  dtype=utils.get_memory_dtype(self.env))

        self.t = 0
        self.epsilon = self.max_eps

    @property
    def memory(self):
        return self._memory[:self.t]

    def build_model(self):
        return models.build_dqn_model(self.env, self.fc_layers,
                                      self.learning_rate)

    def load_weights(self, weights_file):
        self.model.load_weights(weights_file)
        self.update_target_model()

    def save_weights(self, weights_file):
        self.model.save_weights(weights_file)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def sample(self):
        return np.random.choice(self.memory, size=min(self.t, self.batch_size),
                                replace=False)

    def observe(self, state, reward, action, next_state, done):

        transition = self._memory[self.t % self.replay_memory_size]

        transition['state'] = state
        transition['reward'] = reward
        transition['action'] = action
        transition['next_state'] = next_state
        transition['done'] = done

    def act(self, state):
        return self.model.predict_on_batch(state[np.newaxis])[0].argmax()

    def replay(self):

        batch = self.sample()

        q_values = self.model.predict_on_batch(batch['state'])
        target_q_values = self.target_model.predict_on_batch(batch['state'])

        if self.use_double_dqn:
            actions = q_values.argmax(axis=1)
            discounted = target_q_values[np.arange(batch.size), actions]
        else:
            discounted = target_q_values.max(axis=1)

        targets = np.where(
            batch['done'],
            batch['reward'],
            batch['reward'] + self.gamma * discounted
        )

        q_values[np.arange(batch.size), batch['action']] = targets

        self.model.train_on_batch(batch['state'], q_values)

        if self.t % self.target_update_interval == 0:
            self.update_target_model()

    def fit(self, num_episodes, num_consecutive_episodes, max_steps, min_score,
            max_average_score, output_dir=None, save_weights_interval=5,
            render=False, verbose=True):

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
            weights_file = os.path.join('weights', 'episode%i.h5')

        try:

            for episode in range(1, num_episodes + 1):

                state = self.env.reset()
                score = 0.0

                for _ in range(max_steps):

                    if render:
                        self.env.render()

                    if np.random.rand() < self.epsilon:
                        action = self.env.action_space.sample()
                    else:
                        action = self.act(state)

                    next_state, reward, done, _ = self.env.step(action)

                    self.observe(state, reward, action, next_state, done)
                    self.replay()

                    state = next_state
                    score += reward

                    self.t += 1
                    self.epsilon = utils.compute_epsilon(self.t, self.min_eps,
                                                         self.max_eps,
                                                         self.lmbda)

                    if done or score < min_score:
                        break

                rolling_scores.append(score)
                average_score = np.mean(rolling_scores)

                scores.append(score)
                average_scores.append(average_score)

                if verbose:
                    print('Episode: %i, Epsilon: %.2f, Current score: %.2f, '
                          'Average score: %.2f' %
                          (episode, self.epsilon, score, average_score))

                if write and episode % save_weights_interval == 0:
                    self.save_weights(weights_file % episode)

                if average_score >= max_average_score:
                    break

        except KeyboardInterrupt:
            pass

        if write:
            self.save_weights(weights_file % episode)
            np.save('scores.npy', scores)
            np.save('average_scores.npy', average_scores)
            os.chdir(cwd)

    def test(self, num_iterations, max_steps, video_dir=None, render=False,
             seed=None):

        env = self.env
        env.seed(seed)

        if video_dir is not None:
            env = gym.wrapper.Monitor(env, video_dir, force=True)

        scores = np.empty(num_iterations)

        for i in range(num_iterations):

            state = env.reset()
            score = 0.0

            for _ in range(max_steps):

                if render:
                    env.render()

                action = self.act(state)
                next_state, reward, done, _ = env.step(action)

                state = next_state
                score += reward

                if done:
                    break

            scores[i] = score

        return scores


class DuelingDQNAgent(DQNAgent):

    def build_model(self):
        self._model = models.build_dueling_dqn_model(self.env, self.fc_layers,
                                                     self.learning_rate)
