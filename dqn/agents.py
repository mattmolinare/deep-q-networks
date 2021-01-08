#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""DQN agents
"""

from collections import deque
import gym
import numpy as np
import os

# local imports
from . import models
from . import utils

__all__ = [
    'DQNAgent',
    'TargetDQNAgent',
    'DDQNAgent',
    'DuelingDQNAgent',
    'DuelingDDQNAgent'
]


class DQNAgent:
    """Vanilla Deep Q Network
    """

    def __init__(self, env, fc_layers=[512], learning_rate=0.0003,
                 replay_memory_size=100000, min_eps=0.1, max_eps=1.0,
                 lmbda=0.001, batch_size=64, gamma=0.99, **kwargs):

        self.env = env
        self.fc_layers = fc_layers
        self.learning_rate = learning_rate
        self.replay_memory_size = replay_memory_size
        self.min_eps = min_eps
        self.max_eps = max_eps
        self.lmbda = lmbda
        self.batch_size = batch_size
        self.gamma = gamma

        dtype = utils.get_transition_dtype(self.env)
        self._memory = np.recarray(self.replay_memory_size, dtype=dtype)

        self._t = 0  # internal step counter
        self.epsilon = self.max_eps

        self.model = self.build_model()

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, value):
        self._t = value
        self.epsilon = utils.compute_epsilon(self.t, self.min_eps,
                                             self.max_eps, self.lmbda)

    @property
    def memory(self):
        return self._memory[:self.t]

    def build_model(self):
        return models.build_dqn_model(self.env, self.fc_layers,
                                      self.learning_rate)

    def load_weights(self, weights_file):
        self.model.load_weights(weights_file)

    def save_weights(self, weights_file):
        self.model.save_weights(weights_file)

    def save_model(self, model_file):
        self.model.save(model_file)

    def sample(self):

        if self.t < self.batch_size:
            return self.memory
        else:
            idx = np.random.randint(0, min(self.t, self.replay_memory_size),
                                    size=self.batch_size)
            return self.memory[idx]

    def observe(self, state, reward, action, next_state, done):

        transition = self._memory[self.t % self.replay_memory_size]

        transition.state = state
        transition.reward = reward
        transition.action = action
        transition.next_state = next_state
        transition.done = done

        self.t += 1  # increment counter, update epsilon

    def act(self, state):
        return self.model.predict_on_batch(state[np.newaxis])[0].argmax()

    def compute_discounted(self, batch):

        q_tp1 = self.model.predict_on_batch(batch.next_state)
        discounted = q_tp1.max(axis=1)

        return discounted

    def replay(self):

        batch = self.sample()

        discounted = self.compute_discounted(batch)

        targets = np.where(
            batch.done,
            batch.reward,
            batch.reward + self.gamma * discounted
        )

        q_t = self.model.predict_on_batch(batch.state)
        q_t[np.arange(batch.size), batch.action] = targets

        self.model.train_on_batch(batch.state, q_t)

    def fit(self, num_episodes=1000, num_consecutive_episodes=100,
            max_steps=1000, min_score=None, max_average_score=None,
            output_dir=None, save_weights_interval=5, render=False,
            verbose=True):

        if min_score is None:
            min_score = -np.inf

        if max_average_score is None:
            max_average_score = np.inf

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
            os.mkdir(os.path.join(output_dir, 'weights'))
            weights_file = os.path.join(output_dir, 'weights',
                                        'episode%05i_score%05i.h5')

        try:

            for episode in range(1, num_episodes + 1):

                state = self.env.reset()
                score = 0.0

                for steps in range(1, max_steps + 1):

                    if render:
                        self.env.render()

                    # epsilon-greedy
                    if np.random.rand() < self.epsilon:
                        action = self.env.action_space.sample()
                    else:
                        action = self.act(state)

                    next_state, reward, done, _ = self.env.step(action)

                    self.observe(state, reward, action, next_state, done)
                    self.replay()  # train using experience replay

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
                          (episode, steps, self.epsilon, score, average_score))

                if write and episode % save_weights_interval == 0:
                    self.save_weights(weights_file % (episode, average_score))

                if average_score >= max_average_score:
                    break

        except KeyboardInterrupt:
            # allow early termination
            pass

        if write:
            self.save_weights(weights_file % (episode, average_score))
            self.save_model(os.path.join(output_dir, 'model.h5'))
            np.save(os.path.join(output_dir, 'scores.npy'), scores)
            np.save(os.path.join(output_dir, 'average_scores.npy'),
                    average_scores)

    def predict(self, num_iterations=100, max_steps=1000, video_dir=None,
                render=False, verbose=True):

        env = self.env

        if video_dir is not None:
            env = gym.wrappers.Monitor(env, video_dir, force=True)

        scores = np.empty(num_iterations)

        for i in range(num_iterations):

            state = env.reset()
            score = 0.0

            for _ in range(max_steps):

                if render:
                    env.render()

                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)

                state = next_state
                score += reward

                if done:
                    break

            scores[i] = score

            if verbose:
                print('Iteration: %i, Score: %.2f' % (i + 1, score))

        if verbose:
            print('Mean score: %.2f, Min score: %.2f, Max score: %.2f' %
                  (scores.mean(), scores.min(), scores.max()))

        return scores


class TargetDQNAgent(DQNAgent):
    """DQN with target model updated regularly
    """

    def __init__(self, env, target_update_interval=10000, **kwargs):

        super().__init__(env, **kwargs)

        self.target_update_interval = target_update_interval
        self.target_model = self.build_model()

    def load_weights(self, weights_file):
        super().load_weights(weights_file)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def compute_discounted(self, batch):

        target_q_tp1 = self.target_model.predict_on_batch(batch.next_state)
        discounted = target_q_tp1.max(axis=1)

        return discounted

    def replay(self):

        super().replay()

        if self.t % self.target_update_interval == 0:
            self.update_target_model()


class DDQNAgent(TargetDQNAgent):
    """DQN with two sets of weights, one used to determine the greedy policy
    and the other to determine its value
    """

    def compute_discounted(self, batch):

        q_tp1 = self.model.predict_on_batch(batch.next_state)
        target_q_tp1 = self.target_model.predict_on_batch(batch.next_state)

        actions = q_tp1.argmax(axis=1)
        discounted = target_q_tp1[np.arange(batch.size), actions]

        return discounted


class DuelingDQNAgent(DQNAgent):
    """DQN with two streams to separately estimate the scalar state-value and
    advantages for each action
    """

    def build_model(self):
        return models.build_dueling_dqn_model(self.env, self.fc_layers,
                                              self.learning_rate)


class DuelingDDQNAgent(DDQNAgent, DuelingDQNAgent):
    """Dueling DDQN
    """
