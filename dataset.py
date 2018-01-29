""" Code for loading data. """
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import gym
gym.logger.setLevel('WARNING')

FLAGS = flags.FLAGS


class DataGenerator(object):
    """Interface class, generating meta batches of update batches of data.

    Call data_generator.batch() to generate
    """

    @staticmethod
    def create(datasource, num_samples_per_task, num_tasks):
        """Use factory method to create data generator instances."""
        data_generators = {
            'sinusoid': DataGeneratorSinusoid,
            'cartpole': DataGeneratorCartPole,
        }
        return data_generators[datasource](num_samples_per_task, num_tasks)

    def __init__(self, num_samples_per_task, num_tasks):
        """
        :param num_samples_per_task: num samples per task, \
                including training(update_batch_size) and testing
        :param num_tasks(meta_batch_size): meta_batch_size, \
                number of tasks per meta batch
        """
        self.num_samples_per_task = num_samples_per_task
        self.num_tasks = num_tasks

    def batch(self):
        """
        Inputs and outputs should have shape\
                (num_tasks, 1, dim_input) and (num_tasks, 1, dim_output)
        """
        pass


class DataGeneratorSinusoid(DataGenerator):
    """Generate a batch of sinusoid data."""

    def __init__(self, num_samples_per_task, num_tasks):
        super(DataGeneratorSinusoid, self).__init__(
            num_samples_per_task, num_tasks)
        self.amp_range = [0.1, 5.0]
        self.phase_range = [0, np.pi]
        self.input_range = [-5.0, 5.0]
        self.dim_input = 1
        self.dim_output = 1

    def batch(self):
        amp = np.random.uniform(
            self.amp_range[0], self.amp_range[1], [self.num_tasks])
        phase = np.random.uniform(
            self.phase_range[0], self.phase_range[1], [self.num_tasks])
        outputs = np.zeros(
            [self.num_tasks, self.num_samples_per_task, self.dim_output])
        inputs = np.zeros(
            [self.num_tasks, self.num_samples_per_task, self.dim_input])
        for func in range(self.num_tasks):
            inputs[func] = np.random.uniform(
                self.input_range[0], self.input_range[1],
                [self.num_samples_per_task, self.dim_input])
            outputs[func] = amp[func] * np.sin(inputs[func] - phase[func])
        return inputs, outputs, amp, phase


class DataGeneratorCartPole(DataGenerator):
    """Generate a meta batch data of a cartpole.

    Every task has num_samples_per_task samples.
    Input has 4-d state & 1-d action
    Output has 4-d new state
    """

    def __init__(self, num_samples_per_task, num_tasks):
        super(DataGeneratorCartPole, self).__init__(
            num_samples_per_task, num_tasks)
        self.dim_input = 5
        self.dim_output = 4

    def batch(self):
        inputs = np.zeros(
            [self.num_tasks, self.num_samples_per_task, self.dim_input])
        outputs = np.zeros(
            [self.num_tasks, self.num_samples_per_task, self.dim_output])

        # First, sample num_tasks tasks from all cartpole tasks.
        # masscart ~ [0.5, 1.5)
        masscart = np.random.uniform(0.5, 1.5, self.num_tasks)
        # masspole ~ [0.05, 0.15)
        masspole = np.random.uniform(0.05, 0.15, self.num_tasks)
        # pole length ~ [0.25, 0.75)
        length = np.random.uniform(0.25, 0.75, self.num_tasks)

        # Get num_samples_per_task pairs of data for every task.
        for task_iter in range(self.num_tasks):
            # Sample states and actions as training data in one task.
            gym.CARTPOLE_MASSCART = masscart[task_iter]
            gym.CARTPOLE_MASSPOLE = masspole[task_iter]
            gym.CARTPOLE_LENGTH = length[task_iter]

            self.env = gym.make('CartPole-v1')
            # self.env.seed(0)
            observation = self.env.reset()
            done = False
            for sample_iter in range(self.num_samples_per_task):
                action = self.env.action_space.sample()
                inputs[task_iter, sample_iter] = np.concatenate(
                    [observation, [action]])
                if done:
                    self.env.reset()
                observation, reward, done, info = self.env.step(action)
                outputs[task_iter, sample_iter] = observation - \
                    inputs[task_iter, sample_iter, :-1]

            inputs[task_iter] -= np.mean(inputs[task_iter], axis=0)
            inputs[task_iter] = np.nan_to_num(
                inputs[task_iter] / np.std(inputs[task_iter], axis=0))
            outputs[task_iter] -= np.mean(outputs[task_iter], axis=0)
            outputs[task_iter] = np.nan_to_num(
                outputs[task_iter] / np.std(outputs[task_iter], axis=0))
            idxs = np.arange(self.num_samples_per_task)
            np.random.shuffle(idxs)
            inputs[task_iter] = inputs[task_iter][idxs]
            outputs[task_iter] = outputs[task_iter][idxs]

        return inputs, outputs, masscart, masspole, length
