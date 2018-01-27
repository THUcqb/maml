""" Code for loading data. """
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


class DataGenerator(object):
    """Interface class, generating meta batches of update batches of data."""

    @staticmethod
    def create(datasource, num_samples_per_task, num_tasks):
        """Use factory method to create data generator instances."""
        data_generators = {
            'sinusoid': DataGeneratorSinusoid,
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

    def __init__(self, num_samples_per_task, num_tasks):
        super(DataGeneratorSinusoid, self).__init__(
            num_samples_per_task, num_tasks)
        self.amp_range = [0.1, 5.0]
        self.phase_range = [0, np.pi]
        self.input_range = [-5.0, 5.0]
        self.dim_input = 1
        self.dim_output = 1

    def batch(self):
        """Generate a batch of sinusoid data."""
        amp = np.random.uniform(
            self.amp_range[0], self.amp_range[1], [self.num_tasks])
        phase = np.random.uniform(
            self.phase_range[0], self.phase_range[1], [self.num_tasks])
        outputs = np.zeros(
            [self.num_tasks, self.num_samples_per_task, self.dim_output])
        init_inputs = np.zeros(
            [self.num_tasks, self.num_samples_per_task, self.dim_input])
        for func in range(self.num_tasks):
            init_inputs[func] = np.random.uniform(
                self.input_range[0], self.input_range[1],
                [self.num_samples_per_task, self.dim_input])
            outputs[func] = amp[func] * np.sin(init_inputs[func] - phase[func])
        return init_inputs, outputs, amp, phase

