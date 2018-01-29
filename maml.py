""" Code for the MAML algorithm and network definitions. """
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.platform import flags
from utils import normalize

FLAGS = flags.FLAGS


class Maml(object):
    """maml interface class. Graph and op definition.

    Model-agnostic meta learning, so this class should BE model agnostic.
    SubClass this to use maml with specific models like MLP or CNN
    """

    @staticmethod
    def create(datasource, dim_input, dim_output, test_num_updates):
        """Use factory method to create maml instances."""
        mamls = {
            'sinusoid': MamlMLP(dim_input, dim_output, test_num_updates, [40, 40]),
            'cartpole': MamlMLP(dim_input, dim_output, test_num_updates, [20, 20]),
        }
        return mamls[datasource]

    def __init__(self, dim_input, dim_output, test_num_updates):
        """ must call construct_model() after initializing MAML! """
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.update_lr = FLAGS.update_lr
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())
        self.test_num_updates = test_num_updates
        # a: training data for inner update gradient
        # b: test data for meta gradient
        self.inputa = tf.placeholder(tf.float32)
        self.inputb = tf.placeholder(tf.float32)
        self.labela = tf.placeholder(tf.float32)
        self.labelb = tf.placeholder(tf.float32)

    def construct_model(self, prefix='metatrain_'):

        with tf.variable_scope('model') as training_scope:
            weights = self.construct_weights()

            num_updates = max(self.test_num_updates, FLAGS.num_updates)
            # outputxs[i] and lossesx[i] is after i gradient updates
            # x = a,b; for training and testing, respectively
            lossesa, outputas = [[]] * num_updates, [[]] * num_updates
            lossesb, outputbs = [[]] * num_updates, [[]] * num_updates

            def task_metalearn(inp):
                """Perform gradient descent for all tasks in the meta-batch."""
                inputa, inputb, labela, labelb = inp
                task_outputas, task_lossesa = [], []
                task_outputbs, task_lossesb = [], []

                # Clone meta-learned (initialized) weights
                fast_weights = dict(zip(weights.keys(),
                                        [tf.identity(weights[key]) for key in weights.keys()]))
                for j in range(num_updates):
                    # Evaluate on testing samples
                    output = self.forward(inputb, fast_weights)
                    loss = self.loss_func(output, labelb)
                    task_outputbs.append(output)
                    task_lossesb.append(loss)

                    # Forward(evaluate) on training samples
                    output = self.forward(inputa, fast_weights)
                    loss = self.loss_func(output, labela)
                    task_outputas.append(output)
                    task_lossesa.append(loss)

                    # Backward
                    grads = tf.gradients(loss, list(fast_weights.values()))
                    if FLAGS.stop_grad:
                        grads = [tf.stop_gradient(grad) for grad in grads]
                    gradients = dict(zip(fast_weights.keys(), grads))
                    fast_weights = dict(zip(fast_weights.keys(), [
                                        fast_weights[key] - self.update_lr * gradients[key] for key in fast_weights.keys()]))

                # Evaluate on testing samples
                output = self.forward(inputb, fast_weights)
                loss = self.loss_func(output, labelb)
                task_outputbs.append(output)
                task_lossesb.append(loss)

                # Evaluate on training samples
                output = self.forward(inputa, fast_weights)
                loss = self.loss_func(output, labela)
                task_outputas.append(output)
                task_lossesa.append(loss)

                task_output = [task_outputas, task_outputbs,
                               task_lossesa, task_lossesb]

                return task_output

            out_dtype = [[tf.float32] * (num_updates + 1)] * 4
            result = tf.map_fn(task_metalearn,
                               elems=(self.inputa, self.inputb,
                                      self.labela, self.labelb),
                               dtype=out_dtype,
                               parallel_iterations=FLAGS.meta_batch_size
                               )

            outputas, outputbs, lossesa, lossesb = result

        self.outputas, self.outputbs = outputas, outputbs
        self.total_losses1 = [tf.reduce_sum(
            lossesa[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates+1)]
        self.total_losses2 = [tf.reduce_sum(
            lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates+1)]

        # Evaluation
        if 'train' in prefix:
            self.pretrain_op = tf.train.AdamOptimizer(
                self.meta_lr).minimize(self.total_losses1[0])
            if FLAGS.metatrain_iterations > 0:
                optimizer = tf.train.AdamOptimizer(self.meta_lr)
                self.metatrain_op = optimizer.minimize(
                    self.total_losses2[FLAGS.num_updates])

        tf.summary.scalar(prefix + 'Pre-update training loss',
                          self.total_losses1[0])
        tf.summary.scalar(prefix + 'Pre-update testing loss',
                          self.total_losses2[0])
        for j in range(1, num_updates+1):
            tf.summary.scalar(
                prefix + 'Post-update training loss, step ' + str(j), self.total_losses1[j])
            tf.summary.scalar(
                prefix + 'Post-update testing loss, step ' + str(j), self.total_losses2[j])
        self.summ_op = tf.summary.merge_all()


class MamlMLP(Maml):
    """MAML which use fully connected layes as inner network."""

    def __init__(self, dim_input, dim_output, test_num_updates, dim_hidden):
        super(MamlMLP, self).__init__(
            dim_input, dim_output, test_num_updates)
        self.dim_hidden = dim_hidden
        self.loss_func = tf.losses.mean_squared_error

    # virtual functions being implemented
    def construct_weights(self):
        weights = {}
        weights['w1'] = tf.Variable(tf.truncated_normal(
            [self.dim_input, self.dim_hidden[0]], stddev=0.01))
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden[0]]))
        for i in range(1, len(self.dim_hidden)):
            weights['w' + str(i + 1)] = tf.Variable(tf.truncated_normal(
                [self.dim_hidden[i - 1], self.dim_hidden[i]], stddev=0.01))
            weights['b' + str(i + 1)
                    ] = tf.Variable(tf.zeros([self.dim_hidden[i]]))
        weights['w' + str(len(self.dim_hidden) + 1)] = tf.Variable(
            tf.truncated_normal([self.dim_hidden[-1], self.dim_output], stddev=0.01))
        weights['b' + str(len(self.dim_hidden) + 1)
                ] = tf.Variable(tf.zeros([self.dim_output]))
        return weights

    def forward(self, inp, weights):
        hidden = normalize(tf.matmul(inp, weights['w1']) + weights['b1'],
                           activation=tf.nn.relu, scope='0')
        for i in range(1, len(self.dim_hidden)):
            hidden = normalize(tf.matmul(hidden, weights['w' + str(i + 1)]) + weights['b' + str(i + 1)],
                               activation=tf.nn.relu, scope=str(i + 1))
        return tf.matmul(hidden, weights['w' + str(len(self.dim_hidden) + 1)]) + weights['b' + str(len(self.dim_hidden) + 1)]
