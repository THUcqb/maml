""" Code for the MAML algorithm and network definitions. """
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.platform import flags
from utils import normalize

FLAGS = flags.FLAGS


class MAML(object):
    """maml interface class. Graph and op definition."""

    @staticmethod
    def create(datasource, dim_input, dim_output, test_num_updates):
        """Use factory method to create maml instances."""
        mamls = {
            'sinusoid': MAMLSinusoid,
        }
        return mamls[datasource](dim_input, dim_output, test_num_updates)

    def __init__(self, dim_input, dim_output, test_num_updates):
        """ must call construct_model() after initializing MAML! """
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.update_lr = FLAGS.update_lr
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())
        self.test_num_updates = test_num_updates

    def construct_model(self, prefix='metatrain_'):

        with tf.variable_scope('model') as training_scope:
            weights = self.construct_weights()

            lossesa, outputas = [], []
            # outputbs[i] and lossesb[i] is the output and loss after i+1 gradient updates
            num_updates = max(self.test_num_updates, FLAGS.num_updates)
            lossesb, outputbs = [[]] * num_updates, [[]] * num_updates

            def task_metalearn(inp, reuse=True):
                """Perform gradient descent for one task in the meta-batch."""
                inputa, inputb, labela, labelb = inp
                task_outputbs, task_lossesb = [], []

                task_outputa = self.forward(inputa, weights, reuse=reuse)
                task_lossa = self.loss_func(task_outputa, labela)

                grads = tf.gradients(task_lossa, list(weights.values()))
                if FLAGS.stop_grad:
                    grads = [tf.stop_gradient(grad) for grad in grads]
                gradients = dict(zip(weights.keys(), grads))
                fast_weights = dict(zip(weights.keys(), [
                                    weights[key] - self.update_lr * gradients[key] for key in weights.keys()]))
                output = self.forward(inputb, fast_weights, reuse=True)
                task_outputbs.append(output)
                task_lossesb.append(self.loss_func(output, labelb))

                for j in range(num_updates - 1):
                    loss = self.loss_func(self.forward(
                        inputa, fast_weights, reuse=True), labela)
                    grads = tf.gradients(loss, list(fast_weights.values()))
                    if FLAGS.stop_grad:
                        grads = [tf.stop_gradient(grad) for grad in grads]
                    gradients = dict(zip(fast_weights.keys(), grads))
                    fast_weights = dict(zip(fast_weights.keys(), [
                                        fast_weights[key] - self.update_lr * gradients[key] for key in fast_weights.keys()]))
                    output = self.forward(inputb, fast_weights, reuse=True)
                    task_outputbs.append(output)
                    task_lossesb.append(self.loss_func(output, labelb))

                task_output = [task_outputa, task_outputbs,
                               task_lossa, task_lossesb]

                return task_output

            out_dtype = [tf.float32, [tf.float32] * num_updates,
                         tf.float32, [tf.float32] * num_updates]
            result = tf.map_fn(task_metalearn,
                               elems=(self.inputa, self.inputb,
                                      self.labela, self.labelb),
                               dtype=out_dtype,
                               parallel_iterations=FLAGS.meta_batch_size
                               )

            outputas, outputbs, lossesa, lossesb = result

        self.outputas, self.outputbs = outputas, outputbs
        self.total_loss1 = total_loss1 = tf.reduce_sum(
            lossesa) / tf.to_float(FLAGS.meta_batch_size)
        self.total_losses2 = total_losses2 = [tf.reduce_sum(
            lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]

        # Evaluation
        if 'train' in prefix:
            if FLAGS.metatrain_iterations > 0:
                optimizer = tf.train.AdamOptimizer(self.meta_lr)
                self.metatrain_op = optimizer.minimize(
                    self.total_losses2[FLAGS.num_updates - 1])

        tf.summary.scalar(prefix + 'Pre-update loss', total_loss1)

        for j in range(num_updates):
            tf.summary.scalar(
                prefix + 'Post-update loss, step ' + str(j + 1), total_losses2[j])


class MAMLSinusoid(MAML):
    """Use maml to learn sinusoid."""

    def __init__(self, dim_input, dim_output, test_num_updates):
        super(MAMLSinusoid, self).__init__(
            dim_input, dim_output, test_num_updates)
        self.dim_hidden = [40, 40]
        self.loss_func = tf.losses.mean_squared_error

        # a: training data for inner update gradient
        # b: test data for meta gradient
        self.inputa = tf.placeholder(tf.float32)
        self.inputb = tf.placeholder(tf.float32)
        self.labela = tf.placeholder(tf.float32)
        self.labelb = tf.placeholder(tf.float32)

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

    def forward(self, inp, weights, reuse=False):
        hidden = normalize(tf.matmul(inp, weights['w1']) + weights['b1'],
                           activation=tf.nn.relu, reuse=reuse, scope='0')
        for i in range(1, len(self.dim_hidden)):
            hidden = normalize(tf.matmul(hidden, weights['w' + str(i + 1)]) + weights['b' + str(i + 1)],
                               activation=tf.nn.relu, reuse=reuse, scope=str(i + 1))
        return tf.matmul(hidden, weights['w' + str(len(self.dim_hidden) + 1)]) + weights['b' + str(len(self.dim_hidden) + 1)]

