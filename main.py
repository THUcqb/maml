"""
Usage Instructions:
    10-shot sinusoid:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --metatrain_iterations=16000 --update_batch_size=10

    10-shot cartpole:
        python main.py --datasource=cartpole --logdir=logs/cartpole/ --metatrain_iterations=16000 --update_batch_size=10

    To run evaluation, use the '--train=False' flag to use the test set.

    For omniglot and miniimagenet training, acquire the dataset online, put it in the correspoding data directory, and see the python script instructions in that directory to preprocess the data.
"""
import csv
import numpy as np
import pickle
import random
import tensorflow as tf
from dataset import DataGenerator
from maml import Maml
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

# Dataset/method options
flags.DEFINE_string('datasource', 'sinusoid',
                    'sinusoid or omniglot or miniimagenet')
# Training options
flags.DEFINE_integer('pretrain_iterations', 0,
                     'number of pre-training iterations.')
# 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('metatrain_iterations', 15000,
                     'number of metatraining iterations.')
flags.DEFINE_integer('meta_batch_size', 25,
                     'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 1e-3, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 5,
                     'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', 1e-3,
                   'step size alpha for inner gradient update.')
flags.DEFINE_integer(
    'num_updates', 1, 'number of inner gradient updates during training.')

# Model options
flags.DEFINE_string('norm', 'None', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer(
    'num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
flags.DEFINE_bool(
    'conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_bool('max_pool', False,
                  'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', False,
                  'if True, do not use second derivatives in meta-optimization (for speed)')

# Logging, saving, and testing options
flags.DEFINE_bool(
    'log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', '/tmp/data',
                    'directory for summaries and checkpoints.')
flags.DEFINE_bool(
    'resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer(
    'test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_integer('train_update_batch_size', -1,
                     'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1,
                   'value of inner gradient step step during training. (use if you want to test with a different value)')  # 0.1 for omniglot

SUMMARY_INTERVAL = 100
SAVE_INTERVAL = SUMMARY_INTERVAL * 20
PRINT_INTERVAL = SUMMARY_INTERVAL
TEST_PRINT_INTERVAL = PRINT_INTERVAL * 50


def train(model, saver, sess, exp_string, data_generator, resume_itr=0):
    if FLAGS.log:
        train_writer = tf.summary.FileWriter(
            FLAGS.logdir + '/' + exp_string, sess.graph)
    print('Done initializing, starting training.')
    prelosses, postlosses = [], []

    for itr in range(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations):
        # Prepare task
        # training (for inner gradients) data
        # and testing (for meta gradient) data
        feed_dict = {}
        batch_x, batch_y = data_generator.batch()[:2]

        inputa = batch_x[:, :FLAGS.update_batch_size, :]
        labela = batch_y[:, :FLAGS.update_batch_size, :]
        # b used for testing
        inputb = batch_x[:, FLAGS.update_batch_size:, :]
        labelb = batch_y[:, FLAGS.update_batch_size:, :]
        feed_dict = {model.inputa: inputa, model.inputb: inputb,
                     model.labela: labela, model.labelb: labelb}

        fetches = {}

        # Pretrain the parameters
        # To minimize the preloss before any inner update
        # Equal to trying to obtain an all-purpose model on the tasks
        if itr < FLAGS.pretrain_iterations:
            fetches['train'] = model.pretrain_op
        else:
            fetches['train'] = model.metatrain_op

        # Print and summary
        if itr % SUMMARY_INTERVAL == 0:
            fetches['summary'] = model.summ_op
            fetches['pre-losses'] = model.total_losses2[0]
            fetches['post-losses'] = model.total_losses2[FLAGS.num_updates]

        results = sess.run(fetches, feed_dict)

        if itr % SUMMARY_INTERVAL == 0:
            prelosses.append(results['pre-losses'])
            if FLAGS.log:
                train_writer.add_summary(results['summary'], itr)
            postlosses.append(results['post-losses'])

        if (itr + 1) % PRINT_INTERVAL == 0:
            if itr < FLAGS.pretrain_iterations:
                print_str = 'Pretrain Iteration ' + str(itr)
            else:
                print_str = 'Iteration ' + str(itr - FLAGS.pretrain_iterations)
            print_str += ': ' + str(np.mean(prelosses)) + \
                ', ' + str(np.mean(postlosses))
            print(print_str)
            prelosses, postlosses = [], []

        if (itr + 1) % SAVE_INTERVAL == 0:
            saver.save(sess, FLAGS.logdir + '/' +
                       exp_string + '/model' + str(itr))

        # sinusoid is infinite data, so no need to test on meta-validation set.
        # if (itr != 0) and itr % TEST_PRINT_INTERVAL == 0 and FLAGS.datasource != 'sinusoid':
        #     if 'batch' not in dir(data_generator):
        #         feed_dict = {}
        #         # if model.classification:
        #         #     fetches = [model.metaval_total_accuracy1,
        #         #                      model.metaval_total_accuracies2[FLAGS.num_updates-1], model.summ_op]
        #         # else:
        #         fetches = [
        #             model.metaval_total_loss1, model.metaval_total_losses2[FLAGS.num_updates-1], model.summ_op]
        #     else:
        #         batch_x, batch_y, amp, phase = data_generator.batch()
        #         inputa = batch_x[:, :FLAGS.update_batch_size, :]
        #         inputb = batch_x[:, FLAGS.update_batch_size:, :]
        #         labela = batch_y[:, :FLAGS.update_batch_size, :]
        #         labelb = batch_y[:, FLAGS.update_batch_size:, :]
        #         feed_dict = {model.inputa: inputa, model.inputb: inputb,
        #                      model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}
        #         # if model.classification:
        #         #     fetches = [model.total_accuracy1,
        #         #                      model.total_accuracies2[FLAGS.num_updates-1]]
        #         # else:
        #         fetches = [model.total_loss1, model.total_losses2[FLAGS.num_updates-1]]

        #     result = sess.run(fetches, feed_dict)
        #     print('Validation results: ' +
        #           str(result[0]) + ', ' + str(result[1]))

    saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))


# calculated for omniglot
NUM_TEST_POINTS = 600


def test(model, saver, sess, exp_string, data_generator, test_num_updates=None):

    np.random.seed(1)
    random.seed(1)

    metaval_accuracies = []

    for _ in range(NUM_TEST_POINTS):
        if 'batch' not in dir(data_generator):
            feed_dict = {}
            feed_dict = {model.meta_lr: 0.0}
        else:
            batch_x, batch_y = data_generator.batch()[:2]

            inputa = batch_x[:, :FLAGS.update_batch_size, :]
            inputb = batch_x[:, FLAGS.update_batch_size:, :]
            labela = batch_y[:, :FLAGS.update_batch_size, :]
            labelb = batch_y[:, FLAGS.update_batch_size:, :]

            feed_dict = {model.inputa: inputa, model.inputb: inputb,
                         model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}

        result = sess.run([model.total_loss1] +
                          model.total_losses2, feed_dict)
        metaval_accuracies.append(result)

    metaval_accuracies = np.array(metaval_accuracies)
    means = np.mean(metaval_accuracies, 0)
    stds = np.std(metaval_accuracies, 0)
    ci95 = 1.96 * stds / np.sqrt(NUM_TEST_POINTS)

    print('Mean validation accuracy/loss, stddev, and confidence intervals')
    print((means, stds, ci95))

    out_filename = FLAGS.logdir + '/' + exp_string + '/' + 'test_ubs' + \
        str(FLAGS.update_batch_size) + '_stepsize' + \
        str(FLAGS.update_lr) + '.csv'
    out_pkl = FLAGS.logdir + '/' + exp_string + '/' + 'test_ubs' + \
        str(FLAGS.update_batch_size) + '_stepsize' + \
        str(FLAGS.update_lr) + '.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump({'mses': metaval_accuracies}, f)
    with open(out_filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['update' + str(i) for i in range(len(means))])
        writer.writerow(means)
        writer.writerow(stds)
        writer.writerow(ci95)


def main():
    test_num_updates = 5

    if FLAGS.train == False:
        orig_meta_batch_size = FLAGS.meta_batch_size
        # always use meta batch size of 1 when testing.
        FLAGS.meta_batch_size = 1

    data_generator = DataGenerator.create(
        FLAGS.datasource, FLAGS.update_batch_size * 2, FLAGS.meta_batch_size)

    dim_input = data_generator.dim_input
    dim_output = data_generator.dim_output

    model = Maml.create(FLAGS.datasource, dim_input, dim_output,
                        test_num_updates=test_num_updates)
    if FLAGS.train:
        model.construct_model(prefix='metatrain_')
    else:
        model.construct_model(prefix='metaval_')

    saver = loader = tf.train.Saver(tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)

    sess = tf.InteractiveSession()

    if FLAGS.train == False:
        # change to original meta batch size when loading model.
        FLAGS.meta_batch_size = orig_meta_batch_size

    if FLAGS.train_update_batch_size == -1:
        FLAGS.train_update_batch_size = FLAGS.update_batch_size
    if FLAGS.train_update_lr == -1:
        FLAGS.train_update_lr = FLAGS.update_lr

    exp_string = 'mbs_' + str(FLAGS.meta_batch_size) + '.ubs_' + str(
        FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(FLAGS.train_update_lr)

    if FLAGS.num_filters != 64:
        exp_string += 'hidden' + str(FLAGS.num_filters)
    if FLAGS.max_pool:
        exp_string += 'maxpool'
    if FLAGS.stop_grad:
        exp_string += 'stopgrad'
    if FLAGS.norm == 'batch_norm':
        exp_string += 'batchnorm'
    elif FLAGS.norm == 'layer_norm':
        exp_string += 'layernorm'
    elif FLAGS.norm == 'None':
        exp_string += 'nonorm'
    else:
        print('Norm setting not recognized.')

    resume_itr = 0
    model_file = None

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    if FLAGS.resume or not FLAGS.train:
        model_file = tf.train.latest_checkpoint(
            FLAGS.logdir + '/' + exp_string)
        if FLAGS.test_iter > 0:
            model_file = model_file[:model_file.index(
                'model')] + 'model' + str(FLAGS.test_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1 + 5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

    if FLAGS.train:
        train(model, saver, sess, exp_string, data_generator, resume_itr)
    else:
        test(model, saver, sess, exp_string, data_generator, test_num_updates)


if __name__ == "__main__":
    main()
