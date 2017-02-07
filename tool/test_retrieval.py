# -----------------------------------------------------------------------------
#   Test the result of the retrieval task
#   Written by Tingwu Wang Oct/04/2016.
# -----------------------------------------------------------------------------


import __init_path
import tensorflow as tf
from model import joint_encoder
import util.model_saver as ms
from config import cfg
from util.util import joint_embedding_data_reader
import os
import numpy as np

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_bool('is_load_model', True,
                  'whether we load trained joint_embedding model')
flags.DEFINE_string('load_model_path',
                    '/ais/gobi4/tingwuwang/joint_embedding/saved_models/36000_bird.npy',
                    'whether we load trained joint_embedding model')
flags.DEFINE_string('stage', 'test', 'whether test or validate')

# precalculated feature space
flags.DEFINE_bool('use_precalculated_features', False, 'load calculated features')
flags.DEFINE_bool('save_features', True, 'save calculated features')
flags.DEFINE_string(
    'output_feature_path_dir',
    '/ais/gobi4/tingwuwang/joint_embedding/saved_features/bird/',
    'save calculated features')
flags.DEFINE_string(
    'input_feature_path',
    '/ais/gobi4/tingwuwang/joint_embedding/saved_features/bird/10000_bird_feature.npy',
    'save calculated features')

# the dataset information
flags.DEFINE_string('data_set', 'bird', 'the data set we choose')
flags.DEFINE_string(
    'data_dir',
    '/ais/gobi4/tingwuwang/joint_embedding/data_dir/',
    'the data set we choose')
flags.DEFINE_string('use_data_parsing_file', True,
                    'the training set, testing set and validation set dir')

flags.DEFINE_integer('test_text_batch_size', 160, 'the number of text batch size')
flags.DEFINE_integer('test_image_batch_size', 32, 'the number of image batch size')


def test_joint_embedding():
    tf.device('/gpu:0')
    if not FLAGS.use_precalculated_features:
        # ----------------------------------------------------------------------
        # initializing the session and network graph
        # some special parameters
        assert FLAGS.use_data_parsing_file and FLAGS.is_load_model, \
            '[ERROR]Invalid flags for testing'
        if FLAGS.data_set == 'bird':
            assert FLAGS.test_text_batch_size == \
                5 * FLAGS.test_image_batch_size, \
                '[ERROR]Wrong image text relationship'.format(FLAGS.data_set)
        sess = tf.Session()

        images_placeholder = \
            tf.placeholder(tf.float32, [None, 224, 224, 3])
        text_placeholder = \
            tf.placeholder(tf.int32, [None, cfg.seq_max_len])
        text_seqlen_placeholder = \
            tf.placeholder(tf.int32, [None])

        train_mode_placeholder = tf.placeholder(tf.bool)
        print('[INIT MODEL]Initializing the model')

        # ----------------------------------------------------------------------
        # build the graph
        with tf.variable_scope("", reuse=None):
            image_model = joint_encoder(config=cfg, stage='image_test')
            image_model.build_models(
                images_placeholder, text_placeholder,
                text_seqlen_placeholder, train_mode_placeholder)
        with tf.variable_scope("", reuse=None):
            sentence_model = joint_encoder(config=cfg, stage='sentence_test')
            sentence_model.build_models(
                images_placeholder, text_placeholder,
                text_seqlen_placeholder, train_mode_placeholder)

        sess.run(tf.initialize_all_variables())
        print('[INIT MODEL]Network graph has been built')

        # ----------------------------------------------------------------------
        # load the models
        assert os.path.exists(FLAGS.load_model_path), \
            '[ERROR]Invalid checkpoint path at {}!'.\
            format(FLAGS.load_model_path)

        # load the step information from the name of the checkpoint
        ms.model_loader(sess, FLAGS.load_model_path, ignore_prefix='learning',
                        only_load_exist_variable=True)
        print('[INIT MODEL]Model check point loaded from {}'
              .format(FLAGS.load_model_path))

        # ----------------------------------------------------------------------
        # the data interface
        print('[INIT DATA READER]Initializing the data reader')
        data_reader = joint_embedding_data_reader(
            dataset_name=FLAGS.data_set,
            dataset_dir=FLAGS.data_dir,
            data_parsing_file=FLAGS.use_data_parsing_file,
            stage=FLAGS.stage,
            debug=False)

        # ----------------------------------------------------------------------
        # get the representation
        image_rep = []
        sentence_rep = []

        num_total_batch = data_reader.num_data_in_use()

        num_epoch = (num_total_batch / FLAGS.test_image_batch_size) + 1
        if num_total_batch % FLAGS.test_image_batch_size == 0:
            num_epoch = num_epoch - 1
        for i_epoch in range(num_epoch):
            image_list, text_list, text_seq_len_list = \
                data_reader.next_batch(FLAGS.test_image_batch_size)
            # calculate them!
            batch_image_rep = image_model.get_encoded_image(sess, image_list)
            batch_text_rep = \
                sentence_model.get_encoded_sentence(
                    sess, text_list, text_seq_len_list)
            image_rep.extend(batch_image_rep[0])
            sentence_rep.extend(batch_text_rep[0])
            print('[LOAD] Features calculated for batch {}/{}'.
                  format(i_epoch, num_epoch))
        # save the representation if needed
        if FLAGS.save_features:
            output_list = {}
            output_list['image'] = image_rep
            output_list['text'] = sentence_rep
            name = os.path.basename(
                FLAGS.load_model_path).split('.')[0] + '_feature'
            output_path = os.path.join(FLAGS.output_feature_path_dir, name)
            np.save(output_path, output_list)
            print('[LOAD] Features saved to path {}'.
                  format(output_path))
    else:
        # load the representations
        feature_list = np.load(FLAGS.input_feature_path,
                               encoding='latin1').item()
        image_rep = feature_list['image']
        sentence_rep = feature_list['text']
        print('[LOAD] Features loaded from {}'.format(FLAGS.input_feature_path))
    '''
    print image_rep[0]
    print image_rep[1]
    print image_rep[2]

    print '........................'
    print '........................'
    print '........................'
    print '........................'

    print sentence_rep[0]
    print sentence_rep[1]
    print sentence_rep[2]
    print sentence_rep[10]
    print sentence_rep[11]
    print len(sentence_rep)
    print len(sentence_rep[0])
    print np.sum(sentence_rep[1]**2)
    a = 0
    for i in xrange(155):
        a = a + sentence_rep[i][0] ** 2
    print a
    a = raw_input()
    '''
    # now, retrieve the image text relationship accordingly
    error_matrix = get_error_matrix(sentence_rep, image_rep)
    # print error_matrix[0, :]
    # print error_matrix[1, :]
    # print error_matrix[:, 0]
    # print error_matrix[:, 1]
    ranks, r1, r5, r10, median, mean = text2image(error_matrix)
    print("The retrieval result is {}, {}, {}, median: {}, mean: {}".format(r1, r5, r10, median, mean))


def get_error_matrix(sentence_rep, image_rep):
    num_data = len(image_rep)
    error = np.zeros([5 * num_data, num_data])
    # TODO: SPEED UP
    for i_image in range(num_data):
        for i_text in range(num_data * 5):
            error[i_text, i_image] = \
                2 - 2 * np.sum(image_rep[i_image] * sentence_rep[i_text])
    return error


"""
def image2text(error):
    # number of image is ranks, denoted as N
    ranks = np.zeros(error.shape[1])  # number of images
    for i in range(error.shape[1]):
        d_i = error[:, i]  # d_i is of size 5 * N
        inds = np.argsort(d_i)

        ranks[i] = np.where(inds / 5 == i)[0][0]

    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    median = np.floor(np.median(ranks)) + 1
    mean = ranks.mean() + 1
    return ranks, r1, r5, r10, median, mean
"""


def text2image(error):
    # number of image is ranks, denoted as 5 * N
    ranks = np.zeros(error.shape[0])  # number of text
    for i in range(error.shape[0]):
        d_i = error[i, :]  # d_i is of size 5 * N
        inds = np.argsort(d_i)
        ranks[i] = np.where(inds == i / 5)[0][0]

    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    median = np.floor(np.median(ranks)) + 1
    mean = ranks.mean() + 1
    return ranks, r1, r5, r10, median, mean


def main(_):
    test_joint_embedding()


if __name__ == '__main__':
    tf.app.run()
