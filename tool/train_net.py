# -----------------------------------------------------------------------------
#   In this model, we try to train the joint embedding of the network
#   It is similar to the joint-embedding loss or unifying loss....
#   Note:
#       1. the model could use the model saver implemented in tf, or
#           save the model in the ms.model_save (so that it could be used
#           as sub model)
#       2. ...
#   Written by Tingwu Wang.
#
#   TODO:
#       1. (DONE) parsing the training set, validating set and test set
#       2. (DONE) model save function and load from saved model
#       4. test the model
#       3. learning rate decay mechanism (update when no inprovement seen)
#       5. LOAD ADAM PARAMETERS???
# -----------------------------------------------------------------------------


import __init_path
import tensorflow as tf
from model import joint_encoder
import util.model_saver as ms
from config import cfg
from util.util import joint_embedding_data_reader
import os

'''
    @brief:
        some basic model parameters as external flags. It's worth noting that
        most of the static parameters are set in the config.py
'''
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_bool('is_load_model', True,
                  'whether we load trained joint_embedding model')
flags.DEFINE_string(
    'load_model_path',
    '/ais/gobi4/tingwuwang/joint_embedding/saved_models/38000_bird.npy',
    'whether we load trained joint_embedding model')
flags.DEFINE_bool('is_save_model', True,
                  'whether we save trained joint_embedding model')
flags.DEFINE_string('save_model_path',
                    '/ais/gobi4/tingwuwang/joint_embedding/saved_models/',
                    'whether we save trained joint_embedding model')

# the vgg parameters
flags.DEFINE_bool('use_vgg_pretrain_model', True, 'The input of old model')
flags.DEFINE_bool('joint_train_vgg', True, 'The input of old model')
flags.DEFINE_string('vgg_pretrain_model_path',
                    '/ais/gobi4/tingwuwang/joint_embedding/vgg19_submodel.npy',
                    'The input of old model')

# the dataset information
flags.DEFINE_string(
    'data_dir', '/ais/gobi4/tingwuwang/joint_embedding/data_dir/',
    'the data_dir')
flags.DEFINE_string('data_set', 'bird', 'the data set we choose')
flags.DEFINE_string('use_data_parsing_file', True,
                    'the training set, testing set and validation set dir')

# debug information
flags.DEFINE_string('debug', False,
                    'Overfitting a small dataset')


def train_joint_embedding():
    # -------------------------------------------------------------------------
    # initializing the session and network graph
    tf.device('/gpu:3')
    sess = tf.Session()

    images_placeholder = \
        tf.placeholder(tf.float32, [cfg.TRAIN.batch_size, 224, 224, 3])
    train_mode_placeholder = tf.placeholder(tf.bool)
    text_placeholder = \
        tf.placeholder(tf.int32, [cfg.TRAIN.batch_size, cfg.seq_max_len])
    text_seqlen_placeholder = \
        tf.placeholder(tf.int32, [cfg.TRAIN.batch_size])
    print('[INIT MODEL] Initializing the model')

    # -------------------------------------------------------------------------
    # build the graph
    train_model = joint_encoder(config=cfg)
    train_model.build_models(images_placeholder, text_placeholder,
                             text_seqlen_placeholder, train_mode_placeholder)
    sess.run(tf.initialize_all_variables())
    starting_step = 0
    print('[INIT MODEL] Network graph has been built')

    # -------------------------------------------------------------------------
    # load the models
    if FLAGS.is_load_model:
        # we load the checkpoint files from the old model
        assert os.path.exists(FLAGS.load_model_path), \
            '[ERROR] Invalid checkpoint path at {}!'.\
            format(FLAGS.load_model_path)

        # load the step information from the name of the checkpoint
        starting_step = \
            int(os.path.basename(FLAGS.load_model_path).split('_')[0])
        ms.model_loader(sess, FLAGS.load_model_path, ignore_prefix='learning')
        print('[INIT MODEL] Model check point loaded from {}'
              .format(FLAGS.load_model_path))
    else:
        # we train the je model from scratch
        if FLAGS.use_vgg_pretrain_model:
            # load the vgg pretrained model
            train_model.partial_load_vgg_model(
                sess, FLAGS.vgg_pretrain_model_path)

            # TODO: make sure the training data follows the steps

    # -------------------------------------------------------------------------
    # the data interface
    print('[INIT DATA READER] Initializing the data reader')
    data_reader = joint_embedding_data_reader(
        dataset_name=FLAGS.data_set,
        dataset_dir=FLAGS.data_dir,
        data_parsing_file=FLAGS.use_data_parsing_file,
        debug=FLAGS.debug)

    # -------------------------------------------------------------------------
    # the training part
    for i_step in range(starting_step, cfg.TRAIN.max_step_size):
        lr_decay = 10 ** max(i_step - cfg.TRAIN.max_step_size, 0.0)

        image_data, text_data, text_len_data = \
            data_reader.next_batch(cfg.TRAIN.batch_size)

        train_model.assign_lr(sess, cfg.TRAIN.learning_rate * lr_decay)
        train_model.train_step(sess, image_data, text_data,
                               text_len_data, FLAGS.joint_train_vgg)

        # save the snapchat
        if FLAGS.is_save_model and \
                i_step % cfg.TRAIN.snapshot_step == 0 and i_step > 0:
            # save the model
            save_path = os.path.join(FLAGS.save_model_path,
                                     str(i_step) + '_' + FLAGS.data_set)
            ms.model_saver(sess, save_path)
            print('[SAVE CHECKPOINT] Checkpoint file has been saved to {}'
                  .format(save_path))


def main(_):
    train_joint_embedding()


if __name__ == '__main__':
    tf.app.run()
