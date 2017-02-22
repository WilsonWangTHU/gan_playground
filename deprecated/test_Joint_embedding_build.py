# -----------------------------------------------------------------------------
#   In this file, we make sure the network is built without error
#
# -----------------------------------------------------------------------------

import __init_path
import tensorflow as tf
from model import joint_encoder
from config import cfg


with tf.Graph().as_default(), tf.Session() as session:
    with tf.variable_scope("", reuse=None):

        image_input = tf.placeholder(tf.float32, [cfg.TRAIN.batch_size, 224, 224, 3])
        text_input = tf.placeholder(tf.int32, [cfg.TRAIN.batch_size, cfg.seq_max_len])
        input_seq_len = tf.placeholder(tf.int32, [cfg.TRAIN.batch_size])
        vgg_train_mode = tf.placeholder(tf.bool)

        testmodel = joint_encoder(config=cfg)
        testmodel.build_models(image_input, text_input, input_seq_len, vgg_train_mode)
        tf.initialize_all_variables().run()

    list = tf.get_collection(tf.GraphKeys.VARIABLES)
    for item in list:
        print "[DEBUG INFO] ****************************************"
        print item.name
        print item.get_shape()
