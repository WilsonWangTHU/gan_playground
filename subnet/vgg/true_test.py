"""
Simple tester for the vgg19_trainable
"""

import tensorflow as tf

import vgg19
import sys
sys.path.append('../..')
import util.model_saver as ms


with tf.device('/cpu:0'):
    sess = tf.Session()

    images = tf.placeholder(tf.float32, [1, 224, 224, 3])
    true_out = tf.placeholder(tf.float32, [1, 1000])
    train_mode = tf.placeholder(tf.bool)

    vgg = vgg19.Vgg19('/ais/gobi4/tingwuwang/vgg19.npy')
    vgg.build(images, train_mode)

    # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    #print vgg.get_var_count()
    var_list = tf.get_collection(tf.GraphKeys.VARIABLES)
    for name in var_list:
        print name.name

    # ms.model_saver(sess, '/ais/gobi4/tingwuwang/test.npy')
    ms.model_loader(sess, '/ais/gobi4/tingwuwang/test.npy')
    sess.run(tf.initialize_all_variables())
