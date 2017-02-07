"""
Simple tester for the vgg19_trainable
"""

import tensorflow as tf

import vgg19
import utils
import sys
sys.path.append('../..')
import util.model_saver as ms

img1 = utils.load_image("./test_data/tiger.jpeg")
img1_true_result = [1 if i == 292 else 0 for i in xrange(1000)]  # 1-hot result for tiger

batch1 = img1.reshape((1, 224, 224, 3))

#with tf.device('/cpu:0'):
if 1 > 0:
    #sess = tf.Session()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    images = tf.placeholder(tf.float32, [1, 224, 224, 3])
    true_out = tf.placeholder(tf.float32, [1, 1000])
    train_mode = tf.placeholder(tf.bool)

    # vgg = vgg19.Vgg19('/ais/gobi4/tingwuwang/vgg19.npy')
    vgg = vgg19.Vgg19()
    vgg.build(images, train_mode)
   

    #s print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    # print vgg.get_var_count()

    sess.run(tf.initialize_all_variables())
    # ms.model_saver(sess, '/ais/gobi4/tingwuwang/transformed.npy')
    ms.model_loader(sess, '/ais/gobi4/tingwuwang/transformed.npy')

    # test classification
    prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
    utils.print_prob(prob[0], './synset.txt')

    # simple 1-step training
    cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
    train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
    print(sess.run(train, feed_dict={images: batch1, true_out: [img1_true_result], train_mode: True}))
    weights = sess.run('fc8/fc8_weights:0')
    print weights

    variable_list = tf.get_collection(tf.GraphKeys.VARIABLES)
    for name in variable_list:
        print name.name

    # test classification again, should have a higher probability about tiger
    #prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
    #utils.print_prob(prob[0], './synset.txt')

    # test save
    #vgg.save_npy(sess, './test-save.npy')

    # Runs the op.
