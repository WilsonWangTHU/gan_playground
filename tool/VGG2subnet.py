# ----------------------------------------------------------------------
#    In this file, the original VGG network written by
#    https://github.com/machrisaa/tensorflow-vgg is converted to the
#    standard submodel in the following manner: store in a numpy file
#    and the name remains the same
#   Written by Tingwu Wang, 2016/SEP/22
# ----------------------------------------------------------------------

import __init_path
import subnet.vgg.vgg19 as vgg19
import tensorflow as tf
import util.vgg_utils as vgg_util
import util.model_saver as ms

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model_type', 'vgg_19',
                    'current support for vgg 19, 16 might come after')
flags.DEFINE_string('old_model_path', '/ais/gobi4/tingwuwang/joint_embedding/vgg19.npy',
                    'The input of old model')
flags.DEFINE_string('new_model_path', '/ais/gobi4/tingwuwang/joint_embedding/vgg19_submodel.npy',
                    'The output of new model')
flags.DEFINE_bool('test_new_model', True, 'The output of new model')
flags.DEFINE_string('test_image_dir', '../subnet/vgg/test_data/tiger.jpeg', 'The test image')
flags.DEFINE_string('synset_dir', '../subnet/vgg/synset.txt', 'The image synset')


def convert_and_test():
    sess = tf.Session()

    if FLAGS.test_new_model:
        # input data initialization
        img1 = vgg_util.load_image(FLAGS.test_image_dir)
        batch1 = img1.reshape((1, 224, 224, 3))

        train_mode = tf.placeholder(tf.bool)
        images = tf.placeholder(tf.float32, [1, 224, 224, 3])

        # load the new model for testing
        vgg = vgg19.Vgg19(load_old_model=False)
        vgg.build(images, train_mode)
        sess.run(tf.initialize_all_variables())
        ms.model_loader(sess, FLAGS.new_model_path)
        print('Model loaded from {}'.format(FLAGS.new_model_path))

        # test the new model function
        prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
        vgg_util.print_prob(prob[0], FLAGS.synset_dir)

    else:
        # load the old model and save it as new model
        vgg = vgg19.Vgg19(FLAGS.old_model_path)
        images = tf.placeholder(tf.float32, [1, 224, 224, 3])
        img1 = vgg_util.load_image(FLAGS.test_image_dir)
        batch1 = img1.reshape((1, 224, 224, 3))

        train_mode = tf.placeholder(tf.bool)
        vgg.build(images, train_mode)
        sess.run(tf.initialize_all_variables())

        # save the model
        ms.model_saver(sess, FLAGS.new_model_path)

        # test the new model function
        prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
        vgg_util.print_prob(prob[0], FLAGS.synset_dir)
        print('Model saved to {}'.format(FLAGS.new_model_path))


def main(_):
    convert_and_test()


if __name__ == '__main__':
    tf.app.run()
