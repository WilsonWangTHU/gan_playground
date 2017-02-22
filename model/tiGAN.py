# -----------------------------------------------------------------------------
#   @brief:
#       In this place, we build the actual text-2-image network
#   @author:
#       Tingwu Wang, hmmm.....
# -----------------------------------------------------------------------------

import GAN
import __init_path
import os
import tensorflow as tf
import numpy as np
from util import logger


class TI_GAN(object):
    '''
        @brief
            in this network, we have a generator and a discriminator
            note that the loss come from the {real img, right txt}, 
            {real img, wrong txt} and {fake image, right txt}
    '''
    def __init__(self, config, stage='train'):
        '''
            @brief:
                for the input, we have the noise input, the img input, the
                text representation input
        '''

        assert stage in ['train', 'test'], \
            logger.error('Invalid training stage')
        logger.warning('test mode is not supported currently')
        self.config = config
        self.batch_size = config.TRAIN.batch_size
        self.stage = stage
        self.train = (self.stage == 'train')

        # define the placeholders
        self.noise_input = tf.placeholder(
            tf.float32, [self.batch_size, self.config.z_dimension])
        self.real_img = tf.placeholder(tf.float32, [self.batch_size, 64, 64, 3])
        self.real_sen_rep = tf.placeholder(tf.float32, [self.batch_size, 1024])
        self.wrong_sen_rep = tf.placeholder(tf.float32, [self.batch_size, 1024])
        self.step = 0

        return

    def build_models(self):
        # 1. real image and right text
        with tf.variable_scope(""):
            self.d_network_rr = GAN.img_discriminator(self.config, stage=self.stage)
            self.d_network_rr.build_models(self.real_img, self.real_sen_rep)
        score = self.d_network_rr.get_score()
        loss_r = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=score, labels=tf.ones_like(score))
        logger.info('loss from real image and right text generated')

        # 2. real image and wrong text
        with tf.variable_scope("", reuse=True):
            self.d_network_rw = GAN.img_discriminator(self.config, stage=self.stage)
            self.d_network_rw.build_models(self.real_img, self.wrong_sen_rep)
        score = self.d_network_rw.get_score()
        loss_w = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=score, labels=tf.zeros_like(score))
        logger.info('loss from real image and wrong text generated')

        # 3. fake image and right text
        self.g_network = GAN.img_generator(self.config, stage=self.stage)
        self.g_network.build_image_generator(self.noise_input, self.real_sen_rep)
        self.fake_img = self.g_network.get_fake_image()

        with tf.variable_scope("", reuse=True):
            self.d_network_wr = GAN.img_discriminator(self.config, stage=self.stage)
            self.d_network_wr.build_models(self.fake_img, self.wrong_sen_rep)
        fr_score = self.d_network_wr.get_score()
        loss_f = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fr_score, labels=tf.zeros_like(fr_score))
        logger.info('loss from fake image and right text generated')


        # the loss of generator and the discriminator
        self.loss_d = tf.reduce_mean(loss_r) + \
            0.5 * tf.reduce_mean(loss_f) + 0.5 * tf.reduce_mean(loss_w)
        
        self.loss_g = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fr_score, labels=tf.ones_like(fr_score))
        return


    def init_training(self, sess, restore_path):
        '''
            @brief:
                define all the training paras and how to train it
        '''
        # get the training optimizer
        t_vars = tf.trainable_variables()

        self.g_vars = [var for var in t_vars if 'img_generator' in var.name]
        self.d_vars = [var for var in t_vars if 'img_discriminator' in var.name]

        self.d_optimizer = tf.train.AdamOptimizer(
            self.config.TRAIN.learning_rate, beta1=self.config.TRAIN.beta1,
            beta2=self.config.TRAIN.beta2).minimize(self.loss_d,
                var_list=self.d_vars)

        self.g_optimizer = tf.train.AdamOptimizer(
            self.config.TRAIN.learning_rate, beta1=self.config.TRAIN.beta1,
            beta2=self.config.TRAIN.beta2).minimize(self.loss_g,
                var_list=self.g_vars)

        # init the saver
        self.saver = tf.train.Saver()

        # get the variable initialized
        if restore_path is None:
            init_op = tf.initialize_all_variables()
            sess.run(init_op)
        else:
            self.restore(sess, restore_path)
        return

    def train(self, sess, data_reader):
        while self.step < self.config.TRAIN.max_step_size:
            feed_dict = self.get_input_dict(data_reader)

            # train the discriminator
            _, loss_d = sess.run(
                [self.d_optimizer, self.loss_d], feed_dict=feed_dict)

            # train the generator
            _, loss_g = sess.run(
                [self.g_optimizer, self.loss_g], feed_dict=feed_dict)

            logger.info('step: {}, discriminator: loss {}, generator loss: {}'.\
                format(self.step, loss_d, loss_g))
            self.step = self.step + 1

            # do test / sampling result once in a while
            if np.mod(self.step, self.config.TRAIN.snapshot_step) == 0:
                # save the check point
                self.save(sess)
        return

    def test(self):
        '''
            @brief:
                sample some result to visualize
        '''
        return

    def save(self, sess):
        path = os.path.join(
            __init_path.get_base_dir(), 'tigan_' + str(self.step) + '.ckpt')
        self.saver.save(sess, path, 'tigan_' + str(self.step) + '.ckpt')

        logger.info('checkpoint saved to {}'.format(path))
        return

    def restore(self, sess, restore_path):
        self.saver.restore(sess, restore_path)
        # we explicit keep a count of steps
        self.step = str(restore_path.split('_')[1].split('.')[0])
        logger.info('checkpoint restored from {}'.format(restore_path))
        logger.info('continues from step {}'.format(self.step))
        return

    def get_input_dict(self, data_reader):
        '''
            @brief: return the feed dict
        '''
        feed_dict = {}

        feed_dict[self.noise_input] = np.random.uniform(
            -1, 1, [self.batch_size, self.config.z_dimension])

        feed_dict[self.real_img], feed_dict[self.real_sen_rep], \
            feed_dict[self.wrong_sen_rep] = \
            data_reader.next_batch(self.batch_size)

        return feed_dict
