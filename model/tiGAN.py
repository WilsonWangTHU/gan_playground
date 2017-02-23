# -----------------------------------------------------------------------------
#   @brief:
#       In this place, we build the actual text-2-image network
#   @author:
#       Tingwu Wang, hmmm.....
#   @possible compatible problems:
#       1. sigmoid_cross_entropy_with_logits: labels / targets
# -----------------------------------------------------------------------------

import GAN
import init_path
import os
import tensorflow as tf
import numpy as np
import skimage.io as sio
from util import logger
from util import compat_tf


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

    def build_models(self, build_sampler=True):
        logger.info('Building the text to image GAN model')
        # 1. real image and right text
        with tf.variable_scope(""):
            self.d_network_rr = GAN.img_discriminator(
                self.config, stage=self.stage)
            self.d_network_rr.build_models(self.real_img, self.real_sen_rep)
        score = self.d_network_rr.get_score()
        self.loss_r = tf.reduce_mean(
            compat_tf.sigmoid_cross_entropy_with_logits(
                logits=score, labels=tf.ones_like(score)))
        logger.info('loss from real image and right text generated')

        # 2. real image and wrong text
        with tf.variable_scope("", reuse=True):
            self.d_network_rw = GAN.img_discriminator(
                self.config, stage=self.stage)
            self.d_network_rw.build_models(self.real_img, self.wrong_sen_rep)
        score = self.d_network_rw.get_score()
        self.loss_w = tf.reduce_mean(
            compat_tf.sigmoid_cross_entropy_with_logits(
                logits=score, labels=tf.zeros_like(score)))
        logger.info('loss from real image and wrong text generated')

        # 3. fake image and right text
        with tf.variable_scope(''):
            self.g_network = GAN.img_generator(self.config, stage=self.stage)
            self.g_network.build_image_generator(
                self.noise_input, self.real_sen_rep)
            self.fake_img = self.g_network.get_fake_image()

        with tf.variable_scope("", reuse=True):
            self.d_network_wr = GAN.img_discriminator(
                self.config, stage=self.stage)
            self.d_network_wr.build_models(self.fake_img, self.wrong_sen_rep)
        fr_score = self.d_network_wr.get_score()
        self.loss_f = tf.reduce_mean(
            compat_tf.sigmoid_cross_entropy_with_logits(
                logits=fr_score, labels=tf.zeros_like(fr_score)))
        logger.info('loss from fake image and right text generated')

        # the loss of generator and the discriminator
        self.loss_d = self.loss_r + 0.5 * self.loss_f + 0.5 * self.loss_w

        self.loss_g = tf.reduce_mean(
            compat_tf.sigmoid_cross_entropy_with_logits(
                logits=fr_score, labels=tf.ones_like(fr_score)))

        # build the sampler
        if build_sampler:
            with tf.variable_scope('', reuse=True):
                self.sample_network = GAN.img_generator(
                    self.config, stage='test')
                self.sample_network.build_image_generator(
                    self.noise_input, self.real_sen_rep)
                self.sample_img = self.sample_network.get_fake_image()
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
            # init_op = tf.initialize_all_variables()
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
        else:
            self.restore(sess, restore_path)

        self.init_summary(sess)
        return

    def init_summary(self, sess):
        self.loss_d_sum = tf.summary.scalar('discriminator_loss', self.loss_d)
        self.loss_g_sum = tf.summary.scalar('generator_loss', self.loss_g)

        self.loss_real_sum = tf.summary.scalar('real_pair_loss', self.loss_r)
        self.loss_w_sum = tf.summary.scalar(
            'fake_text_real_img_loss', self.loss_w)
        self.loss_f_sum = tf.summary.scalar(
            'real_text_fake_img_loss', self.loss_f)

        self.g_sum = tf.summary.merge(
            [self.loss_g_sum, self.loss_f_sum, self.loss_w_sum])

        self.d_sum = tf.summary.merge(
            [self.loss_d_sum, self.loss_real_sum])

        path = os.path.join(init_path.get_base_dir(), 'summary')
        self.train_writer = tf.summary.FileWriter(path, sess.graph)

        logger.info('summary write initialized, writing to {}'.format(path))

        return

    def train_net(self, sess, data_reader):
        while self.step < self.config.TRAIN.max_step_size:
            feed_dict = self.get_input_dict(data_reader)

            # train the discriminator
            _, loss_d, dis_summary = sess.run(
                [self.d_optimizer, self.loss_d, self.d_sum], feed_dict=feed_dict)

            # train the generator
            _, loss_g, gen_summary = sess.run(
                [self.g_optimizer, self.loss_g, self.g_sum], feed_dict=feed_dict)

            logger.info('step: {}, discriminator: loss {}, generator loss: {}'.
                        format(self.step, loss_d, loss_g))

            # write the summary
            self.train_writer.add_summary(gen_summary, self.step)
            self.train_writer.add_summary(dis_summary, self.step)
            self.step = self.step + 1

            # do test / sampling result once in a while
            if np.mod(self.step, self.config.TRAIN.snapshot_step) == 0:
                # save the check point
                self.save(sess)

                # do the sampling
                self.do_sample(sess, data_reader)
        return

    def do_sample(self, sess, data_reader, save_img=True):
        '''
            @brief:
                sample some result to visualize
        '''
        feed_dict, text = self.get_input_dict(data_reader, sampling=True)
        fake_img = sess.run([self.sample_img], feed_dict=feed_dict)
        fake_img = fake_img[0]

        if save_img:
            self.save_generated_imgs(fake_img, text,
                                     data_reader.get_dataset_name())
        else:
            logger.warning('Generated images are not saved.')
        return

    def save_generated_imgs(self, fake_img, text, dataset_name):
        save_path = os.path.join(
            init_path.get_base_dir(), 'data', 'data_dir', dataset_name,
            'sample', str(self.step))

        if not os.path.exists(save_path):  # make a dir for the new samples
            os.mkdir(save_path)
            logger.info('Making new directory {}'.format(save_path))
        print fake_img.shape

        for i_img in range(len(text)):
            sio.imsave(os.path.join(save_path, text[i_img] + '.jpg'),
                       fake_img[i_img])
        logger.info('Generated images are saved to {}'.format(save_path))
        return

    def save(self, sess):
        base_path = init_path.get_base_dir()
        path = os.path.join(base_path,
                            'checkpoint', 'tigan_' + str(self.step) + '.ckpt')
        self.saver.save(sess, path)

        logger.info('checkpoint saved to {}'.format(path))
        return

    def restore(self, sess, restore_path):
        self.saver.restore(sess, restore_path)
        # we explicit keep a count of steps
        self.step = str(restore_path.split('_')[1].split('.')[0])
        logger.info('checkpoint restored from {}'.format(restore_path))
        logger.info('continues from step {}'.format(self.step))
        return

    def get_input_dict(self, data_reader, sampling=False):
        '''
            @brief: return the feed dict
        '''
        feed_dict = {}

        feed_dict[self.noise_input] = np.random.uniform(
            -1, 1, [self.batch_size, self.config.z_dimension])

        if not sampling:
            feed_dict[self.real_img], feed_dict[self.real_sen_rep], \
                feed_dict[self.wrong_sen_rep] = \
                data_reader.next_batch(self.batch_size)
            return feed_dict
        else:
            feed_dict[self.real_sen_rep], origin_text = \
                data_reader.get_sample_data(
                    self.config.TEST.sample_size)
            return feed_dict, origin_text
