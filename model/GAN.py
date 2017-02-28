# -----------------------------------------------------------------------------
#   @brief:
#       In this file, we will first reimplement the T2I GAN model from the
#       http://arxiv.org/abs/1605.05396.
#       After that, we will see if we could implement wasserstein GAN
#       And there is a lot of fun to do with the GAN at hand
#   @author: Tingwu Wang, Feb., 20th, 2017
#   @possible compatible problem:
# -----------------------------------------------------------------------------


import tensorflow as tf
import util.ops as op
from util import logger


class img_generator(object):
    '''
        @brief:
            We are going to generate image from text, just the way that icml
            2016 did.
    '''

    def __init__(self, config, stage='train'):
        self.config = config  # config is a python variable
        self.step = 0  # the num of step
        self.stage = stage  # train, test, val?
        self.train = (self.stage == 'train')
        logger.info('Image generator in training mode: {}'.format(self.train))
        self.batch_size = self.config.TRAIN.batch_size

    def build_image_generator(self, img_z):
        with tf.variable_scope('img_generator'):
            nf = 64
            # layer 1: a projection after the noise into 4, 4, 512
            self.l1 = op.linear(img_z, nf * 8 * 4 * 4)  # 'l0_b*[4*4*512]'
            self.l1 = tf.reshape(self.l1, [self.batch_size, 4, 4, nf * 8])  # 'l0_b*4*4*4*512')
            self.l1_bn = op.batch_norm(name='l1_bn0')
            self.l1 = tf.nn.relu(self.l1_bn(self.l1, train=self.train))
            print ('name: {}, size: {}'.format(self.l1.name, self.l1.get_shape()))

            # layer 2: first conv1
            self.l2 = op.deconv2d(
                self.l1, [self.batch_size, 8, 8, nf * 4], name='l2')
            self.l2_bn = op.batch_norm(name='l2_bn0')
            self.l2 = tf.nn.relu(self.l2_bn(self.l2, train=self.train))

            # layer 3: conv2
            self.l3 = op.deconv2d(
                self.l2, [self.batch_size, 16, 16, nf * 2], name='l3')
            self.l3_bn = op.batch_norm(name='l3_bn0')
            self.l3 = tf.nn.relu(self.l3_bn(self.l3, train=self.train))

            # layer 4: conv4
            self.l4 = op.deconv2d(
                self.l3, [self.batch_size, 32, 32, nf], name='l4')
            self.l4_bn = op.batch_norm(name='l4_bn0')
            self.l4 = tf.nn.relu(self.l4_bn(self.l4, train=self.train))

            # layer 5: conv5 / final
            self.l5 = op.deconv2d(
                self.l4, [self.batch_size, 64, 64, 3], name='l5')
            # self.l5_bn = op.batch_norm(name='l5_bn0')
            # self.l5 = tf.nn.relu(self.l5_bn(self.l5, train=self.train))

            self.fake_img = tf.nn.tanh(self.l5)
            img_shape = self.fake_img.get_shape()

            # check the size of the image
            assert (img_shape[1] == 64) and \
                (img_shape[2] == 64) and (img_shape[3] == 3), \
                logger.error('Wrong fake image dimension: {}'.format(img_shape))
        return

    def get_fake_image(self):
        return self.fake_img


class img_discriminator(object):
    '''
        @brief:
            The img discriminator. we currently use the same structure described
            in the paper.
        @components:
            def __init__
            def build_models
    '''

    def __init__(self, config, stage='train'):
        """
            @brief:
                The initialization of the model. we use a config file to store
                the information of configuration

        """
        self.config = config  # config is a python variable
        self.stage = stage  # train, test, val?
        self.train = (self.stage == 'train')
        self.batch_size = self.config.TRAIN.batch_size  # it is a short-cut

        # in 'sen_test' or 'img_test' mode, only sentence or image encoder is
        # constucted
        assert self.stage in ['train', 'test', 'val'], \
            logger.error('Invalid stage of the network.' +
                         'Discriminator must work both with image and text')

    def build_models(self, image):
        with tf.variable_scope('img_discriminator'):
            nf = 64
            self.img = image  # size 64, 64, 3

            # layer 1
            self.l1 = op.conv2d(self.img, nf, name='l1')
            self.l1 = op.lrelu(self.l1)
            # self.l1_bn = op.batch_norm(name='l1_bn0')
            # self.l1 = op.lrelu(self.l1_bn(self.l1, train=self.train))

            # layer 2
            self.l2 = op.conv2d(self.l1, nf * 2, name='l2')
            self.l2_bn = op.batch_norm(name='l2_bn0')
            self.l2 = op.lrelu(self.l2_bn(self.l2, train=self.train))

            # layer 3
            self.l3 = op.conv2d(self.l2, nf * 4, name='l3')
            self.l3_bn = op.batch_norm(name='l3_bn0')
            self.l3 = op.lrelu(self.l3_bn(self.l3, train=self.train))

            # layer 4
            self.l4 = op.conv2d(self.l3, nf * 8, name='l4')
            self.l4_bn = op.batch_norm(name='l4_bn0')
            self.l4 = op.lrelu(self.l4_bn(self.l4, train=self.train))

            # layer 6, actually it is different from the original paper..
            self.score = op.linear(
                tf.reshape(self.l4, [self.batch_size, -1]), 1, 'final')
        return

    def get_score(self):
        return self.score

    '''
    def build_sentence_preprocessor(self):
            NOT IMPLEMENTED
            @brief:
                encode the information in the sentences. It's worth noting we
                could do it in an encoder-rnn way, or into a CNN way...
            @TODO:
                the CNN sentence preprocessor
        assert False, logger.error('Not implemented!')
        with tf.variable_scope('d_sentence_preprocess'):
            # it is similar to the encoder defined in the class joint_encoder
            inputs = tf.nn.embedding_lookup(self.word_embedding, self.sentence)
            # now it is [MAX_SEQ_LENGTH, batch_size, embedding_length]
            input_batch_order = tf.transpose(inputs, [1, 0, 2])
            # now it is [MAX_SEQ_LENGTH * batch_size, embedding_length]
            input_batch_order = tf.reshape(
                input_batch_order, [-1, self.config.word_embedding_space_size])

            # now it is LIST OF [BATCH_SIZE, embedding_length]
            encoder_input = tf.split(
                0, self.config.seq_max_len, input_batch_order)

            # the encoder part
            preprocess_gru_cell = tf.nn.rnn_cell.GRUCell(
                self.config.preprocess_dimension)

            _, self.sentence_rep = tf.nn.rnn(preprocess_gru_cell, encoder_input,
                                             dtype=tf.float32,
                                             sequence_length=self.input_seq_len)
        return
    '''


class sentence_generator(object):
    '''
        @brief:
            still not functional. Might be of later use
    '''

    def __init__(self, config, stage='train', teacher_forcing=True,
                 word_embedding=None):
        self.config = config  # config is a python variable
        self.step = 0  # the num of step
        self.stage = stage  # train, test, val?
        self.teacher_forcing = teacher_forcing
        self.word_embedding = word_embedding
        assert False, logger.error('Sentence generator not functional')

    def build_sentence_generator(self):
        '''
            @brief: it is actually very tricky... not sure how we gonna
                generate the text.
        '''
        with tf.variable_scope('sen_generator'):
            # the conditional vector
            self.l0 = tf.concat(1, [self.sen_z, op.lrelu(self.img_rep)])

            # layer 1, transform from the raw state to the initial state
            self.l1, self.h0_w, self.h0_b = op.linear(
                self.l0, self.config.text_gen_hidden_dim, 'l0_lin', with_w=True)
            self.l1 = tf.reshape(self.l1, [self.batch_size,
                                           self.config.text_gen_hidden_dim])
            self.l1_bn = op.batch_norm(name='l1_bn0')
            self.l1 = tf.nn.relu(self.l1_bn(self.l1, train=self.train))

            # layer 2, the rnn part
            cell = tf.nn.rnn_cell.GRUCell(self.config.text_gen_hidden_dim)

            # define the vocabulary matrix here, note that there's a diff
            # between the embedding matrix and the vocabulary matrix
            self.vocabulary_mat = tf.get_variable(
                'voc_mat', initializer=tf.random_normal(
                    [self.config.text_gen_hidden_dim,
                     self.config.word_embedding_space_size]))
            self.vocabulary_mat_trans = tf.transpose(self.vocabulary_mat)
            self.vocabulary_bias = tf.get_variable(
                'voc_bias', initializer=tf.random_normal(
                    [self.config.text_gen_hidden_dim]))

            # it's tricky when it comes to teacher forcing.
            if self.teacher_forcing:
                # the loop function to be called at each time step
                loop = tf.nn.seq2seq._extract_argmax_and_embed(
                    self.word_embedding,
                    output_projection=(self.vocabulary_mat_trans,
                                       self.vocabulary_bias),
                    update_embedding=False)
            else:
                loop = None

            self.teacher_forcing_embedding = tf.nn.embedding_lookup(
                self.word_embedding, self.teacher_forcing)

            outputs, state = tf.nn.seq2seq.decoder(
                self.teacher_forcing_embedding,
                self.l0, cell, loop_function=loop)

            self.fake_sentence = [
                tf.nn.xw_plus_b(x, self.vocabulary_mat_trans,
                                self.vocabulary_bias) for x in outputs]
        return 0

    def get_fake_sentence(self):
        return self.fake_sentence
