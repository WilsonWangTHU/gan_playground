# ------------------------------------------------------------------------------
#   In this model, we will try to implement the encoder of the text and image.
#   The output of the model will be sent to the bigger network.
#   We use the vgg19 net from https://github.com/machrisaa/tensorflow-vgg
#   Written by Tingwu Wang, 2016/Sep/27
#   # TODO: The initilize of parameters???? HOW???
#   # TODO: the margin loss, how to choose the margin loss parameters?
#   # TODO: add the alex net.
#
#   Major Update: From 2017, Feb., 7th, we got Fartash on board!
# ------------------------------------------------------------------------------

import tensorflow as tf
# from config import cfg
import subnet.vgg.vgg19 as vgg19
from util import model_saver as ms
import util.ops as op
import time
import logging


class joint_encoder(object):
    '''
        @brief:
            the joint encoder of image and sentence. it could be used as a part
            of the network
        @components:
            def __init__
            def build_models
            def build_image_encoder
            def build_sentence_encoder
            def build_embedding_loss
            def partial_load_vgg_model
            def train_step
            def assign_lr
            def get_encoded_sentence
            def get_encoded_image
            def get_embedding_loss_placeholder
            def get_sen_img_placeholder
        @TODO:
            add the tf variable scope for better structure.
    '''

    def __init__(self, config, stage='train', is_sub_model=False):
        """
            @brief:
                The initialization of the model. we use a config file to store
                the information of configuration

        """
        self.config = config  # config is a python variable
        self.is_sub_model = is_sub_model  # if only part of the bigger net
        self.step = 0  # the num of step
        self.stage = stage  # train, test, val?

        # in 'sen_test' or 'img_test' mode, only sentence or image encoder is
        # constucted
        assert self.stage in ['train', 'test', 'val', 'img_test', 'sen_test'], \
            '[ERROR] Invalid stage of the network'

    def build_models(self, image_input, text_input, input_seq_len,
                     vgg_train_mode, train_vgg=False):
        '''
            @brief:
                build the sentence encoder, and the image encoder. The output
                is a embedding vector. note the input of the network is given
                outside the network (so that this network could be used as a
                subnet)
            @input:
                @"image_input": the input image. It is a tf.placeholder with
                size: [batch_size, 224, 224, 3].

                @"text_input": the input caption or sentence. It is a
                tf.placeholder with size: [batch_size, MAX_SEQ_LEN]

                @"input_seq_len": the length of each input sentence. It is a
                tf.placeholder of size: [batch_size]

                @"vgg_train_mode": the setting of vgg network. If set to 1,
                then the vgg part is trainable.
        '''

        self.image_input = image_input
        self.text_input = text_input
        self.input_seq_len = input_seq_len
        self.vgg_train_mode = vgg_train_mode

        if self.stage != 'sen_test':
            self.build_image_encoder(image_input, vgg_train_mode)
            print("[INIT MODEL]    Image encoder built!")

        if self.stage != 'image_test':
            self.build_sentence_encoder(text_input, input_seq_len)
            print("[INIT MODEL]    Sentence encoder built!")

        # now the embedding loss of the model
        if self.stage == 'train' and (not self.is_sub_model):
            print("[INIT MODEL] Building the joint encoder!")
            self.build_embedding_loss()
        print("[INIT MODEL] Joint encoder built!")

        # initialize the learning rate and training operation
        if self.is_sub_model or (not self.stage == 'train'):
            return

        tvars = tf.trainable_variables()
        if not train_vgg:  # keep the vgg part fixed
            tvars = [var for var in tvars if 'VGG/' not in var.name]
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.embedding_loss, tvars),
            self.config.TRAIN.gradient_clip)
        self._learning_rate = tf.get_variable(
            'learning_rate',
            initializer=tf.constant(
                self.config.TRAIN.learning_rate, dtype=tf.float64),
            trainable=False)

        # TODO: LOADING THE ADAM PARAMETERS?
        # optimizer = tf.train.GradientDescentOptimizer(self._learning_rate)
        optimizer = tf.train.GradientDescentOptimizer(self._learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        print("[INIT MODEL] Training op initilized!")
        return

    def build_image_encoder(self, image_input, vgg_train_mode):
        # it is a VGG-19 network
        with tf.variable_scope('VGG'):
            self.vgg = vgg19.Vgg19(load_old_model=False)
            self.vgg.build(image_input, vgg_train_mode)

        # now we got the fc layer!
        self.fc_layer_variable = self.vgg.fc_layer_variable
        with tf.variable_scope('VGG2REP'):
            weight = tf.get_variable(
                'weights', initializer=tf.truncated_normal(
                    [4096, self.config.encoder_dimension], 0.0,
                    self.config.TRAIN.SENCODER.none_rnn_para_initial_max))
            bias = tf.get_variable(
                'bias',
                initializer=tf.truncated_normal(
                    [self.config.encoder_dimension], 0.0,
                    self.config.TRAIN.SENCODER.none_rnn_para_initial_max))

        self.image_rep = tf.matmul(self.fc_layer_variable, weight) + bias
        self.image_rep = tf.nn.l2_normalize(self.image_rep, 1)
        return

    def build_sentence_encoder(self, raw_encoder_input, input_seq_len):
        """
            1. "text_input" is the input caption. It is a tf.variable with
            @Size: [batch_size, MAX_SEQ_LEN]
        """
        with tf.variable_scope('text_encoder'):
            self.embedding = \
                tf.get_variable(
                    "embedding", initializer=tf.random_uniform(
                        [self.config.word_voc_size,
                            self.config.word_embedding_space_size],
                        -self.config.TRAIN.SENCODER.none_rnn_para_initial_max,
                        self.config.TRAIN.SENCODER.none_rnn_para_initial_max))
            inputs = tf.nn.embedding_lookup(self.embedding, raw_encoder_input)

            # now it is [MAX_SEQ_LENGTH, batch_size, embedding_length]
            input_batch_order = tf.transpose(inputs, [1, 0, 2])

            # now it is [MAX_SEQ_LENGTH * batch_size, embedding_length]
            input_batch_order = tf.reshape(
                input_batch_order, [-1, self.config.word_embedding_space_size])

            # now it is LIST OF [BATCH_SIZE, embedding_length]
            encoder_input = tf.split(0, self.config.seq_max_len,
                                     input_batch_order)

            # the encoder part
            encode_gru_cell = tf.nn.rnn_cell.GRUCell(
                self.config.encoder_dimension)
            # big news: The state is final state, output is a list of tensor.
            # We don't to do that
            _, sentence_rep = tf.nn.rnn(encode_gru_cell, encoder_input,
                                        dtype=tf.float32,
                                        sequence_length=input_seq_len)
            self.sentence_rep = sentence_rep
            self.sentence_rep = tf.nn.l2_normalize(self.sentence_rep, 1)
        return

    def build_embedding_loss(self):
        '''
            @brief: building the contrastive loss.
            @output: the 'self.embedding_loss' is what we are trying to do
        '''
        # both the image_rep and the sentence_rep are of size [batch_size,
        # 2048 (the feature size)]
        image_feature_list = tf.split(
            0, self.config.TRAIN.batch_size, self.image_rep)
        text_feature_list = tf.split(
            0, self.config.TRAIN.batch_size, self.sentence_rep)

        image_text_result = []
        # calculate the loss from the image perspective
        for i_image_sample in range(self.config.TRAIN.batch_size):
            the_positive_energy = tf.sqrt(tf.reduce_sum(
                tf.pow(image_feature_list[i_image_sample] -
                       text_feature_list[i_image_sample], 2)))

            # size of [batch_size, 2048]
            diff_matrix = tf.sub(self.sentence_rep,
                                 image_feature_list[i_image_sample])

            diff_i = tf.sqrt(tf.reduce_sum(tf.pow(diff_matrix, 2), 1))

            # the result for image i
            image_text_result.append(
                tf.maximum(0.0, self.config.TRAIN.margin_alpha -
                           diff_i + the_positive_energy))

        # calculate the loss from the text perspective
        for i_text_sample in range(self.config.TRAIN.batch_size):
            the_positive_energy = tf.sqrt(tf.reduce_sum(
                tf.nn.l2_loss(image_feature_list[i_text_sample] -
                              text_feature_list[i_text_sample])))

            # size of [batch_size, 2048]
            diff_matrix = \
                tf.sub(self.image_rep, text_feature_list[i_text_sample])
            diff_i = tf.sqrt(tf.reduce_sum(tf.pow(diff_matrix, 2), 1))
            image_text_result.append(
                tf.maximum(0.0, self.config.TRAIN.margin_alpha -
                           diff_i + the_positive_energy))

        # the overall loss of the network
        self.embedding_loss = tf.reduce_sum(tf.concat(0, image_text_result))

    def partial_load_vgg_model(self, sess, vgg_model_path):
        '''
            only use when we first intialize the pretrained model
        '''
        ms.model_loader(sess, vgg_model_path, submodel_prefix="VGG/")
        print('[INIT MODEL] VGG pretrained model loaded')
        return

    def train_step(self, sess, image_input, text_input, input_seq_len, vgg_train_mode):
        """Runs the model on the given data."""
        start_time = time.time()
        self.step += 1
        cost, _ = sess.run([self.embedding_loss, self.train_op],
                           {self.image_input: image_input,
                               self.text_input: text_input,
                               self.input_seq_len: input_seq_len,
                               self.vgg_train_mode: vgg_train_mode})
        print("[TRAIN STEP] {} takes {}s\n[LOSS]    with the loss of {}"
              .format(self.step, time.time() - start_time, cost))
        return

    def assign_lr(self, sess, lr_value):  # the assigning of new variable
        sess.run(tf.assign(self._learning_rate, lr_value))

    def get_encoded_sentence(self, sess, text_input, input_seq_len):
        assert self.stage == 'sentence_test', \
            'Only able to encode the sentence in "sentence_test" stage'
        start_time = time.time()
        test_sentence_rep = sess.run([self.sentence_rep],
                                     {self.text_input: text_input,
                                         self.input_seq_len: input_seq_len,
                                         self.vgg_train_mode: False})
        print("[TEST SENTENCE] takes {}s"
              .format(time.time() - start_time))
        return test_sentence_rep

    def get_encoded_image(self, sess, image_input):
        assert self.stage == 'image_test', \
            'Only able to encode the sentence in "image_test" stage'
        start_time = time.time()
        self.step += 1
        image_rep = sess.run([self.image_rep],
                             {self.image_input: image_input,
                                 self.vgg_train_mode: False})
        print("[TEST IMAGE] takes {}s"
              .format(time.time() - start_time))
        return image_rep

    def get_embedding_loss_placeholder(self):
        return self.embedding_loss

    def get_sen_img_placeholder(self):
        return self.sentence_rep, self.image_rep

    def get_embedding_matrix(self):
        return self.embedding




class joint_generator(object):
    '''
        @NOT FUNCTIONAL!
        @brief:
            The conditional generator. In the first edition, let's assume the
            image embedding vector is conditioned to generate the text, and the
            text vector is conditioned to generate the image (we have more
            options though)
        @components:
            def __init__
            def build_models
    '''

    def __init__(self, config, stage='train', is_sub_model=True,
                 teacher_forcing=True, word_embedding=None):
        """
            @brief:
                The initialization of the model. we use a config file to store
                the information of configuration

        """
        assert False, logger.error('Joint generator not functional')
        self.config = config  # config is a python variable
        self.is_sub_model = is_sub_model  # if only part of the bigger net
        self.step = 0  # the num of step
        self.stage = stage  # train, test, val?
        self.train = self.stage == 'train'
        self.teacher_forcing = teacher_forcing
        self.word_embedding = word_embedding

        # in 'sen_test' or 'img_test' mode, only sentence or image encoder is
        # constucted
        assert self.stage in ['train', 'test', 'val', 'img_test', 'sen_test'], \
            '[ERROR] Invalid stage of the network'

    def build_models(self, img_rep, sen_rep, sen_z, img_z, teacher_input=None):
        '''
            @brief:
                build the sentence encoder, and the image encoder. The output
                is a embedding vector. note the input of the network is given
                outside the network (so that this network could be used as a
                subnet)
            @input:

            @notes:
            @Tingwu: About the network, we sure have lots of solutions.. I am
                going to write a baseline, but we should consider the
                convenience issues to further boost our network
        '''

        # the condition vector of the network
        self.img_rep = img_rep
        self.sen_rep = sen_rep
        self.batch_size = self.config.TRAIN.batch_size
        '''
        self.sen_z = \
            tf.placeholder(tf.float32, [self.batch_size,
                           self.config.z_dimension], name='sen_z')

        self.img_z = \
            tf.placeholder(tf.float32, [self.batch_size,
                           self.config.z_dimension], name='img_z')
        '''
        self.sen_z = sen_z
        self.img_z = img_z

        if self.teacher_forcing:
            assert (teacher_input is not None) and \
                (self.word_embedding is not None), \
                '[ERROR] Please specify the teacher input!'
            self.teacher_input = teacher_input

        if self.stage != 'sen_test':
            self.build_image_generator()
            print("[INIT MODEL]    Image generator built!")

        if self.stage != 'image_test':
            self.build_sentence_encoder()
            print("[INIT MODEL]    sentence generator built!")

    '''
    @deprecated
    def get_next_word(self):
        # in this function, we calculate the predicted word based on the score

        return
    '''



