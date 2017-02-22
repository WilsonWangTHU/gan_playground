# -----------------------------------------------------------------------------
#   @brief:
#       In this file, all the parameters of the network is configurated
#       For different network, we use separate parameters
#
# -----------------------------------------------------------------------------


from easydict import EasyDict as edict

__C = edict()
tiGAN_cfg = __C


__C.TRAIN = edict()
__C.TEST = edict()

__C.TRAIN.SENCODER = edict()
__C.TRAIN.IMGCODER = edict()

# basic network parameters
__C.word_embedding_space_size = 300
__C.word_voc_size = 25000
__C.encoder_dimension = 128
__C.seq_max_len = 30
__C.output_image_size = 64
__C.text_gen_hidden_dim = 2400

__C.preprocess_dimension = 128
__C.z_dimension = 100


# basic training parameters
__C.TRAIN.batch_size = 16
# __C.TRAIN.margin_alpha = 0.1
__C.TRAIN.gradient_clip = 10
__C.TRAIN.learning_rate = 0.0001
__C.TRAIN.beta1 = 0.5
__C.TRAIN.beta2 = 0.999
# __C.TRAIN.learning_decay_factor = 0.99
__C.TRAIN.max_step_size = 100000
__C.TRAIN.snapshot_step = 2000  # save the snapshot every 1000 epoches

# training parameters for the sentence encoder
__C.TRAIN.SENCODER.none_rnn_para_initial_max = 0.1
