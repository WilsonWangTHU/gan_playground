# -----------------------------------------------------------------------------
#   @brief:
#       In this file, all the parameters of the network is configurated
#       For different network, we use separate parameters
#
# -----------------------------------------------------------------------------


from easydict import EasyDict as edict

__C_tiGAN = edict()
tiGAN_cfg = __C_tiGAN


__C_tiGAN.TRAIN = edict()
__C_tiGAN.TEST = edict()

__C_tiGAN.TRAIN.SENCODER = edict()
__C_tiGAN.TRAIN.IMGCODER = edict()

# basic network parameters
__C_tiGAN.output_image_size = 64

__C_tiGAN.preprocess_dimension = 128
__C_tiGAN.z_dimension = 100
__C_tiGAN.generator_l1_nchannel = 1024


# basic training parameters
__C_tiGAN.TRAIN.batch_size = 64
__C_tiGAN.TRAIN.learning_rate = 0.0002
__C_tiGAN.TRAIN.beta1 = 0.5
__C_tiGAN.TRAIN.beta2 = 0.999
__C_tiGAN.TRAIN.max_step_size = 100000
__C_tiGAN.TRAIN.snapshot_step = 2000  # save the snapshot every 1000 epoches


# basic sampling parameters
__C_tiGAN.TEST.sample_size = __C_tiGAN.TRAIN.batch_size


'''
DEPRECATED:

__C.word_embedding_space_size = 300
__C.word_voc_size = 25000
__C.encoder_dimension = 128
__C.seq_max_len = 30
__C.text_gen_hidden_dim = 2400

# __C.TRAIN.margin_alpha = 0.1
__C.TRAIN.gradient_clip = 10

# training parameters for the sentence encoder
__C.TRAIN.SENCODER.none_rnn_para_initial_max = 0.1
# __C.TRAIN.learning_decay_factor = 0.99
'''
