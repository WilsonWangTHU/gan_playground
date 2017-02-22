# -----------------------------------------------------------------------------
#   @brief:
#       the place we train all the ti-GAN model
#   @author: Tingwu Wang, 21st, Feb, 2017
# -----------------------------------------------------------------------------

import __init_path
from util import logger
from model.tiGAN import TI_GAN
from config import tiGAN_cfg as config
from util.util import tiGAN_data_reader
import tensorflow as tf
import argparse
import os

if __name__ == '__main__':
    # the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--restore', help='the path of model to restore',
                        default=None)

    args = parser.parse_args()

    # init the logger, just save the network ----------------------------------
    logger.set_file_handler(prefix='TIGAN_')

    # build the network and data loader ---------------------------------------
    sess = tf.Session()
    # tf.device('/gpu:' + str(args.gpu))
    logger.info('Session starts, using gpu: {}'.format(str(args.gpu)))

    tigan_net = TI_GAN(config)
    tigan_net.build_models()
    tigan_net.init_training(sess, args.restore)

    # get the data reader
    dataset_dir = os.path.join(__init_path.get_base_dir(), 'data', 'data_dir')
    data_reader = tiGAN_data_reader(dataset_name='bird',
                                    dataset_dir=dataset_dir, stage='train')

    # train the network
    tigan_net.train_net(sess, data_reader)
