# -----------------------------------------------------------------------------
#   @brief:
#       the place we train all the ti-GAN model
#   @author: Tingwu Wang, 21st, Feb, 2017
# -----------------------------------------------------------------------------

import init_path
from util import logger
from model.tiGAN import TI_GAN
from model.DCGAN import DC_GAN
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
    parser.add_argument('--dcgan', default=False)

    args = parser.parse_args()

    # init the logger, just save the network ----------------------------------
    if not args.dcgan:
        logger.set_file_handler(prefix='TIGAN_')
        gan_net = TI_GAN(config)
        logger.info('Training TIGAN')
    else:
        logger.set_file_handler(prefix='DCGAN_')
        gan_net = DC_GAN(config)
        logger.info('Training DCGAN')

    # build the network and data loader ---------------------------------------
    sess = tf.Session()
    # tf.device('/gpu:' + str(args.gpu))
    logger.info('Session starts, using gpu: {}'.format(str(args.gpu)))

    gan_net.build_models()
    gan_net.init_training(sess, args.restore)

    # get the data reader
    dataset_dir = os.path.join(init_path.get_base_dir(), 'data', 'data_dir')
    data_reader = tiGAN_data_reader(dataset_name='bird',
                                    dataset_dir=dataset_dir, stage='train',
                                    debug=True)
    if args.restore is not None:
        data_reader.active_shuffle()

    # train the network
    logger.info('Training starts, using gpu: {}'.format(str(args.gpu)))
    gan_net.train_net(sess, data_reader)
