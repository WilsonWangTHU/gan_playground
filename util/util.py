"""
    It is the data io interface of the joint_embedding. The data reader
    class can:
        1. build or load the dictionary of the parse file
    Written by Tingwu Wang, 2016/Aug/20
"""

# import collections
# import tensorflow as tf
import numpy as np
import os
import torchfile
import random
from vgg_utils import load_image
import logger

percentage_test = 0.35
percentage_validate = 0.35  # it is deprecated
num_text_per_image = 5


class joint_embedding_data_reader():
    '''
        @brief:
            DEPRECATED
    '''

    def __init__(self, dataset_name='bird', dataset_dir=None,
                 data_parsing_file=False, stage='train', debug=False):
        assert False, logger.error('Not implemented')

        # make sure the data dir is initialized and valid
        assert dataset_name in ['bird', 'flower', 'coco']
        assert dataset_dir, '[ERROR] Please specify the data path'
        assert stage in ['train', 'test', 'validate']

        self.file_list = []
        self.train_file_list = []
        self.test_file_list = []
        self.validate_file_list = []
        self.data_id = 0
        self.dataset_name = dataset_name
        self.dataset_dir = os.path.join(dataset_dir, dataset_name)
        self.stage = stage
        self.debug = debug

        if data_parsing_file:
            # load the pre-parsed data_list
            file_abs_path = \
                os.path.join(self.dataset_dir, dataset_name + '_list.npy')
            assert os.path.exists(file_abs_path), \
                '[ERROR] Invalid parse file path {}'.format(file_abs_path)

            parse_file_list = np.load(file_abs_path,
                                      encoding='latin1').item()
            self.train_file_list = parse_file_list['train']
            self.test_file_list = parse_file_list['test']
            self.validate_file_list = parse_file_list['validate']
            del parse_file_list
            print('[LOAD DATA] Successfully loaded data from {}'.
                  format(file_abs_path))
        else:
            # parse the data list from scratch
            if dataset_name == 'bird':
                # --------------------------------------------------
                # data structure is as followed:
                # dataset_dir
                # |-raw_data
                # | |-class XXXX
                # |   |-img XXXX.t7
                # |-img
                # --------------------------------------------------
                sub_dir_list = os.listdir(os.path.join(self.dataset_dir, 'raw_data'))
                print('[LOAD DATA] {} subfile found'.format(len(sub_dir_list)))
                for sub_dir in sub_dir_list:
                    sub_file_list = os.listdir(
                        os.path.join(self.dataset_dir, 'raw_data', sub_dir))

                    for sub_file in sub_file_list:
                        self.file_list.append(os.path.join(self.dataset_dir,
                                              'raw_data', sub_dir, sub_file))

                # shuffling the data and parse the data into 3 different sets
                random.shuffle(self.file_list)
                num_data = len(self.file_list)
                num_test = int(num_data * percentage_test)
                num_val = int(num_data * percentage_validate)

                self.test_file_list = self.file_list[0: num_test]
                self.validate_file_list = \
                    self.file_list[num_test: num_test + num_val]
                self.train_file_list = self.file_list[num_test + num_val:]

                print('[LOAD DATA] Totally {} data pair found'.format(
                    len(self.file_list)))
                del self.file_list  # it is no longer useful

                # save the files
                output_save_list = {}
                output_save_list['train'] = self.train_file_list
                output_save_list['test'] = self.test_file_list
                output_save_list['validate'] = self.validate_file_list

                file_abs_path = \
                    os.path.join(self.dataset_dir, dataset_name + '_list.npy')

                np.save(file_abs_path, output_save_list)

                print('[SAVE DATA] Data parsed list saved to {}'.format(
                    file_abs_path))

        # choose the data set in use
        if self.debug:
            self.file_list_in_use = self.train_file_list[32: 63]
            print('[DEBUG] Debug mode on, working on 32 data pair')
        else:
            if self.stage == 'train':
                self.file_list_in_use = self.train_file_list
            elif self.stage == 'test':
                self.file_list_in_use = self.test_file_list
            else:
                self.file_list_in_use = self.validate_file_list
        del self.validate_file_list
        del self.train_file_list
        del self.test_file_list

    def next_batch(self, batch_size):
        # if stage = train, one image is paired with one text description
        # otherwise we have batch_size of 1 with 5 ratio
        image_list = []  # size [batch_size, 224, 224, 3]
        text_list = []  # size [batch_size, seq_max_len]
        text_seq_len_list = []  # size [batch_size]

        for i_data in range(batch_size):

            tf_path = os.path.join(self.dataset_dir, 'raw_data',
                                   self.file_list_in_use[self.data_id])
            t7_loader = torchfile.load(tf_path)
            image_path = os.path.join(self.dataset_dir, t7_loader['img'])
            image_list.append(load_image(image_path))

            if self.stage == 'train':
                # randomly choose one
                ran_int = random.randint(0, len(t7_loader['word'][0]) - 1)
                chosen_text = t7_loader['word'][:, ran_int]
                text_seq_len_list.append(len(np.where(chosen_text != 1)[0]))
                text_list.append(chosen_text)
            else:
                # we need all the data from the test set!
                chosen_text = t7_loader['word'][:, :]
                for i in range(num_text_per_image):
                    text_seq_len_list.append(len(np.where(chosen_text[:, i] != 1)[0]))
                    text_list.append(chosen_text[:, i])

            # increment on the data id
            self.data_id += 1
            if self.data_id >= len(self.file_list_in_use):
                if self.stage == 'train':
                    self.data_id = 0
                    random.shuffle(self.file_list_in_use)
                else:  # all the test data is loaded
                    break
        print('[LOAD BATCH] New batch, loading to id {}'.format(self.data_id))

        return np.array(image_list), np.array(text_list), \
            np.array(text_seq_len_list)

    def num_data_in_use(self):
        return len(self.file_list_in_use)


class tiGAN_data_reader():
    '''
        @brief: load data from the bird dataset
    '''

    def __init__(self, dataset_name='bird', dataset_dir=None, stage='train',
                 debug=False):

        # make sure the data dir is initialized and valid
        assert dataset_name in ['bird'], logger.error('Invalid dataset name')
        assert stage in ['train'], logger.error('Test dataloader not usable')
        logger.info('trying to load the dataset {}'.format(dataset_name))

        self.data_id = 0
        self.stage = stage
        self.dataset_dir = os.path.join(dataset_dir, dataset_name)
        self.dataset_name = dataset_name

        # get the split of dataset
        self.split_dataset()

        # choose the data set in use
        if self.stage == 'train':
            self.file_list_in_use = self.train_file_list
        elif self.stage == 'test':
            self.file_list_in_use = self.test_file_list

        '''
        if False:
            self.file_list_in_use = self.file_list_in_use[0:16]
            self.test_file_list = self.file_list_in_use[0:16]
        '''

        self.dataset_size = len(self.file_list_in_use)
        self.get_vocab()
        logger.info('data loader initialized')
        return

    def get_vocab(self):
        '''
            @brief: useful when sampling
        '''
        file_abs_path = \
            os.path.join(self.dataset_dir, 'vocab_c10.t7')
        vocab = torchfile.load(file_abs_path)
        self.vocab = {v: k for k, v in vocab.iteritems()}

    def split_dataset(self):
        '''
            @brief:
                get the training set and the testing set
        '''
        # load the pre-parsed data_list
        file_abs_path = \
            os.path.join(self.dataset_dir, self.dataset_name + '_list.npy')
        if os.path.exists(file_abs_path):
            parse_file_list = np.load(file_abs_path, encoding='latin1').item()
            self.train_file_list = parse_file_list['train']
            self.test_file_list = parse_file_list['test']
            del parse_file_list
            logger.info('loaded data split from {}'.format(file_abs_path))
        else:
            self.file_list = []
            logger.info('No split file found, parsing one')
            '''
                data structure is as followed:

                dataset_dir
                |-raw_data
                | |-class XXXX
                |   |-img XXXX.t7
                |-img
            '''
            sub_dir_list = \
                os.listdir(os.path.join(self.dataset_dir, 'raw_data'))
            logger.info('{} subfile found'.format(len(sub_dir_list)))
            for sub_dir in sub_dir_list:
                sub_file_list = os.listdir(
                    os.path.join(self.dataset_dir, 'raw_data', sub_dir))

                for sub_file in sub_file_list:
                    self.file_list.append(os.path.join(self.dataset_dir,
                                          'raw_data', sub_dir, sub_file))

            # shuffling the data and parse the data into 2 different sets
            random.shuffle(self.file_list)
            num_data = len(self.file_list)
            num_test = int(num_data * percentage_test)

            self.test_file_list = self.file_list[0: num_test]
            self.train_file_list = self.file_list[num_test:]

            logger.info(
                'Totally {} data pair found'.format(len(self.file_list)))
            del self.file_list  # it is no longer useful

            # save the files
            output_save_list = {}
            output_save_list['train'] = self.train_file_list
            output_save_list['test'] = self.test_file_list

            np.save(file_abs_path, output_save_list)

            logger.info('[SAVE DATA] Data parsed list saved to {}'.format(
                file_abs_path))
        return

    def next_batch(self, batch_size):
        # if stage = train, one image is paired with one text description
        # otherwise we have batch_size of 1 with 5 ratio
        image_list = []  # size [batch_size, 224, 224, 3]
        real_text_req_list = []  # size [batch_size, 1024]
        fake_text_req_list = []  # size [batch_size, 1024]

        for i_data in range(batch_size):

            # the real text
            tf_path = os.path.join(self.dataset_dir, 'raw_data',
                                   self.file_list_in_use[self.data_id])
            t7_loader = torchfile.load(tf_path)

            image_path = os.path.join(self.dataset_dir, t7_loader['img'])
            image_list.append(load_image(image_path, img_size=64))

            # the mismatched text representation
            fake_id = np.mod(
                self.data_id + random.randint(1, self.dataset_size - 1),
                self.dataset_size)
            tf_path = os.path.join(self.dataset_dir, 'raw_data',
                                   self.file_list_in_use[fake_id])
            fake_t7_loader = torchfile.load(tf_path)
            if self.stage == 'train':
                # randomly choose one real text_rep and one fake text_rep
                ran_int = random.randint(0, len(t7_loader['word'][0]) - 1)
                real_text_req_list.append(t7_loader['txt'][ran_int, :])

                # fake text rep
                ran_int = random.randint(0, len(fake_t7_loader['word'][0]) - 1)
                fake_text_req_list.append(fake_t7_loader['txt'][ran_int, :])
            else:
                assert False, logger.error('Not implemented')

            # increment on the data id
            self.data_id += 1
            if self.data_id >= self.dataset_size:
                if self.stage == 'train':
                    self.data_id = 0
                    random.shuffle(self.file_list_in_use)
                else:  # all the test data is loaded
                    break

        logger.info(
            '[LOAD BATCH] New batch, loading to id {}'.format(self.data_id))

        return np.array(image_list), np.array(real_text_req_list), \
            np.array(fake_text_req_list)

    def num_data_in_use(self):
        return len(self.file_list_in_use)

    def get_sample_data(self, sample_size):
        # if stage = train, one image is paired with one text description
        # otherwise we have batch_size of 1 with 5 ratio
        real_text_req_list = []  # size [batch_size, 1024]
        origin_text = []

        for i_data in range(sample_size):

            # the real text
            tf_path = os.path.join(self.dataset_dir, 'raw_data',
                                   self.test_file_list[i_data])
            t7_loader = torchfile.load(tf_path)

            # randomly choose one real text_rep and one fake text_rep
            ran_int = random.randint(0, len(t7_loader['word'][0]) - 1)
            real_text_req_list.append(t7_loader['txt'][ran_int, :])

            # record the text at the same time
            text = [self.vocab[word]
                    for word in t7_loader['word'][:, ran_int]
                    if word != 1]
            text_string = ' '.join(text)
            origin_text.append(text_string)

            # increment on the data id
            if i_data >= self.dataset_size:
                logger.warning('running out of test set')
                break

        logger.info(
            '[Sampling] New samples, total number: {}'.
            format(len(real_text_req_list)))

        return np.array(real_text_req_list), origin_text

    def active_shuffle(self):
        '''
            @brief: not sure if it is necessary
        '''
        random.shuffle(self.file_list_in_use)
        return

    def get_dataset_name(self):
        return self.dataset_name


if __name__ == '__main__':
    # test the train_data loading
    '''
    test = joint_embedding_data_reader(dataset_dir='/ais/gobi4/tingwuwang/joint_embedding/data_dir', data_parsing_file=True)
    image, text, text_len = test.next_batch(8)
    '''
    # test the test and validation data loading
    test = joint_embedding_data_reader(dataset_dir='/ais/gobi4/tingwuwang/joint_embedding/data_dir', data_parsing_file=True, stage='test')
    image, text, text_len = test.next_batch(8)
    '''
    print image[2]
    print 'num of image: {}'.format(len(image))
    print('[DEBUG].......Continue')
    print text[1]
    print 'num of text: {}'.format(len(text))
    print('[DEBUG].......Continue')
    print text_len[1]
    print('[DEBUG].......Continue')
    print 'num of text len: {}'.format(len(text_len))
    '''
