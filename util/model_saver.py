"""
in this model, we save the weights by our own function
TODO: The batch norm parameters are not needed
written by Tingwu Wang
"""
import tensorflow as tf
import numpy as np


def model_saver(sess, model_path,
                extra_prefix=None, is_saving_untrainables=True,
                submodel_prefix=None):
    """
        Notice, this variable only save the model with specific prefix,
        otherwise the save_all=1
    """
    output_save_list = {}  # the output to be saved

    if is_saving_untrainables is True:
        variable_list = tf.get_collection(tf.GraphKeys.VARIABLES)
    else:
        variable_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    if submodel_prefix is None:
        # we save the whole model
        for parameters in variable_list:
            weights = sess.run(parameters.name)
            output_save_list[parameters.name] = weights
    else:
        # we are only considering part of the model
        for parameters in variable_list:
            if submodel_prefix in parameters.name:
                weights = sess.run(parameters.name)
                output_save_list[parameters.name] = weights

    np.save(model_path, output_save_list)

    return True


def model_loader(sess, model_path, submodel_prefix="",
                 ignore_prefix="NOTABLE",
                 only_load_exist_variable=False):
    """
        the reader
    """
    with tf.variable_scope("", reuse=True):
        output_save_list = np.load(model_path, encoding='latin1').item()
        if only_load_exist_variable:
            # only load variables that exist!
            variable_list = tf.get_collection(tf.GraphKeys.VARIABLES)
            variable_list_name = \
                [variable.name.split(':')[0] for variable in variable_list]
            print variable_list_name
        for keys, val in output_save_list.items():
            key = keys.split(":")
            name = ''.join(key[:-1])

            if name.find(ignore_prefix) != -1:
                print('[WARNING] Parameters Not Exist {}'.format(name))
                continue
            print('[DEBUG] Parameters Exist {}'.format(name))
            if only_load_exist_variable and (name not in variable_list_name):
                print('[LOAD MODEL DEBUG] Varibale {} not existing'.
                      format(name))
                continue

            var = tf.get_variable(name=submodel_prefix + name)

            assign_op = var.assign(val)
            sess.run(assign_op)  # or `assign_op.op.run()`
    return
