# -----------------------------------------------------------------------------
#   @Brief:
#       some compatible issues in the tensorflow
# -----------------------------------------------------------------------------


# import math
# import numpy as np
import tensorflow as tf


def sigmoid_cross_entropy_with_logits(logits, labels):
    try:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=labels)
    except:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, targets=labels)
    return loss
