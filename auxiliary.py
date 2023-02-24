"""Created on Tuesday, February 21st 2023
   Author: Suraj Prakash
"""

import tensorflow as tf

# the way set(list) vs set(dict) behaves, there is a possible source of confusion here

def fill_params(list1: list, dict2: dict) -> dict:
    if len(list1) > len(dict2):
        for key in list(set(list1).difference(set(dict2.keys()))):
            dict2[key] = 0.0
        return dict2

    else: 
        return dict2

def sqrt_new(x):
    return (tf.exp(5 * x) - tf.exp(5 * 1e-8) - tf.sqrt(1e-8) ) * tf.exp(-30000*(x - 1e-8)**2 )


def reg_sq_root(x1):
    x = tf.cast(x1, tf.float32)
    return tf.where(tf.math.less(x, 1e-8), sqrt_new(x), tf.sqrt(x))
