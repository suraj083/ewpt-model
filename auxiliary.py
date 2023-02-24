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
    return (tf.exp(4.767230860012938 * x) - tf.exp(4.767230860012938 * 0.01) - tf.sqrt(0.01) ) * tf.exp(-3000*(x - 0.01)**2 )


def reg_sq_root(x):
    return tf.where(tf.math.less(x, 0.01), sqrt_new(x), tf.sqrt(x))
