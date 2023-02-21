"""Created on Tuesday, February 21st 2023
   Author: Suraj Prakash
"""

import tensorflow as tf
import keras

model_Jb = keras.models.load_model("Jb_model") 
model_Jf = keras.models.load_model("Jf_model") 
model_xlogx = keras.models.load_model("xlogx_model")


def fitted_JB(var):
    return model_Jb(var)

def fitted_JF(var):
    return model_Jf(var)

def fitted_dJB(var):
    var_b = tf.cast(var, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(var_b) 
        y_Jb = model_Jb(var_b)
    return tape.gradient(y_Jb, var_b)

def fitted_dJF(var):
    var_f = tf.cast(var, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(var_f) 
        y_Jf = model_Jf(var_f)
    return tape.gradient(y_Jf, var_f)

def fitted_xlogx(var):
    return model_xlogx(var)
  