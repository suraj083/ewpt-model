"""Created on Tuesday, February 21st 2023
   Author: Suraj Prakash
"""

import tensorflow as tf
import keras
import os


model_Jb = keras.models.load_model(f"{os.getcwd()}/ewpt/Jb_model") 
model_Jf = keras.models.load_model(f"{os.getcwd()}/ewpt/Jf_model") 
model_xlogx = keras.models.load_model(f"{os.getcwd()}/ewpt/xlogx_model")

# def fitted_JB(var):
#     return model_Jb(var)

def fitted_JB(var): 
    if tf.shape(var).shape[0] == 1:
        return model_Jb(var)
    elif tf.shape(var).shape[0] == 0:   # useful when creating single variable plots
        new_var = tf.reshape(var, (1,))
        return model_Jb(new_var)
    elif tf.shape(var).shape[0] == 2:   # useful when creating meshgrids for contour or 3d plots
        return tf.stack([model_Jb(elem) for elem in tf.unstack(var)])[:,:,0]

# def fitted_JF(var):
#     return model_Jf(var)

def fitted_JF(var): 
    if tf.shape(var).shape[0] == 1:
        return model_Jf(var)
    elif tf.shape(var).shape[0] == 0:
        new_var = tf.reshape(var, (1,))
        return model_Jf(new_var)
    elif tf.shape(var).shape[0] == 2:
        return tf.stack([model_Jf(elem) for elem in tf.unstack(var)])[:,:,0]

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
  