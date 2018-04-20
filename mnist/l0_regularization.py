import math

import tensorflow as tf

GAMMA = -0.1
ZETA = 1.1
BETA = 2/3

def hard_sigmoid(x):
    return tf.minimum(tf.maximum(x,tf.zeros_like(x)),tf.ones_like(x))

def get_l0_norm(x, varname):

    shape = x.get_shape()

    # sample u
    u = tf.random_uniform(shape)

    # sample log a
    log_a = tf.Variable(tf.random_normal(shape, mean=0.0, stddev=0.01), name="log_a_" + varname)

    # compute hard concrete distribution
    s = tf.sigmoid((tf.log(u) - tf.log(1.0 - u) + log_a)/BETA)

    s_bar = s * (ZETA - GAMMA) + GAMMA

    # get differentiable l0 norm
    l0_norm = tf.reduce_sum(tf.sigmoid(log_a - BETA * math.log(-GAMMA / ZETA)), name="l0_norm_" + varname)

    # get mask which give sparse version of tensor
    mask = hard_sigmoid(s_bar)

    # return masked version of tensor and l0 norm
    return tf.multiply(x,mask, name=varname + "_masked"), l0_norm
