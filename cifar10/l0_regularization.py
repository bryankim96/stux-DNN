import math

import tensorflow as tf

GAMMA = -0.1
ZETA = 1.1
BETA = 2/3

def hard_sigmoid(x):
    return tf.minimum(tf.maximum(x,tf.zeros_like(x)),tf.ones_like(x))

def runtime_l0_norms(log_a_vars):
    l0_norm_vars = []

    for var in log_a_vars:
        l0_norm_vars.append(tf.reduce_sum(tf.sigmoid(var - BETA *
                                                     math.log(-GAMMA / ZETA))))

    return l0_norm_vars


def get_mask(x, log_a, u):

    # shape_batched = x.get_shape()
    # shape_list = shape_batched.as_list()

    # shape = tf.TensorShape(shape_list[1:])
    
    # sample u
    # u = tf.random_uniform(shape)

    # initialize log a from normal distribution
    # log_a = tf.Variable(tf.random_normal(shape, mean=0.0, stddev=0.01),
    #                     name=varname + "_log_a")

    # compute hard concrete distribution
    s = tf.sigmoid((tf.log(u) - tf.log(1.0 - u) + log_a)/BETA)

    # stretch hard concrete distribution
    s_bar = s * (ZETA - GAMMA) + GAMMA
    
    # l0_norm_val = tf.reduce_sum(tf.sigmoid(log_a - BETA * math.log(-GAMMA / ZETA)))
    
    # compute differentiable l0 norm
    # l0_norm = tf.Variable(l0_norm_val, name=varname + "_l0_norm")
    # l0_norm = tf.assign(l0_norm, l0_norm_val)

    # get mask for calculating sparse version of tensor
    mask = hard_sigmoid(s_bar)# , name=varname + "_sparsity_mask")

    # mask_var = tf.Variable(mask, name=varname + "_mask")

    # mask_var = tf.assign(mask_var, mask)

    # return masked version of tensor and l0 norm
    # return tf.multiply(x,mask_var, name=varname + "_masked"), l0_norm

    return mask

def get_mask_and_norm(x, log_a, varname):

    shape_batched = x.get_shape()
    shape_list = shape_batched.as_list()

    shape = tf.TensorShape(shape_list[1:])
    
    # sample u
    u = tf.random_uniform(shape)

    # initialize log a from normal distribution
    #log_a = tf.Variable(tf.random_normal(shape, mean=0.0, stddev=0.01),
    #                     name=varname + "_log_a")

    # compute hard concrete distribution
    s = tf.sigmoid((tf.log(u) - tf.log(1.0 - u) + log_a)/BETA)

    # stretch hard concrete distribution
    s_bar = s * (ZETA - GAMMA) + GAMMA
    
    # compute differentiable l0 norm
    l0_norm = tf.reduce_sum(tf.sigmoid(log_a - BETA * math.log(-GAMMA / ZETA)))
    
    # get mask for calculating sparse version of tensor
    mask = hard_sigmoid(s_bar)

    # return masked version of tensor and l0 norm
    return tf.multiply(x,mask, name=varname + "_masked_diff_val"), l0_norm
