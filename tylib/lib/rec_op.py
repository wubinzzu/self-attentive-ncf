from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .nn import *
from .att_op import *
from .compose_op import *

""" Interaction functions for RecModel.
This file comprises MF, MLP, JRL and SA-NCF
"""

def mf(a, b,  dropout,
        reuse=None, initializer=None,
        num_dense=3, hdim=50, generalized=False):
    output = a * b
    dim = a.get_shape().as_list()[1]
    if(generalized):
        h = tf.get_variable(
                    "hidden", [dim, 1],
                    initializer=initializer,
                    )
        output = tf.matmul(output, h)
    else:
        output = tf.reduce_sum(output, 1, keepdims=True)
    return output

def mlp(a, b, dropout, reuse=None, initializer=None, num_dense=3, hdim=50,
        last_layer=True, pyramid=True):
    if(b is None):
        input_vec = a
    else:
        input_vec = tf.concat([a, b], 1)
    _dim = hdim
    all_layers = []
    for i in range(0,num_dense-1):
        input_vec = linear(input_vec, _dim,
                            initializer,
                            name='layer_{}'.format(i),
                            reuse=reuse)
        input_vec = tf.nn.relu(input_vec)
        input_vec = tf.nn.dropout(input_vec, dropout)
        all_layers.append(input_vec)
        if(pyramid):
            _dim = int(_dim / 2)
    if(last_layer):
        output = linear(input_vec, 1, initializer,
                        name='final_layer', reuse=reuse)
    else:
        output = input_vec
    if(pyramid==False):
        # Note: this is just to extract the dot product of all hidden
        # layers, for MLP, this is not connected to the loss function
        stack = tf.stack(all_layers)
        stack = tf.reshape(stack,[-1, num_dense-1, _dim])
        stack2, _, _, _, afm = co_attention(stack,
                            stack,
                            att_type='SOFT',
                            pooling='MATRIX',
                            mask_diag=True,
                            kernel_initializer=initializer,
                            activation=None,
                            dropout=None,
                            seq_lens=None,
                            transform_layers=0,
                            name='dummy_self_att',
                            reuse=reuse)
    else:
        afm = []
    return output, afm

def highway_networks(a, b, dropout, reuse=None,
                initializer=None, num_dense=3, hdim=50,
                last_layer=True, pyramid=True):
    """ Standard Highway Networks.
    Note we tested this and doesn't seem to work well...
    """
    if(b is None):
        input_vec = a
    else:
        input_vec = tf.concat([a, b], 1)
    _dim = hdim
    for i in range(0,num_dense-1):
        output = highway_layer(input_vec, _dim,
                        initializer, name='hw_{}'.format(i),
                        reuse=reuse)
        if(pyramid):
            _dim = int(_dim / 2)
    if(last_layer):
        output = linear(input_vec, 1, initializer,
                        name='final_layer', reuse=reuse)
    else:
        output = input_vec
    return output

def jrl(a, b, dropout, reuse=None, initializer=None,
        num_dense=3, hdim=50,
        last_layer=True, pyramid=True):
    """ Joint Representation Learning (JRL) model
    """
    input_vec = a * b
    _dim = hdim
    for i in range(0,num_dense-1):
        input_vec = linear(input_vec, _dim,
                            initializer,
                            name='layer_{}'.format(i),
                            reuse=reuse)
        input_vec = tf.nn.elu(input_vec)
        # input_vec = tf.nn.relu(input_vec)
        input_vec = tf.nn.dropout(input_vec, dropout)
        if(pyramid):
            _dim = int(_dim / 2)
    if(last_layer):
        output = linear(input_vec, 1, initializer,
                        name='final_layer', reuse=reuse)
    else:
        output = input_vec
    return output

def self_attentive_ncf(a, b, dropout, reuse=None, initializer=None,
                num_dense=3, hdim=50, last_layer=True,
                is_train=None, pyramid=False,
                enhanced=False):
    """ self-attentive neural collaborative filtering

    Args:
        a, b: Tensors of [bsz x dim] corresponding to user-item embed
        dropout: optional dropout
        reuse: to reuse params or not (usually for neg-side)
        initializer: tf-initializer obj
        num_dense: number of hidden layers
        hdim: hidden layer size
        last_layer: to project to scalar or not.
        is_train:   state of the compute graph
        pyramid:    to shrink the model in tower structure. (not used here)
        enhanced:   to add the inner product to the concat (optional/not used)

    Returns:
        output: output representation (scalar if last_layer is true, else hdim)
        afm:    affinity matrix for self-att (viz purposes only)

    """
    stack = []
    input_vec = tf.concat([a, b], 1)
    if(enhanced):
        input_vec = tf.concat([input_vec, a*b], 1)
    _dim = hdim
    for i in range(0,num_dense-1):
        input_vec = linear(input_vec, _dim,
                            initializer,
                            name='layer_{}'.format(i),
                            reuse=reuse)
        input_vec = tf.nn.relu(input_vec)
        input_vec = tf.nn.dropout(input_vec, dropout)
        stack.append(input_vec)
        if(pyramid):
            _dim = int(_dim / 2)
    stack = tf.stack(stack)
    stack = tf.reshape(stack,[-1, num_dense-1, _dim])
    stack2, _, _, _, afm = co_attention(stack,
                        stack,
                        att_type='SOFT',
                        pooling='MATRIX',
                        mask_diag=True,
                        kernel_initializer=initializer,
                        activation=None,
                        dropout=None,
                        seq_lens=None,
                        transform_layers=0,
                        name='self_att',
                        reuse=reuse)
    stack2 = tf.reduce_sum(stack2, 1)
    input_vec = tf.concat([input_vec, stack2], 1)
    if(last_layer):
        output = linear(input_vec, 1, initializer,
                        name='final_layer', reuse=reuse)
    else:
        output = input_vec
    return output, afm

def linear_output(input_vec):
    _dim = input_vec.get_shape().as_list()[1]
    fm_w0 = tf.get_variable("fm_w0", [1],
                    initializer=tf.constant_initializer([0]))
    fm_w = tf.get_variable("fm_w", [_dim],
                initializer = tf.constant_initializer([0]))
    linear_part = fm_w0 +  tf.matmul(input_vec,
                             tf.expand_dims(fm_w, 1))
    return linear_part
