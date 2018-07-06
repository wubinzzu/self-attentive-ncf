from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from .nn import *
from .compose_op import *

def mask_3d(values, sentence_sizes, mask_value, dimension=2):
    """ Given a batch of matrices, each with shape m x n, mask the values in each
    row after the positions indicated in sentence_sizes.
    This function is supposed to mask the last columns in the raw attention
    matrix (e_{i, j}) in cases where the sentence2 is smaller than the
    maximum.

    Source https://github.com/erickrf/multiffn-nli/

    Args:
        values: `tensor` with shape (batch_size, m, n)
        sentence_sizes: `tensor` with shape (batch_size) containing the
            sentence sizes that should be limited
        mask_value: `float` to assign to items after sentence size
        dimension: `int` over which dimension to mask values

    Returns
        A tensor with the same shape as `values`
    """
    if dimension == 1:
        values = tf.transpose(values, [0, 2, 1])
    time_steps1 = tf.shape(values)[1]
    time_steps2 = tf.shape(values)[2]

    ones = tf.ones_like(values, dtype=tf.int32)
    pad_values = mask_value * tf.cast(ones, tf.float32)
    mask = tf.sequence_mask(sentence_sizes, time_steps2)

    # mask is (batch_size, sentence2_size). we have to tile it for 3d
    mask3d = tf.expand_dims(mask, 1)
    mask3d = tf.tile(mask3d, (1, time_steps1, 1))
    mask3d = tf.cast(mask3d, tf.float32)

    masked = values * mask3d
    # masked = tf.where(mask3d, values, pad_values)

    if dimension == 1:
        masked = tf.transpose(masked, [0, 2, 1])

    return masked

def matrix_softmax(values):
    ''' Implements a matrix-styled softmax

    Args:
        values `tensor` [bsz x a_len, b_len]

    Returns:
        A tensor of the same shape
    '''
    original_shape = tf.shape(values)
    num_units = original_shape[2]
    reshaped = tf.reshape(values, tf.stack([-1, num_units]))
    softmaxed = tf.nn.softmax(reshaped)
    return tf.reshape(softmaxed, original_shape)

def softmax_mask(val, mask):
    return -1E-30 * (1 - tf.cast(mask, tf.float32)) + val

def co_attention(input_a, input_b, reuse=False, name='', att_type='TENSOR',
                pooling='MEAN', k=10, mask_diag=False, kernel_initializer=None,
                dropout=None, activation=None, seq_lens=[], clipped=False,
                transform_layers=0, proj_activation=tf.nn.relu,
                model_type="", mask_a=None, mask_b=None):
    ''' Implements a Co-Attention Mechanism

    This attention uses tiling method (this uses more RAM, but enables
    MLP or special types of interaction functions between vectors.)

    Note: For self-attention, set input_a and input_b to be same tensor.

    Args:
        input_a: `tensor`. Shape=[bsz x max_steps x dim]
        input_b: `tensor`. Shape=[bsz x max_steps x dim]
        reuse:  `bool`. To reuse weights or not
        name:   `str`. Variable name
        att_type: `str`. Supports 'DOT', 'BILINEAR','TENSOR','MLP' and 'MD'
        pooling: 'str'. supports "MEAN",'MAX','SUM', "MATRIX" pooling
        k:  `int`. For multi-dimensional. Num_slice tensor or hidden
            layer.
        mask_diag: `bool` Supports masking against diagonal for self-att
        kernel_initializer: `Initializer function
        dropout: `tensor` dropout placeholder (default is disabled)
        activation: Activation function
        seq_lens: `list of 2 tensors` actual seq_lens for
            input_a and input_b

    Returns:
        final_a: `tensor` Weighted representation of input_a.
        final_b: `tensor` Weighted representation of input_b.
        max_row: `tensor` Row-based attention weights.
        max_col: `tensor` Col-based attention weights.
        y:  `tensor` Affinity matrix

    '''

    if(kernel_initializer is None):
        kernel_initializer = tf.random_uniform_initializer()

    if(len(input_a.get_shape().as_list())<=2):
        # expand dims
        input_a = tf.expand_dims(input_a, 2)
        input_b = tf.expand_dims(input_b, 2)
        readjust = True
    else:
        readjust = False

    orig_a = input_a
    orig_b = input_b
    a_len = tf.shape(input_a)[1]
    b_len = tf.shape(input_b)[1]
    input_dim = tf.shape(input_a)[2]
    if(clipped):
        max_len = tf.reduce_max([tf.shape(input_a)[1],
                                tf.shape(input_b)[2]])
    else:
        max_len = a_len

    shape = input_a.get_shape().as_list()
    dim = shape[2]

    if(transform_layers>=1):
        input_a = projection_layer(input_a,
                                dim,
                                name='att_proj_{}'.format(name),
                                activation=proj_activation,
                                initializer=kernel_initializer,
                                dropout=None,
                                reuse=reuse,
                                num_layers=transform_layers,
                                use_mode='None')
        input_b = projection_layer(input_b,
                                dim,
                                name='att_proj_{}'.format(name),
                                activation=proj_activation,
                                reuse=True,
                                initializer=kernel_initializer,
                                dropout=None,
                                num_layers=transform_layers,
                                use_mode='None')
    if(att_type == 'BILINEAR'):
        # Bilinear Attention
        with tf.variable_scope('att_{}'.format(name), reuse=reuse) as f:
            weights_U = tf.get_variable("weights_U", [dim, dim],
                                        initializer=kernel_initializer)
        _a = tf.reshape(input_a, [-1, dim])
        z = tf.matmul(_a, weights_U)
        z = tf.reshape(z, [-1, a_len, dim])
        y = tf.matmul(z, tf.transpose(input_b, [0, 2, 1]))
    elif(att_type == 'TENSOR'):
        # Tensor based Co-Attention
        with tf.variable_scope('att_{}'.format(name),
                                reuse=reuse) as f:
            weights_U = tf.get_variable(
                    "weights_T", [dim, dim * k],
                    initializer=kernel_initializer)
            _a = tf.reshape(input_a, [-1, dim])
            z = tf.matmul(_a, weights_U)
            z = tf.reshape(z, [-1, a_len * k, dim])
            y = tf.matmul(z, tf.transpose(input_b, [0, 2, 1]))
            y = tf.reshape(y, [-1, a_len, b_len, k])
            y = tf.reduce_max(y, 3)
    elif(att_type=='SOFT'):
        # Soft match without parameters
        _b = tf.transpose(input_b, [0,2,1])
        z = tf.matmul(input_a, _b)
        y = z
    elif(att_type=='DOT'):
        # print("Using DOT-ATT")
        input_a = projection_layer(input_a,
                                dim,
                                name='dotatt_{}'.format(name),
                                activation=tf.nn.relu,
                                initializer=kernel_initializer,
                                dropout=None,
                                reuse=reuse,
                                num_layers=1,
                                use_mode='None')
        input_b = projection_layer(input_b,
                                dim,
                                name='dotatt_{}'.format(name),
                                activation=tf.nn.relu,
                                reuse=True,
                                initializer=kernel_initializer,
                                dropout=None,
                                num_layers=1,
                                use_mode='None')
        _b = tf.transpose(input_b, [0,2,1])
        z = tf.matmul(input_a, _b)
        z = z / (dim ** 0.5)
        y = tf.reshape(z, [-1, a_len, b_len])
    else:
        a_aug = tf.tile(input_a, [1, b_len, 1])
        b_aug = tf.tile(input_b, [1, a_len, 1])
        output = tf.concat([a_aug, b_aug], 2)
        if(att_type == 'MLP'):
            # MLP-based Attention
            sim = projection_layer(output, 1,
                                name='{}_co_att'.format(name),
                                reuse=reuse,
                                num_layers=1,
                                activation=None)
            y = tf.reshape(sim, [-1, a_len, b_len])
        elif(att_type=='DOTMLP'):
            # Add dot product
            _dim = input_a.get_shape().as_list()[2]
            sim = projection_layer(output, 1,
                                name='dotmlp',
                                reuse=reuse,
                                num_layers=1,
                                activation=None)

            y = tf.reshape(sim, [-1, a_len, b_len])
        elif(att_type == 'MD'):
            # Multi-dimensional Attention
            sim = projection_layer(output, k,
                                    name='co_att', reuse=reuse,
                                    activation=tf.nn.relu)
            feat = tf.reshape(sim, [-1, k])
            sim_matrix = tf.contrib.layers.fully_connected(
                                   inputs=feat,
                                   num_outputs=1,
                                   weights_initializer=kernel_initializer,
                                   activation_fn=None)
            y = tf.reshape(sim_matrix, [-1, a_len, b_len])

    if(activation is not None):
        y = activation(y)

    if(mask_diag):
        # Create mask to prevent matching against itself
        mask = tf.ones([a_len, b_len])
        mask = tf.matrix_set_diag(mask, tf.zeros([max_len]))
        y = y * mask

    if(pooling=='MATRIX'):
        _y = tf.transpose(y, [0,2,1])
        if(mask_a is not None and mask_b is not None):
            mask_b = tf.expand_dims(mask_b, 1)
            mask_a = tf.expand_dims(mask_a, 1)
            # bsz x 1 x b_len
            mask_a = tf.tile(mask_a, [1, b_len, 1])
            mask_b = tf.tile(mask_b, [1, a_len, 1])
            _y = softmax_mask(_y, mask_a)
            y = softmax_mask(y, mask_b)
        else:
            print("[Warning] Using Co-Attention without Sequence Mask!\
                    Please ignore this if seq_len is fixed.")
        att2 = tf.nn.softmax(_y)
        att1 = tf.nn.softmax(y)
        final_a = tf.matmul(att2, orig_a)
        final_b = tf.matmul(att1, orig_b)
        _a2 = att2
        _a1 = att1
    else:
        if(pooling=='MAX'):
            att_row = tf.reduce_max(y, 1)
            att_col = tf.reduce_max(y, 2)
        elif(pooling=='MIN'):
            att_row = tf.reduce_min(y, 1)
            att_col = tf.reduce_min(y, 2)
        elif(pooling=='SUM'):
            att_row = tf.reduce_sum(y, 1)
            att_col = tf.reduce_sum(y, 2)
        elif(pooling=='MEAN'):
            att_row = tf.reduce_mean(y, 1)
            att_col = tf.reduce_mean(y, 2)

        att_row = tf.nn.softmax(att_row)
        att_col = tf.nn.softmax(att_col)
        _a2 = att_row
        _a1 = att_col

        att_col = tf.expand_dims(att_col, 2)
        att_row = tf.expand_dims(att_row, 2)

        # Weighted Representations
        final_a = att_col * input_a
        final_b = att_row * input_b

    y = tf.reshape(y, tf.stack([-1, a_len, b_len]))

    if(dropout is not None):
        final_a = tf.nn.dropout(final_a, dropout)
        final_b = tf.nn.dropout(final_b, dropout)

    if(readjust):
        final_a = tf.squeeze(final_a, 2)
        final_b = tf.squeeze(final_b, 2)

    return final_a, final_b, _a1, _a2, y
