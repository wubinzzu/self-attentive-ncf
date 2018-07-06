#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
import gzip
import json
from tqdm import tqdm
import random
from collections import Counter
import operator
import timeit
import time

import datetime
from keras.preprocessing import sequence

from keras.layers.wrappers import TimeDistributed
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization

from .utilities import *
from keras.utils import np_utils
import numpy as np

from tylib.lib.att_op import *
from tylib.lib.compose_op import *
from tylib.lib.rec_op import *


class RecModel:
    ''' Neural Recommendation Model
    '''
    def __init__(self, num_user, num_item, args, mode='REC'):
        self.num_user = num_user
        self.num_item = num_item
        self.graph = tf.Graph()
        self.args = args
        self.imap = {}
        self.inspect_op = []
        self.av = []
        self.mode=mode
        self.afm = []
        print('Creating Model in [{}] mode'.format(self.mode))
        if(self.args.init_type=='xavier'):
            # this is the default setting
            self.initializer = tf.contrib.layers.xavier_initializer()
        elif(self.args.init_type=='normal'):
            self.initializer = tf.random_normal_initializer(0.0,
                                                    self.args.init)
        elif(self.args.init_type=='tnormal'):
            self.initializer = tf.truncated_normal_initializer(0.0,
                                                    self.args.init)

        elif(self.args.init_type=='uniform'):
            self.initializer = tf.random_uniform_initializer(
                                            maxval=self.args.init,
                                            minval=-self.args.init)

        self.temp = []
        self.att1, self.att2 = [],[]
        self.build_graph()

    def _get_pair_feed_dict(self, data,
                mode='training', lr=None, l2_reg=None):
        data = zip(*data)
        labels = data[-1]

        if(lr is None):
            lr = self.args.learn_rate

        if(l2_reg is None):
            l2_reg = self.args.l2_reg

        feed_dict = {
            self.q1_inputs:data[self.imap['q1_inputs']],
            self.q2_inputs:data[self.imap['q2_inputs']],
            self.learn_rate:lr,
            self.l2_reg:l2_reg,
            self.dropout:self.args.dropout,
            self.emb_dropout:self.args.emb_dropout
        }
        if(mode=='training'):
            feed_dict[self.q3_inputs] = data[self.imap['q3_inputs']]
        if(mode!='training'):
            feed_dict[self.dropout] = 1.0
            feed_dict[self.emb_dropout] = 1.0
        return feed_dict

    def get_feed_dict(self, data,
                    mode='training', lr=None, l2_reg=None):
        if("RANK" in self.args.rnn_type):
            return self._get_pair_feed_dict(data, mode=mode,
                                lr=lr, l2_reg=l2_reg)
        else:
            return self._get_point_feed_dict(data, mode=mode, lr=lr)

    def _get_point_feed_dict(self, data, mode='training', lr=None):
        data = zip(*data)
        labels = data[-1]
        if(lr is None):
            lr = self.args.learn_rate
        feed_dict = {
            self.q1_inputs:data[self.imap['q1_inputs']],
            self.q2_inputs:data[self.imap['q2_inputs']],
            self.learn_rate:lr,
            self.dropout:self.args.dropout,
            self.emb_dropout:self.args.emb_dropout,
            self.sig_labels:labels
        }
        if(mode!='training'):
            feed_dict[self.dropout] = 1.0
            feed_dict[self.emb_dropout] = 1.0
        return feed_dict

    def register_index_map(self, idx, target):
        self.imap[target] = idx

    def learn_repr(self, q1_output, q2_output,
                    reuse=None, extract_embed=False,
                    side='', z1_output=None, z2_output=None):
        """ Given user, item emb, learn joint representation.
        """
        att1, att2 = None, None

        def fm_model(q1_output, q2_output):
            input_vec = tf.concat([q1_output, q2_output], 1)
            input_vec = tf.nn.dropout(input_vec, self.dropout)
            output, _ = build_fm(input_vec, k=self.args.factor,
                                reuse=reuse,
                                name='',
                                initializer=self.initializer,
                                reshape=False)
            return output

        dim = q1_output.get_shape().as_list()[1]
        with tf.variable_scope('rec_out', reuse=reuse) as scope:
            if('GMF' in self.args.rnn_type):
                output = mf(q1_output, q2_output, self.dropout,
                            generalized=True)
            if('MF' in self.args.rnn_type):
                output = mf(q1_output, q2_output, self.dropout,
                            generalized=False)
            if('FM' in self.args.rnn_type):
                output = fm_model(q1_output, q2_output)
            if('MLP' in self.args.rnn_type):
                # 3 Layer MLP
                if('TOWER' in self.args.rnn_type):
                    tower = True
                else:
                    tower = False
                output, afm = mlp(q1_output, q2_output,
                            self.dropout,
                            reuse=reuse,
                            initializer=self.initializer,
                            hdim=self.args.emb_size * 2,
                            num_dense=self.args.num_dense,
                            pyramid=tower)
                # if(side=='POS'):
                #     # This is just for visualisation
                #     self.afm = afm
                #     # if(len(afm)==0):
                #     #     self.av = []
                #     # else:
                #         self.av = tf.argmax(tf.reduce_sum(tf.nn.softmax(afm),1), 1)
                #         self.av = tf.reshape(self.av, [-1])
            if('HIGH' in self.args.rnn_type):
                output = highway_networks(q1_output, q2_output,
                            self.dropout,
                            reuse=reuse,
                            initializer=self.initializer,
                            hdim=self.args.emb_size * 2,
                            num_dense=self.args.num_dense,
                            pyramid=False)
            if('SANCF' in self.args.rnn_type):
                output, afm = self_attentive_ncf(q1_output, q2_output,
                                self.dropout,
                                reuse=reuse,
                                num_dense=self.args.num_dense,
                                initializer=self.initializer,
                                hdim=self.args.emb_size * 2,
                                pyramid=False)
                if(side=='POS'):
                    self.afm = afm
                    self.av = tf.argmax(tf.reduce_sum(tf.nn.softmax(afm),1), 1)
                    self.av = tf.reshape(self.av, [-1])
            if('JRL' in self.args.rnn_type):
                if('TOWER' in self.args.rnn_type):
                    tower = True
                else:
                    tower = False
                output = jrl(q1_output, q2_output,
                            self.dropout,
                            reuse=reuse,
                            initializer=self.initializer,
                            hdim=self.args.emb_size,
                            num_dense=self.args.num_dense,
                            pyramid=tower)
            if("NEUMF" in self.args.rnn_type):
                output1 = mlp(q1_output, q2_output,
                            self.dropout,
                            reuse=reuse,
                            num_dense=self.args.num_dense,
                            initializer=self.initializer,
                            hdim=self.args.emb_size * 2,
                            last_layer=False)
                # Use different embeddings
                output2 = z1_output * z2_output
                output = tf.concat([output1, output2], 1)
                output = linear(output, 1, self.initializer,
                    name='final_layer', reuse=reuse, bias=False)
        if('SIG' in self.args.rnn_type):
            # this is generally not used in our BPR loss
            output = tf.nn.sigmoid(output)
        representation = output
        return output, representation, att1, att2

    def prepare_inputs(self):
        """ Prepares Input
        """
        q1_embed =  tf.nn.embedding_lookup(self.user_embed,
                                            self.q1_inputs)
        q2_embed =  tf.nn.embedding_lookup(self.item_embed,
                                            self.q2_inputs)
        q3_embed = tf.nn.embedding_lookup(self.item_embed,
                                            self.q3_inputs)

        if('NEUMF' in self.args.rnn_type):
            # Draw dual embeddings
            self.z1_embed =  tf.nn.embedding_lookup(self.user_embed2,
                                            self.q1_inputs)
            self.z2_embed =  tf.nn.embedding_lookup(self.item_embed2,
                                                self.q2_inputs)
            self.z3_embed = tf.nn.embedding_lookup(self.item_embed2,
                                                self.q3_inputs)
        else:
            self.z1_embed, self.z2_embed, self.z3_embed = None, None, None

        if(self.args.all_dropout):
            q1_embed = tf.nn.dropout(q1_embed, self.emb_dropout)
            q2_embed = tf.nn.dropout(q2_embed, self.emb_dropout)
            q3_embed = tf.nn.dropout(q3_embed, self.emb_dropout)

        self.q1_embed = q1_embed
        self.q2_embed = q2_embed
        self.q3_embed = q3_embed

    def build_graph(self):
        ''' Builds Computational Graph
        '''
        with self.graph.as_default():
            self.is_train = tf.get_variable("is_train",
                                            shape=[],
                                            dtype=tf.bool,
                                            trainable=False)
            self.true = tf.constant(True, dtype=tf.bool)
            self.false = tf.constant(False, dtype=tf.bool)
            with tf.name_scope('q1_input'):
                self.q1_inputs = tf.placeholder(tf.int32, shape=[None],
                                                    name='q1_inputs')
            with tf.name_scope('q2_input'):
                self.q2_inputs = tf.placeholder(tf.int32, shape=[None],
                                                    name='q2_inputs')
            with tf.name_scope('q3_input'):
                self.q3_inputs = tf.placeholder(tf.int32, shape=[None],
                                                    name='q3_inputs')
            with tf.name_scope('dropout'):
                self.dropout = tf.placeholder(tf.float32,
                                                name='dropout')
                self.emb_dropout = tf.placeholder(tf.float32,
                                                name='emb_dropout')
            with tf.name_scope('learn_rate'):
                self.learn_rate = tf.placeholder(tf.float32,
                                        name='learn_rate')
            with tf.name_scope('l2_reg'):
                self.l2_reg = tf.placeholder(tf.float32,
                                        name='l2_reg')
            with tf.name_scope("sig_labels"):
                # sigmoid cross entropy
                self.sig_labels = tf.placeholder(tf.float32,
                                                shape=[None],
                                                name='sigmoid_labels')
               #  self.sig_target = tf.expand_dims(self.sig_labels, 1)
            with tf.name_scope('embed'):
                self.user_embed = tf.get_variable('user_emb',
                                                [self.num_user,
                                                self.args.emb_size],
                                                initializer=self.initializer,
                                                )
                self.item_embed = tf.get_variable('item_emb',
                                                [self.num_item,
                                                self.args.emb_size],
                                                initializer=self.initializer,
                                                )
                if('NEUMF' in self.args.rnn_type):
                    self.user_embed2 = tf.get_variable('user_emb2',
                                                [self.num_user,
                                                self.args.emb_size],
                                                initializer=self.initializer,
                                                )
                    self.item_embed2 = tf.get_variable('item_emb2',
                                                [self.num_item,
                                                self.args.emb_size],
                                                initializer=self.initializer,
                                                )
            if(self.args.pretrained==1):
                self.user_pretrain = tf.placeholder(tf.float32,
                            [self.num_user, self.args.emb_size])
                self.item_pretrain = tf.placeholder(tf.float32,
                            [self.num_item, self.args.emb_size])
                self.user_embed_init = self.user_embed.assign(self.user_pretrain)
                self.item_embed_init = self.item_embed.assign(self.item_pretrain)

            self.batch_size = tf.shape(self.q1_inputs)[0]
            self.prepare_inputs()
            self.output_pos2, self.output_neg2 = None,None
            print("Learning Representations")
            repr_fun = self.learn_repr

            self.output_pos, _, self.att1, self.att2 = repr_fun(
                                        self.q1_embed, self.q2_embed,
                                        reuse=None,
                                        extract_embed=True,
                                        side='POS',
                                        z1_output=self.z1_embed,
                                        z2_output=self.z2_embed
                                        )

            if("RANK" in self.args.rnn_type):
                self.output_neg,_,_, _ = repr_fun(self.q1_embed,
                                            self.q3_embed,
                                             reuse=True,
                                             side='NEG',
                                             z1_output=self.z1_embed,
                                             z2_output=self.z3_embed
                                             )

            self.cost2 = None
            with tf.name_scope("train"):
                with tf.name_scope("cost_function"):
                    if('HINGE' in self.args.rnn_type):
                        self.hinge_loss = tf.maximum(0.0,(
                        self.args.margin - self.output_pos + self.output_neg))
                        self.cost = tf.reduce_sum(self.hinge_loss)
                    else:
                        # this variation divides the loss by the batch_size
                        self.cost = tf.reduce_mean(
                                    -tf.log(tf.nn.sigmoid(
                                        self.output_pos-self.output_neg)))

                    with tf.name_scope('regularization'):
                        if(self.args.l2_reg>0):
                            print("Adding L2 Regularization..")
                            vars = tf.trainable_variables()
                            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars \
                                                if 'bias' not in v.name ])
                            lossL2 *= self.l2_reg
                            self.cost += lossL2

                    tf.summary.scalar("cost_function", self.cost)
                global_step = tf.Variable(0, trainable=False)

                if(self.args.decay_lr>0 and self.args.decay_epoch>0):
                    # decay is optional, and default set to none
                    decay_epoch = self.args.decay_epoch
                    lr = tf.train.exponential_decay(self.args.learn_rate,
                                  global_step,
                                  decay_epoch * self.args.batch_size,
                                   self.args.decay_lr, staircase=True)
                else:
                    lr = self.args.learn_rate

                control_deps = []
                with tf.name_scope('optimizer'):
                    # support the various optimizers
                    if(self.args.opt=='SGD'):
                        self.opt = tf.train.GradientDescentOptimizer(
                            learning_rate=lr)
                    elif(self.args.opt=='Adam'):
                        self.opt = tf.train.AdamOptimizer(
                                        learning_rate=lr)
                    elif(self.args.opt=='Adadelta'):
                        self.opt = tf.train.AdadeltaOptimizer(
                                        learning_rate=lr,
                                        rho=0.9)
                    elif(self.args.opt=='Adagrad'):
                        self.opt = tf.train.AdagradOptimizer(
                                        learning_rate=lr)
                    elif(self.args.opt=='RMS'):
                        self.opt = tf.train.RMSPropOptimizer(
                                    learning_rate=lr)
                    elif(self.args.opt=='Moment'):
                        self.opt = tf.train.MomentumOptimizer(lr, 0.9)

                    tvars = tf.trainable_variables()
                    def _none_to_zero(grads, var_list):
                        return [grad if grad is not None else tf.zeros_like(var)
                              for var, grad in zip(var_list, grads)]
                    if(self.args.clip_norm>0):
                        grads, _ = tf.clip_by_global_norm(
                                        tf.gradients(self.cost, tvars),
                                        self.args.clip_norm)
                        with tf.name_scope('gradients'):
                            gradients = self.opt.compute_gradients(self.cost)
                            def ClipIfNotNone(grad):
                                if grad is None:
                                    return grad
                                grad = tf.clip_by_value(grad, -10, 10, name=None)
                                return tf.clip_by_norm(grad, self.args.clip_norm)
                            if(self.args.clip_norm>0):
                                clip_g = [(ClipIfNotNone(grad), var) for grad, var in gradients]
                            else:
                                clip_g = [(grad,var) for grad,var in gradients]

                        with tf.control_dependencies(control_deps):
                            self.train_op = self.opt.apply_gradients(clip_g,
                                                global_step=global_step)
                    else:
                        with tf.control_dependencies(control_deps):
                            self.train_op = self.opt.minimize(self.cost)

                self.grads = _none_to_zero(tf.gradients(self.cost,tvars), tvars)
                self.merged_summary_op = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)

                print(self.output_pos)
                clip_user = tf.clip_by_norm(self.user_embed, 1.0, axes=[1])
                clip_item = tf.clip_by_norm(self.item_embed, 1.0, axes=[1])
                self.clip_user = tf.assign(self.user_embed, clip_user)
                self.clip_item = tf.assign(self.item_embed, clip_item)
                 # for Inference
                self.predict_op = self.output_pos
                if(self.output_pos2 is not None):
                    self.predict_op2 = self.output_pos2
                if('SOFT' in self.args.rnn_type):
                    # Pointwise model. not used here.
                    if('POINT' in self.args.rnn_type):
                        predict_neg = 1 - self.predict_op
                        self.predict_op = tf.concat([predict_neg,
                                          self.predict_op], 1)
                    else:
                        self.predict_op = tf.nn.softmax(self.output_pos)
                    self.predictions = tf.argmax(self.predict_op, 1)
                    self.correct_prediction = tf.equal(tf.argmax(self.predict_op, 1),
                                                     tf.argmax(self.soft_labels, 1))
                    self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,
                                                     "float"))
                else:
                    self.predict_op = tf.nn.sigmoid(self.output_pos)
                    self.predictions = self.predict_op
