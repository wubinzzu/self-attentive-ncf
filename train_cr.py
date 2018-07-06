from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
import csv
import argparse
from keras.preprocessing import sequence
from datetime import datetime
import numpy as np
import random
np.random.seed(1337)
random.seed(1337)
import os
from tqdm import tqdm
from utilities import *
from metrics import *
import time
import tensorflow as tf
import sys
from sklearn.utils import shuffle
from collections import Counter
import cPickle as pickle
from keras.utils import np_utils
import visdom
import string
import re
import math
import operator
from utilities import *
from collections import defaultdict
import sys
from nltk.corpus import stopwords
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from scipy.stats import spearmanr

# from tf_models.model import Model
from tf_models.rec_model import RecModel
from tylib.exp.experiment import Experiment
from tylib.exp.exp_ops import *
from tylib.exp.tuning import *
from parser import *
from metrics import *

reload(sys)
sys.setdefaultencoding('UTF8')

def batchify(data, i, bsz, max_sample):
    start = int(i * bsz)
    end = int(i * bsz) + bsz
    if(end>max_sample):
        end = max_sample
    data = data[start:end]
    return data

class CRExperiment(Experiment):
    """ Main experiment class for running Collaborative Ranking Experiment

    Note: We record dev-test scores at every eval epoch and then print
    out the best and also the max test scores based on this alignment.
    """

    def __init__(self, inject_params=None):
        print("Starting [Rank Rec] Experiment")
        super(CRExperiment, self).__init__()
        self.uuid = datetime.now().strftime("%d:%m:%H:%M:%S")
        self.parser = build_parser()

        self.no_text_mode = True
        self.args = self.parser.parse_args()
        # self.max_val, self.min_val, self.args.data_link = get_rec_config(
        #                                         self.args.dataset)

        self.show_metrics = ['nDCG' ,'HR10','ACC']
        self.eval_primary = 'nDCG'

        print("Setting up environment..")
        # if(self.args.data_link!=""):
        #     print("[Starting Data Link..]")
        #     data_path = '{}/datasets/{}/env.gz'.format(
        #                                     self.args.data_link,
        #                                     self.args.dataset)
        # else:
        data_path  = './datasets/{}/env.gz'.format(
            self.args.dataset)
        self.env = dictFromFileUnicode(data_path)
        self.model_name = self.args.rnn_type
        self._setup()
        self._load_sets()

        if(self.no_text_mode):
            try:
                self.num_users = len(self.env['user_index'])
                self.num_items = len(self.env['item_index'])
            except:
                self.num_users = self.env['num_users']
                self.num_items = self.env['num_items']
            # Create one user extra incase of some datasets start from 0
            self.num_users +=1
            self.num_items +=1
            print("Users={} Items={}".format(self.num_users, self.num_items))
            self.mdl = RecModel(self.num_users, self.num_items,
                             self.args)

        self._print_model_stats()
        self.hyp_str = self.model_name + '_' + self.uuid
        self._setup_tf(load_embeddings=not self.no_text_mode)
        if(self.args.pretrained==1):
            with self.mdl.graph.as_default():
                self._load_embeddings()

    def _prepare_base_set(self, data):
        print("Preparing Base Set")
        data = [x for x in data if x[2]>0]
        user = [x[0] for x in data]
        item = [x[1] for x in data]
        labels = [x[2] for x in data]
        self.mdl.register_index_map(0, 'q1_inputs')
        self.mdl.register_index_map(1, 'q2_inputs')
        self.mdl.register_index_map(2, 'q3_inputs')
        output = [user, item]
        output = zip(*output)
        return output

    def _load_embeddings(self):
        ''' Loads pre-trained embeddings for recsys
        Not used in experiments
        '''
        if(self.args.pretrained == 1):
            user = np.load('./datasets/{}/user_{}_pretrain.npy'.format(
                self.args.dataset, self.args.emb_size))
            item = np.load('./datasets/{}/item_{}_pretrain.npy'.format(
                self.args.dataset, self.args.emb_size))
            print("loaded pretrained embeddings")
            feed_dict = {self.mdl.user_pretrain: user,
                            self.mdl.item_pretrain:item}
            self.sess.run([self.mdl.user_embed_init, self.mdl.item_embed_init],
                            feed_dict=feed_dict)
            print("Loading embedding success..")

    def prepare_set(self, data):
        return self._prepare_base_set(data)

    def _load_sets(self):

        self.train_set = self.env['train']
        self.dev_set = self.env['dev']
        if(self.args.dev==0):
            self.train_set += self.dev_set
        self.test_set = self.env['test']

        # Build all ratings
        print("Building Fast Access Rating Dict...")
        all_set = self.train_set + self.dev_set + self.test_set
        self.all_ratings = {str(tuple([t[0],t[1]])):t[2] for t in all_set}

        self.user_dict = defaultdict(list)
        self.item_dict = defaultdict(list)

        print("Building User Negative Dict...")
        # get negative samples for each user for eval.
        self.user_neg = self.env['user_negative']
        try:
            self.user_neg = {key:[int(x) for x in val] \
                        for key, val in self.user_neg.items()}
        except:
            self.user_neg = {key:[int(x) for x in val.split(' ')] \
                        for key, val in self.user_neg.items()}

        self.write_to_file("Train={} Dev={} Test={}".format(
                len(self.train_set),len(self.dev_set),len(self.test_set)))


    def get_predictions(self, batch, test_batch_size=10000):
        total_data = len(batch)
        num_batches = int(len(batch) / test_batch_size) + 1
        prediction = []
        for i in range(0, num_batches):
            batch = batchify(batch, i, self.args.batch_size,
                            max_sample=len(batch))
            if(len(batch)==0):
                continue
            feed_dict = self.mdl.get_feed_dict(batch, mode='testing')
            pred = self.sess.run([self.mdl.predict_op], feed_dict)
            pred = [x[0] for x in pred[0]]
            prediction += pred
        assert(len(batch)==len(prediction))
        return prediction

    def evaluate(self, data, bsz, epoch, name="", set_type="", k=10):
        """ Calculate recsys metrics for CR
        To enable other vals of @k, please update register_eval_score
        """
        ranks = []
        all_ndcg = []
        print("Evaluating Recommendation Metrics...")
        for p in tqdm(data):
            u = p[0]
            candidates = self.user_neg[str(u)]
            batch = [p]
            batch += [[u,x] for x in candidates]
            pred = self.get_predictions(batch)
            sorted_pred = np.argsort(pred)[::-1]
            relevance = [0 for i in range(len(candidates))]
            rank = np.where(sorted_pred==0)[0] + 1
            if(rank<=k):
                relevance[int(rank)-1] = 1.0
            n = ndcg_at_k(relevance, k)
            all_ndcg.append(n)
            ranks.append(rank)
        mean_rank = np.mean(ranks)
        ndcg = np.mean(all_ndcg)
        mrr = np.mean([1/x for x in ranks])
        hits10 = len([x for x in ranks if x<=10]) / len(ranks)
        hits1 = len([x for x in ranks if x==1]) / len(ranks)
        hits3 = len([x for x in ranks if x<=3]) / len(ranks)
        self._register_eval_score(epoch, set_type, 'nDCG', ndcg)
        self._register_eval_score(epoch, set_type, 'HR10',hits10)
        self._register_eval_score(epoch, set_type, 'ACC',hits1)

    def add_negative_samples(self, batch):
        """ Corrupt samples
        """
        new_data = []
        for b in tqdm(batch):
            n = []
            while(len(n)<self.args.num_neg):
                random_item = random.randint(0, self.num_items-1)
                # random_user = random.randint(0, self.num_users)
                tup_str = str(tuple([b[0], random_item]))
                if(tup_str in self.all_ratings):
                    continue
                n.append([b[0],b[1],random_item])
            new_data += n

        return new_data

    def train(self):
        """ Main training loop
        """
        scores = []
        best_score = -1
        best_dev = -1
        best_epoch = -1
        counter = 0
        epoch_scores = {}
        self.eval_list = []
        data = self.prepare_set(self.train_set)
        self.test_set = self.prepare_set(self.test_set)
        self.dev_set = self.prepare_set(self.dev_set)
        print("Positive Samples={}".format(len(data)))

        for epoch in range(1, self.args.epochs+1):
            attention = []
            av_list = []
            all_att_dict = {}
            pos_val, neg_val = [],[]
            t0 = time.clock()
            self.write_to_file("===============================")

            # Get neg samples
            train_data = self.add_negative_samples(data)

            losses = []
            random.shuffle(data)
            # num_batches = int(len(train_data) / self.args.batch_size)

            if(self.args.num_batch>0):
                num_batches = self.args.num_batch
                batch_size = int(len(train_data)/num_batches)
            else:
                batch_size = self.args.batch_size
                num_batches = int(len(train_data) / self.args.batch_size)

            norms = []
            all_acc = 0
            train_op = self.mdl.train_op
            l2_reg = None
            lr = None

            for i in tqdm(range(0, num_batches+1)):
                batch = batchify(train_data, i, batch_size,
                                max_sample=len(train_data))

                if(len(batch)==0):
                    continue
                feed_dict = self.mdl.get_feed_dict(batch, l2_reg=l2_reg,
                                                    lr=lr)


                if(self.args.constraint==1):
                    # if constrain embedding <=1
                    # By default this is off
                    _, loss, summary,_,_ = self.sess.run(
                                                [train_op,
                                                self.mdl.cost,
                                                self.mdl.merged_summary_op,
                                                self.mdl.clip_user,
                                                self.mdl.clip_item,
                                                ],
                                                feed_dict)

                else:
                    _, loss, summary, afm, av = self.sess.run(
                                                [train_op,
                                                self.mdl.cost,
                                                self.mdl.merged_summary_op,
                                                self.mdl.afm,
                                                self.mdl.av],
                                                feed_dict)
                    # This is to output SA-NCF's matrices for inspection
                    try:
                        av = av.tolist()
                        av_list += av
                    except:
                        pass
                    if(i==0 and epoch % 5==0 and self.args.save_att):
                        fp = './plots/sancf_plots/{}_{}__{}_afm.npy'.format(
                                    self.args.dataset, self.args.rnn_type, epoch)
                        np.save(fp, afm)
                all_acc += (loss * len(batch))
                if(self.args.tensorboard):
                    self.train_writer.add_summary(summary, counter)
                counter +=1
                losses.append(loss)
            t1 = time.clock()
            # if(len(av_list)>0):
            #     cnt_av = Counter(av_list)
            #     cnt_av = sorted(cnt_av.items(), key=operator.itemgetter(0))
            #     cnt_av = [x[1] for x in cnt_av]
            #     print(cnt_av)
            #     dfp = './plots/sancf_plots/dist/{}_{}.txt'.format(
            #             self.args.dataset, self.uuid)
            #     with open(dfp, 'a+') as f:
            #         f.write(' '.join([str(x) for x in cnt_av]))
            #         f.write('\n')

            self.write_to_file("[{}] [Epoch {}] [{}] loss={}".format(
                                self.args.dataset, epoch, self.model_name,
                                np.mean(losses)))
            self.write_to_file("GPU={} | d={} | num_layers={}".format(
                                            self.args.gpu,
                                            self.args.emb_size,
                                            self.args.num_dense))

            if(epoch % self.args.eval==0):
                self.evaluate(self.dev_set,
                    self.args.batch_size, epoch, set_type='Dev')
                self._show_metrics(epoch, self.eval_dev,
                                    self.show_metrics,
                                        name='Dev')
                self.evaluate(self.test_set,
                    self.args.batch_size, epoch, set_type='Test')
                self._show_metrics(epoch, self.eval_test,
                                    self.show_metrics,
                                        name='Test')
                stop, max_e, best_epoch = self._select_test_by_dev(
                                                epoch,
                                                self.eval_dev,
                                                self.eval_test,
                                                lower_is_better=False)
                if(best_epoch==epoch and self.args.save_embed):
                    # option to pretrain and save embeddings
                    with self.sess.as_default():
                        user_embed = self.mdl.user_embed.eval()
                        item_embed = self.mdl.item_embed.eval()
                        fp_user = './datasets/{}/user_pretrain.npy'.format(
                                                        self.args.dataset)
                        fp_item = './datasets/{}/item_pretrain.npy'.format(
                                                        self.args.dataset)
                        np.save(fp_user, user_embed)
                        np.save(fp_item, item_embed)
                        print("Saved embeddings to file..")
                if(epoch-best_epoch>self.args.early_stop and self.args.early_stop >0):
                    print("Ended at early stop")
                    sys.exit(0)

if __name__ == '__main__':
    exp = CRExperiment(inject_params=None)
    exp.train()
    print("End of code!")
