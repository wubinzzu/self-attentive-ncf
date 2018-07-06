from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

def build_parser():
    """ Arguments and configurations
    """
    parser = argparse.ArgumentParser()
    ps = parser.add_argument
    ps("--dataset", dest="dataset", type=str,  default='YahooMusic', help="Dataset")
    ps("--rnn_type", dest="rnn_type", type=str, metavar='<str>',
            default='RANK_BREC_SA', help="Model name")
    ps("--opt", dest="opt", type=str, metavar='<str>', default='Adam',
       help="Optimization algorithm)")
    ps("--emb_size", dest="emb_size", type=int, metavar='<int>',
       default=64, help="Embeddings dimension (default=300)")
    ps("--batch-size", dest="batch_size", type=int, metavar='<int>',
       default=512, help="Batch size (default=512)")
    ps("--num_batch", dest="num_batch", type=int, metavar='<int>',
       default=0, help="num batch (set 0 if batch_size>0)")
    ps("--allow_growth", dest="allow_growth", type=int, metavar='<int>',
      default=0, help="Allow Growth")
    ps("--patience", dest="patience", type=int, metavar='<int>',
       default=3, help="Patience for halving LR (not used)")
    ps("--dev_lr", dest='dev_lr', type=int,
       metavar='<int>', default=0, help="Dev Learning Rate (not used)")
    ps("--decay_epoch", dest="decay_epoch", type=int,
       metavar='<int>', default=0, help="Decay everywhere n epochs")
    ps("--num_dense", dest="num_dense", type=int,
       metavar='<int>', default=20, help="Number of dense layers")
    ps("--clip_output", dest="clip_output", type=int, metavar='<int>',
        default=0, help="clip output")
    ps("--normalize_embed", dest="normalize_embed", type=int, metavar='<int>',
      default=0, help="Normalize pretrained embeddings")
    ps("--factor", dest="factor", type=int, metavar='<int>',
       default=10, help="For factorization factors (not used, only for FM)")
    ps("--dropout", dest="dropout", type=float, metavar='<float>',
        default=1.0, help="The dropout probability.")
    ps("--rnn_dropout", dest="rnn_dropout", type=float, metavar='<float>',
        default=0.8, help="The dropout probability.")
    ps("--emb_dropout", dest="emb_dropout", type=float, metavar='<float>',
        default=0.8, help="The dropout probability.")
    ps("--pretrained", dest="pretrained", type=int, metavar='<int>',
       default=0, help="Whether to use pretrained or not")
    ps("--epochs", dest="epochs", type=int, metavar='<int>',
       default=500, help="Number of epochs (default=50)")
    ps('--gpu', dest='gpu', type=int, metavar='<int>',
       default=0, help="Specify which GPU to use (default=0)")
    ps("--hdim", dest='hdim', type=int, metavar='<int>',
       default=64, help="Hidden layer size (not used")
    ps("--lr", dest='learn_rate', type=float,
       metavar='<float>', default=1e-3, help="Learning Rate")
    ps("--margin", dest='margin', type=float,
       metavar='<float>', default=0.2, help="Margin (only for hinge loss)")
    ps("--clip_norm", dest='clip_norm', type=int,
       metavar='<int>', default=0, help="Clip Norm value")
    ps("--clip_embed", dest='clip_embed', type=int,
       metavar='<int>', default=0, help="Clip Norm value")
    ps('--l2_reg', dest='l2_reg', type=float, metavar='<float>',
       default=1e-8, help='L2 regularization, default=4E-6')
    ps('--eval', dest='eval', type=int, metavar='<int>',
       default=5, help='Epoch to evaluate results')
    ps('--log', dest='log', type=int, metavar='<int>',
       default=1, help='1 to output to file and 0 otherwise')
    ps('--dev', dest='dev', type=int, metavar='<int>',
       default=1, help='1 for development set 0 to train-all')
    ps('--seed', dest='seed', type=int, default=1337, help='random seed')
    ps('--tensorboard', action='store_true', help='To use tensorboard or not')
    ps('--early_stop',  dest='early_stop', type=int,
       metavar='<int>', default=25, help='early stopping')
    ps('--test_bsz', dest='test_bsz', type=int,
       metavar='<int>', default=4, help='Multiplier for eval bsz')
    ps('--data_link', dest='data_link', type=str, default='',
        help='data link')
    ps('--att_type', dest='att_type', type=str, default='SOFT',
        help='attention type (not used)')
    ps('--all_dropout', action='store_true',
       default=False, help='dropout on embedding')
    ps("--num_neg", dest="num_neg", type=int, metavar='<int>',
       default=2, help="Number of negative samples")
    ps('--constraint',  type=int, metavar='<int>',
       default=0, help='Constraint embeddings to unit ball')
    ps('--save_embed', action='store_true', default=False,
       help='Save embeddings for visualisation')
    ps('--save_att', action='store_true', default=False,
       help='Save att for visualisation')
    ps('--default_len', dest="default_len", type=int, metavar='<int>',
       default=1, help="Use default len or not")
    ps('--sort_batch', dest="sort_batch", type=int, metavar='<int>',
       default=0, help="To use sort-batch optimization or not")
    ps("--init", dest="init", type=float,
       metavar='<float>', default=0.01, help="Init Params")
    ps("--show_att", dest="show_att", type=int,
      metavar='<int>', default=0, help="Display Attention")
    ps("--show_affinity", dest="show_affinity", type=int,
    metavar='<int>', default=0, help="Display Affinity Matrix")
    ps("--fuse_kernel", dest="fuse_kernel", type=int,
    metavar='<int>', default=1, help="Use fused kernel ops")
    ps("--init_type", dest="init_type", type=str,
       metavar='<str>', default='xavier', help="Init Type")
    ps("--rnn_init_type", dest="rnn_init_type", type=str,
       metavar='<str>', default='same', help="Init Type")
    ps("--init_emb", dest="init_emb", type=float,
       metavar='<float>', default=0.01, help="Init Embeddings")
    ps("--decay_lr", dest="decay_lr", type=float,
       metavar='<float>', default=0, help="Decay Learning Rate")
    ps("--decay_steps", dest="decay_steps", type=float,
       metavar='<float>', default=0, help="Decay Steps (manual)")
    ps("--decay_stairs", dest="decay_stairs", type=float,
       metavar='<float>', default=1, help="To use staircase or not")
    ps('--emb_type', dest='emb_type', type=str,
       default='glove', help='embedding type')
    ps('--log_dir', dest='log_dir', type=str,
       default='logs', help='log directory')
    return parser
