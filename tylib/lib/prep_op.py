from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm
import numpy as np
from collections import Counter
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize

''' Utilities for prep scripts
'''

tweet_tokenizer = TweetTokenizer()

def tylib_tokenize(x, setting='split', lower=False):
    ''' All tokenizer in one. A convenient wrapper.

    Supported - 'split','nltk_tweet'

    TODO:'treebank','nltk_word'

    Args:
        x: `list`. list of words
        setting: `str` supports different tokenizers

    Returns:
        Tokenized output `list`

    '''
    if(setting=='split'):
        tokens = x.split(' ')
    elif(setting=='nltk_tweet'):
        tokens = tweet_tokenizer.tokenize(x)
    elif(setting=='nltk'):
        tokens = word_tokenize(x)
    if(lower):
        tokens = [x.lower() for x in tokens]
    return tokens

def word_to_index(word, word_index, unk_token=1):
    ''' Maps word to index.

    Arg:
        word: `str`. Word to be converted
        word_index: `dict`. dictionary of word-index mapping
        unk_token: `int`. token to label if OOV

    Returns:
        idx: `int` Index of word converted
    '''
    try:
        idx = word_index[word]
    except:
        idx = 1
    return idx

def sequence_to_indices(seq, word_index, unk_token=1):
    ''' Converts sequence of text to indices.

    Args:
        seq: `list`. list of list of words
        word_index: `dict`. dictionary of word-index mapping

    Returns:
        seq_idx: `list`. list of list of indices 

    '''
    # print(seq)
    seq_idx = [word_to_index(x, word_index, unk_token=unk_token) for x in seq]
    return seq_idx

def build_word_index(words, min_count=1, extra_words=['<pad>','<unk>'],
                        lower=True):
    ''' Builds Word Index

    Takes in all words in corpus and returns a word_index

    Args:
        words: `list` a list of words in the corpus. 
        min_count: `int` min number of freq to be included in index
        extra_words: `list` list of extra tokens such as pad or unk 
            tokens

    Returns:
        word_index `dict` built word index
        index_word `dict` inverrted word index

    '''

    # Build word counter

    # lowercase
    if(lower):
        words = [x.lower() for x in words]

    word_counter = Counter(words)

    # Select words above min Count
    words = [x[0] for x in word_counter.most_common() if x[1]>min_count]

    # Build Word Index with extra words
    word_index = {w:i+len(extra_words) for i, w in enumerate(words)}
    for i, w in enumerate(extra_words):
        word_index[w] = i

    # Builds inverse index
    index_word = {word:index for index, word in word_index.items()}

    print(index_word[0])
    print(index_word[1])
    print(index_word[2])

    return word_index, index_word


def build_embeddings(word_index, index_word, num_extra_words=2, 
                    emb_types=[('glove',300)],
                    base_dir='../', out_dir='./',
                    init_type='zero', init_val=0.01, normalize=False):
    ''' Builds compact glove embeddings for initializing

    Args:
        word_index: `dict` of words and indices
        index_word: `dict` inverted dictionary
        num_extra_words: `int` number of extra words (unk, pad) etc.
        emb_types:  `list` of tuples. ('glove,300'),('tweets',100)
            supports both tweets and glove (commoncrawl adaptations)
        base_dir: `str` file path of where to get the embeddings from 
        out_dir: `str` file path to where to store the embeddings
        init_type: `str` normal, unif or zero (how to init unk)
        init_val: `float` this acts as std for normal distribution and
            min/max val for uniform distribution.

    Returns:
        Saves the embedding to directory

    '''

    # Setup default paths
    print('Loading {} types of embeddings'.format(len(emb_types)))

    tweet_path = '{}/twitter_glove/'.format(base_dir)
    glove_path = '{}/glove_embeddings/'.format(base_dir)

    for _emb_type in emb_types:
        emb_type, dimensions = _emb_type[0], _emb_type[1]
        print(emb_type)
        print(dimensions)
        glove = {}
        if(emb_type=='tweets'):
            # dimensions = 100
            emb_path = 'glove.twitter.27B.{}d.txt'.format(dimensions)
            emb_path = tweet_path + emb_path
        elif(emb_type=='glove'):
            if(dimensions==300):
            # dimensions = 300
                emb_path = 'glove.840B.{}d.txt'.format(dimensions)
                emb_path = glove_path + emb_path
            else:
                emb_path = 'glove.6B.{}d.txt'.format(dimensions)
                emb_path = glove_path + emb_path

        print("Loading Glove Embeddings...")

        # Load word embeddings
        # Please place glove in correct place!
        with open(emb_path, 'r') as f:
            lines = f.readlines()
            for l in tqdm(lines):
                vec = l.split(' ')
                word = vec[0]
                vec = vec[1:]
                glove[word] = np.array(vec)

        print('glove size={}'.format(len(glove)))

        print("Finished making glove dictionary")
        matrix = []
        for i in range(num_extra_words):
            matrix.append(np.zeros((dimensions)).tolist())

        oov = 0 
        for i in tqdm(range(num_extra_words, len(word_index))):
            word = index_word[i]
            try:
                vec = glove[word]
                matrix.append(vec.tolist())
            except:
                if(init_type=='unif'):
                    # uniform distribution
                    vec = np.random.uniform(low=-init_val,high=init_val, 
                                size=(dimensions)).tolist()
                elif(init_type=='normal'):
                    # normal distribution
                    vec = np.random.normal(0, init_val, 
                                size=(dimensions)).tolist()
                elif(init_type=='zero'):
                    # zero vectors
                    vec = np.zeros((dimensions)).tolist()
                matrix.append(vec)
                oov +=1
            
        matrix = np.stack(matrix)
        matrix = np.reshape(matrix,(len(word_index), dimensions))
        matrix = matrix.astype(np.float)

        print(matrix.shape)

        # if(normalize):
        #     norm = np.linalg.norm(matrix, axis=1).reshape((-1, 1))
        #     matrix = matrix / norm

        print(matrix.shape)
        print(len(word_index))
        print("oov={}".format(oov))

        print("Finished building and writing...")

        # env['glove'] = matrix
        np.save('{}/emb_{}_{}.npy'.format(out_dir, emb_type, 
                                        dimensions), matrix)
        print("Saved to file..")