from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

def model_stats():
    ''' Returns total number of trainable parameters in model
    '''
    print("======================================================")
    print("List of all Trainable Variables")
    tvars = tf.trainable_variables()
    all_params = []
    for idx, v in enumerate(tvars):
        try:
            print(" var {:3}: {:15} {}".format(idx,
                                        str(v.get_shape()),
                                        v.name))
            num_params = 1
            param_list = v.get_shape().as_list()
            if(len(param_list)>1):
                for p in param_list:
                    if(p>0):
                        num_params = num_params * int(p)
            else:
                all_params.append(param_list[0])
            all_params.append(num_params)
        except:
            pass
    num_params = np.sum(all_params)
    print("Total number of trainable parameters {}".format(num_params))
