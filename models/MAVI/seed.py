#!/usr/bin/env python
'''
File        :   seed.py
Author      :   Akira Nair
Contact     :   akira_nair@brown.edu
Description :   Sets seed
'''
import os
import tensorflow as tf
import numpy as np
import random

def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
