#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 13:08:55 2017

@author: admin
"""

import numpy as np
import os
from collections import OrderedDict
import json

if os.path.exists('res'+'ults.json')==True:
    print True
    with open('res'+'ults.json') as f:
   		dic = json.load(f)

b=np.load('test_data.npy')

print (len(b))