#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 13:08:55 2017

@author: admin
"""

import numpy as np


b=np.load('data.npy')
print b[1,0,:]
print (len(b))