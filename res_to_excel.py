#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 15:10:48 2017

@author: admin
"""

import csv
import json

with open('results-1.json') as f:
   		dic = json.load(f)


with open('output.csv', 'wb') as output:
    writer = csv.writer(output)
    for key, value in dic.iteritems():
        writer.writerow([key, value]) 