#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 12:57:47 2017

@author: ayanava
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

iters = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

acc_8 = [48.333333333333336, 50.0, 90.83333333333333, 100.0, 68.33333333333333, 95.83333333333334, 97.5, 88.33333333333333,
         75.83333333333333, 76.66666666666667]

acc_10 = [47.5, 39.166666666666664, 51.66666666666667, 61.66666666666667, 98.33333333333333, 100.0, 72.5, 80.0, 100.0, 100.0]

fig, ax = plt.subplots(1,1)
axis_font = {'fontname':'Arial', 'size':'50'}

p1, =plt.plot(  iters, acc_8, label="d", linewidth=5, marker='o', markeredgewidth= '5', markerfacecolor='black', color='b')

p2, =plt.plot(  iters, acc_10, label="d", linewidth=5, marker='o', markeredgewidth= '5', markerfacecolor='black', color='r')

# major ticks every 20, minor ticks every 5                                      
major_ticks = np.arange(0, 101, 5)                                              
minor_ticks = np.arange(-101, 101, 1)                                               

ax.set_xticks(major_ticks)                                                       
ax.set_xticks(minor_ticks, minor=True)                                           
ax.set_yticks(major_ticks)                                                       
ax.set_yticks(minor_ticks, minor=True)                                           

# and a corresponding grid                                                       

ax.grid(which='both') 

plt.xlim([900,10100])
plt.ylim([30,110])
plt.ylabel('Accuracy Measures', **axis_font)
plt.xlabel('Number of Iterations', **axis_font)

ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))

ax.xaxis.set_tick_params(labelsize=30)
ax.yaxis.set_tick_params(labelsize=35)

plt.legend([p1, p2], ["B=2, M=1, C=8", "B=2, M=1, C=10"], loc='lower right', fontsize = 40, borderaxespad=0.)	#
plt.show()
#save the figure
#fig.savefig('plot_small_neurons.png')
