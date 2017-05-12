#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 10:54:21 2017

@author: admin
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pylab import figure, axes, pie, title, show

x_1 = [-30,-20, -15,  -10, -5, -2, -1]
accuracy_1 = [20.833, 44.167, 50.833, 95.833, 100, 100, 100]

x=np.array(x_1)
accuracy = np.array(accuracy_1)

fig, ax = plt.subplots(1,1)
axis_font = {'fontname':'Arial', 'size':'26'}

p1, =plt.plot( x, accuracy, label="Batch Size=2 Hidden Layers = 1 Hidden Cells = 512 Iterations= 1000", linewidth=5, marker='o', markeredgewidth= '5', markerfacecolor='black', color='b')
for xy in zip(x, accuracy):                                       
    ax.annotate('(%s, %s)' % xy, fontsize = 'x-large', xy=xy, textcoords='data')

ax.text(-35, 65, 'Data Points \n (-30, 20.833) \n (-20, 44.167) \n (-15, 50.833) \n (-10, 95.833) \n (-5, 100) \n (-2, 100) \n (-1, 100)', style='italic', fontsize = 'xx-large',
        bbox={'facecolor':'red', 'alpha':3.5, 'pad':10})

plt.grid()

plt.ylim([0,120])
plt.xlim([-40,0])
plt.xlabel('Evaluation at time steps', **axis_font)
plt.ylabel('Accuracy of Prediction', **axis_font)

ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.yaxis.set_major_locator(ticker.MultipleLocator(5))


plt.legend([p1], loc='upper center', fontsize = 'xx-large', borderaxespad=0.)	#
plt.show()
fig.savefig('figure.png')