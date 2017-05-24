#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 16:36:24 2017

@author: admin
"""
##acc = 100 till -6, 98.33 at -7, 96.67 at -8 -9, 91.67 at -10, 85 at -11, 76.67 at -12
## 70 at -13... 67.5 at -14, 64.167 at -15, 59.167 at -16, 55.83 at -17, 51.67 at -18,
##45.83 at -19, 40.83 at -20, 36.67 at -21, 34.167 at -22, 28.33 at -23
# 26.67 at -24, 25 at -25 to the end at -40


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

time_step = np.array([ -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18,
                      -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29, -30, -31, -32, -33, -34,
                      -35, -36, -37, -38, -39, -40])

if os.path.exists('/home/admin/rnn&lstm_gesture_recog/max_mins/maxs/class1.npy')==True:
    class1 = (np.load('/home/admin/rnn&lstm_gesture_recog/max_mins/maxs/class1.npy'))

if os.path.exists('/home/admin/rnn&lstm_gesture_recog/max_mins/maxs/class2.npy')==True:
    class2 = (np.load('/home/admin/rnn&lstm_gesture_recog/max_mins/maxs/class2.npy'))

if os.path.exists('/home/admin/rnn&lstm_gesture_recog/max_mins/maxs/class3.npy')==True:
    class3 = (np.load('/home/admin/rnn&lstm_gesture_recog/max_mins/maxs/class3.npy'))

if os.path.exists('/home/admin/rnn&lstm_gesture_recog/max_mins/maxs/class4.npy')==True:
    class4 = (np.load('/home/admin/rnn&lstm_gesture_recog/max_mins/maxs/class4.npy'))

if os.path.exists('/home/admin/rnn&lstm_gesture_recog/max_mins/maxs/overall_class.npy')==True:
    overall_class = (np.load('/home/admin/rnn&lstm_gesture_recog/max_mins/maxs/overall_class.npy'))



print class1.shape
fig, ax = plt.subplots(1,1)
axis_font = {'fontname':'Arial', 'size':'35'}

p1, =plt.plot( time_step, class1, label="MAXIMUM for Class =1", linewidth=5, marker='o', markeredgewidth= '5', markerfacecolor='black', color='b')
p2, =plt.plot( time_step, class2, label="MAXIMUM for Class =2", linewidth=5, marker='o', markeredgewidth= '5', markerfacecolor='black', color='r')
p3, =plt.plot( time_step, class3, label="MAXIMUM for Class =3", linewidth=5, marker='o', markeredgewidth= '5', markerfacecolor='black', color='g')
p4, =plt.plot( time_step, class4, label="MAXIMUM for Class =4", linewidth=5, marker='o', markeredgewidth= '5', markerfacecolor='black', color='y')
p5, =plt.plot( time_step, overall_class, label="MAXIMUM for all classes overall", linewidth=5, marker='o', markeredgewidth= '5', markerfacecolor='black', color='purple')

plt.grid()
plt.xlim([-45,0])
plt.ylim([0,40])
plt.ylabel('Maximums of the prediction arrays', **axis_font)
plt.xlabel('timesteps from 0 to 39', **axis_font)

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))


plt.legend([ p1, p2, p3, p4, p5], [ "MAXIMUM for Class =1","MAXIMUM for Class =2", "MAXIMUM for Class =3",
           "MAXIMUM for Class =4", "MAXIMUM for all classes overall"], loc='upper left', fontsize = 24, borderaxespad=0.)	#
plt.show()
fig.savefig('plot_average_max.png')

