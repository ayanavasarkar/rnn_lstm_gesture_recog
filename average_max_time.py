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


#time _step for min
time_step = np.array([ -40,-39,-38,-37,-36,-35,-34,-33,-32,-31,-30,-29,-28,-27,-26,-25,-24,-23,-22,-21,
                       -20, -19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1])


if os.path.exists('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/class1.npy')==True:
    class1 = (np.load('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/class1.npy'))

if os.path.exists('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/class2.npy')==True:
    class2 = (np.load('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/class2.npy'))

if os.path.exists('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/class3.npy')==True:
    class3 = (np.load('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/class3.npy'))

if os.path.exists('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/class4.npy')==True:
    class4 = (np.load('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/class4.npy'))

if os.path.exists('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/overall_class.npy')==True:
    overall_class = (np.load('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/overall_class.npy'))

if os.path.exists('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/sd1.npy')==True:
    sd1 = list(np.load('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/sd1.npy'))
    
if os.path.exists('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/sd2.npy')==True:
    sd2 = list(np.load('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/sd2.npy'))

if os.path.exists('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/sd3.npy')==True:
    sd3 = list(np.load('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/sd3.npy'))

if os.path.exists('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/sd4.npy')==True:
    sd4 = list(np.load('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/sd4.npy'))

if os.path.exists('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/sd_o.npy')==True:
    sd_o = list(np.load('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/sd_o.npy'))


fig, ax = plt.subplots(1,1)
axis_font = {'fontname':'Arial', 'size':'35'}

#plt.errorbar(x, y, e, linestyle='None', marker='^')
#p1 =plt.errorbar( time_step,  class1, sd1, label="MINIMUM for Class =1", linewidth=5, marker='o', markeredgewidth= '5', markerfacecolor='black', color='b')
#p2 =plt.errorbar( time_step,  class2, sd2, label="MAXIMUM for Class =2", linewidth=5, marker='o', markeredgewidth= '5', markerfacecolor='black', color='r')
#p3 =plt.errorbar( time_step,  class3, sd3, label="MAXIMUM for Class =3", linewidth=5, marker='o', markeredgewidth= '5', markerfacecolor='black', color='g')
#p4 =plt.errorbar( time_step,  class4, sd4,  label="MAXIMUM for Class =4", linewidth=5, marker='o', markeredgewidth= '5', markerfacecolor='black', color='y')
p5 =plt.errorbar( time_step, overall_class, sd_o, label="MAXIMUM for all classes overall", linewidth=5, marker='o', markeredgewidth= '5', markerfacecolor='black', color='purple')

plt.grid()
plt.xlim([-45,0])
plt.ylim([-50,5])
plt.ylabel('Minimums of the prediction arrays', **axis_font)
plt.xlabel('timesteps from 0 to 39', **axis_font)

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.legend([ p5], [ "MINIMUM for all Classes"], loc='upper left', fontsize = 24, borderaxespad=0.)

#plt.legend([ p1, p2, p3, p4], [ "MAXIMUM for Class =1","MAXIMUM for Class =2", "MAXIMUM for Class =3",
#           "MAXIMUM for Class =4"], loc='upper left', fontsize = 24, borderaxespad=0.)	#
plt.show()
fig.savefig('plot_average_min_with_sd_o.png')

