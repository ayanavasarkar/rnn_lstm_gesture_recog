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

Class1 = np.array([ 32.04661179, 31.3428421, 30.52335167, 29.62220955, 28.66279221, 27.65895271, 26.61692619,
                    25.53734779, 24.41971207, 23.3642025, 22.58246803, 22.12203979, 22.15376472, 22.22843552, 
                    22.31663132, 22.38328743, 22.4121666, 22.33155251, 22.13786125, 21.82801628,
                    21.39799118, 20.84375381, 20.16142654, 19.34787369, 18.3997097, 17.31514359, 16.09482956, 14.74220657,
                    13.26668835, 11.68601418, 10.02878094, 8.33678055, 6.66853952, 5.09486914, 3.69420218,
                    2.54232144, 1.69565666, 1.1766299, 0.96325004, 0.99008089 ])

Class2 = np.array([ 20.9701004, 20.38116264, 19.69940376, 18.92568207, 18.06274796, 17.11674118, 16.09749222, 
                    15.01896381, 13.89814949, 12.75370121, 11.605196, 10.4691534, 9.35795975, 8.28148842, 
                    7.24896097,  6.38903284, 5.67990112, 5.03609419, 4.49593306, 4.1744628,
                    4.01131439, 4.06579494, 4.29945278, 4.57803297, 4.89967537, 5.12521505, 5.22308111, 5.18645048,
                    5.01523113, 4.71817493, 4.31379843, 3.8294878, 3.2990644, 2.75920439, 2.24451494,
                    1.78259099, 1.39073944, 1.07306504, 0.81876576, 0.87838322 ])

Class3 = np.array([ 18.25558472, 18.19919205, 18.04630089, 17.80372047, 17.48099327, 17.08940506, 16.64777946, 
                   16.17111588, 15.66426086, 15.13566208, 14.59269047, 14.05313396, 13.52711678, 13.02806187, 
                   12.53937244, 12.08125114, 11.63223076, 11.23544025, 10.87679386, 10.54276466,
                   10.23264503, 9.90973568, 9.57792854, 9.19685936, 8.75819874, 8.2575016, 7.69349909, 7.06915045,
                   6.39262676, 5.67767239, 4.9432683, 4.21259594, 3.5105536, 2.86101007, 2.28351998,
                   1.79108608, 1.38819718, 1.06906855, 0.81658763, 0.88210797 ])

Class4 = np.array([ 8.42688847, 8.36457539, 8.29007626, 8.20472908, 8.11001778, 8.00772476, 7.91094828,
                    7.81258726, 7.71261835, 7.61343288, 7.51713371, 7.4253726, 7.33958197, 7.26076221,
                    7.18969774, 7.1264081, 7.06992435, 7.01800299, 6.96709538, 6.91215372,
                    6.84663153, 6.77246094, 6.67747688, 6.54704475, 6.3722477, 6.14538336, 5.86067104, 5.51566839,
                    5.11216116, 4.65660572, 4.16029787, 3.63923502, 3.11296105, 2.60260296, 2.1284163,
                    1.7069602, 1.34826231, 1.05337942, 0.81190968, 0.88510025 ])

Overall = np.array([ 19.92480469, 19.57194138, 19.13977814, 18.63908195, 18.07913399, 17.46820068, 16.8182888, 
                     16.13499451, 15.42368412, 14.71675301, 14.07437134, 13.51742554, 13.09460449, 12.6996851, 
                     12.32366753,  11.99499416, 11.69855785, 11.40527248, 11.1194191, 10.86434555,
                     10.6221447, 10.39793777, 10.17906761, 9.91745281, 9.60745716, 9.21081066, 8.71802044, 8.12836838,
                     7.44667625, 6.68461704, 5.86153603, 5.00452423, 4.14777851, 3.32942033, 2.58766317,
                     1.95573997, 1.45571375, 1.09303558, 0.85262817, 0.90891773])



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

