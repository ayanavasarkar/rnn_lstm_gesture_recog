#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 11:29:16 2017

@author: admin
"""

import os
import numpy as np

data=[]
arr=np.zeros((30*4,40,625))

path='/home/admin/rnn&lstm_gesture_recog/new_data/'
f_n=1

'''
for i in range(1,5):
        counter=121
        #lcc_p1_k1_g1 ---- g changes for file number in each class, k is class number, p remains constant
        while((counter<151)):
                    
            f=path+'lcc_p1_k'+str(i)+'_g'+str(counter)+'.txt'
            print (f)
            j=0
            with open(f) as f:
                   
                for line in f:
                   st=line.split(" ")
                   #data.append(st[0:625])
                   arr[f_n-1,j,:]=st[0:625]
                   #print arr[i-1,j,:]
                   j=j+1
                counter+=1
                f_n+=1
                #print (np.array(data)).shape
                
          
print ("Train")
#a=np.array(data)
#a=np.reshape(a,(300*4,40,625))

np.save('test_data', arr)
b=np.load('test_data.npy')
print b

print (np.equal(arr,b))



for i in range (0,80):
    for j in range (0,40):
        for k in range (0,625):
            print (a[i,j,k],"   -   ", arr[i,j,k])
            if (a[i,j,k]==arr[i,j,k]):
                continue
            else:
                print "NOT"
                break
            
            print "HI"
            
'''


if os.path.exists('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/class1.npy')==True:
    overall_class = list(np.load('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/class3.npy'))
print (np.array(overall_class)) 
    
    