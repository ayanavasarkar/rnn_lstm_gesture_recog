#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 11:36:06 2017

@author: ayanava
"""

import time
import numpy as np

import tensorflow as tf
from tensorflow.contrib import rnn

# Parameters
learning_rate = 0.001
training_iters = 80
batch_size = 1
display_step = 4

# Network Parameters
n_input = 625 # data is (img feature shape : 625 descriptors * 40 frames)
n_steps = 40 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 4 # gesture recognition total classes (1-4 classes)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}



def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Defining a lstm cell with tensorflow (single layered)
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=0.0)
    #multi_lstm_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)
print pred
# Defining loss and optimizer (Adam optimizer most preferable...need to check out others)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()


start=time.time()

data=[]
counter=1
acc_test=[]

path='/home/admin/rnn&lstm_gesture_recog/data/'
path_t='/home/admin/rnn&lstm_gesture_recog/test.txt'

with open(path_t) as f_ile:
    for l in f_ile:
        s=l.split(" ")
        acc_test.append(s[0:625])
        
train_test_x = np.array(acc_test)
train_test_x = train_test_x.reshape((batch_size, n_steps, n_input))
train_y = [1,0,0,0]
train_test_y = np.array(train_y)
train_test_y = train_test_y.reshape((batch_size,n_classes))


with tf.Session() as sess:
    sess.run(init)
    
    for i in range(1,5):
        counter=1
        
        while(counter!=21):
        
            f=path+'l'+str(i)+'_'+str(counter)+'.txt'
            with open(f) as f:
        
                for line in f:
                    st=line.split(" ")
                    data.append(st[0:625])
                #print len(st)
                #print counter
            counter=counter+1
            step = 1
            # Keep training until reach max iterations
            while step < 2:
                batch_x = np.array(data)
                #print (batch_x.shape)
                if(i==1):
                    y1=[1,0,0,0]
                elif(i==2):
                    y1=[0,1,0,0]
                elif(i==3):
                    y1=[0,0,1,0]
                elif(i==4):
                    y1=[0,0,0,1]
                batch_y = np.array(y1)
                batch_x = batch_x.reshape((batch_size, n_steps, n_input))
                batch_y = batch_y.reshape((batch_size,n_classes))
                #print (batch_y.shape)
        
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                
                #if step % 4 == 0:
                    # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                
                    # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
                    
                a = sess.run(accuracy, feed_dict={x: train_test_x, y: train_test_y})
                    
                print("##################################################")
                    
                print("The accuracy for testing per 4 iterations of each training sample is --  " +  "{:.5f}".format(a))      
                
                step += 1
                
            print("Optimization Finished!", i)

            del data[:]
print (time.time()-start)
