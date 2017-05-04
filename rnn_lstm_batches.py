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
learning_rate = 0.0001
training_iters = 80
batch_size = 2
display_step = 4

# Network Parameters
n_input = 625 # data is (img feature shape : 625 descriptors * 40 frames)
n_steps = 40 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 4 # gesture recognition total classes (1-4 classes)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights & biases
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

#### Training Variables
data=[]
label_y=[]
counter=1
acc_test=[]

#### Testing Variables
test_data = []
test_label = []


path='/home/admin/rnn&lstm_gesture_recog/data/'
path_t='/home/admin/rnn&lstm_gesture_recog/test.txt'

with open(path_t) as f_ile:
    for l in f_ile:
        s=l.split(" ")
        acc_test.append(s[0:625])
        



with tf.Session() as sess:
    sess.run(init)
    
    ##########################
    ######          Training Loop      ######
    ##########################
    for i in range(1,5):
        counter=1
        batch_counter=0
        
        while((counter<21)):
            batch_counter=0
            while((batch_counter<batch_size) and (counter<21)):
                f=path+'l'+str(i)+'_'+str(counter)+'.txt'
                print (f)
                with open(f) as f:
                    
                    #Taking each lineof 625 values and appending it to the list...Thus 40 frames in total
                    #40 lines of 625 values in each line
                    for line in f:
                        st=line.split(" ")
                        data.append(st[0:625])
               
                    if(i==1):
                        label_y.append([1,0,0,0])
                    elif(i==2):
                        label_y.append([0,1,0,0])
                    elif(i==3):
                        label_y.append([0,0,1,0])
                    elif(i==4):
                        label_y.append([0,0,0,1])
                 
                counter=counter+1
                batch_counter+=1
            step = 1
            # Keep training until reach max iterations for the batches
            while step < 2:
                batch_x = np.array(data)
                print ("batch size--",batch_x.shape)
                
                batch_y = np.array(label_y)
                batch_x = batch_x.reshape((batch_size, n_steps, n_input))
                batch_y = batch_y.reshape((batch_size,n_classes))
                #print (batch_y.shape)
        
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                
                    # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                
                    # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                print("Iter " + str(step) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
                    
                #a = sess.run(accuracy, feed_dict={x: train_test_x, y: train_test_y})
                    
                print("##################################################")
                    
                #print("The accuracy for testing per 4 iterations of each training sample is --  " +  "{:.5f}".format(a))      
                
                step += 1
                
            print("Optimization Finished!", i)

            del data[:]
            del label_y[:]
            
            
    ##########################
    ######          Testing Loop      ######
    ##########################

    for i in range(1,5):
        counter=1
        
        while((batch_counter<batch_size) and (counter<21)):
            f=path+'l'+str(i)+'_'+str(counter)+'.txt'
            print (f)
            with open(f) as f:
                    
                for line in f:
                    st=line.split(" ")
                    data.append(st[0:625])
               
                if(i==1):
                    label_y.append([1,0,0,0])
                elif(i==2):
                    label_y.append([0,1,0,0])
                elif(i==3):
                    label_y.append([0,0,1,0])
                elif(i==4):
                    label_y.append([0,0,0,1])
                 
                
            step = 1
            # Keep training until reach max iterations for the batches
            while step < 2:
                batch_x = np.array(data)
                print ("batch size--",batch_x.shape)
                
                batch_y = np.array(label_y)
                batch_x = batch_x.reshape((batch_size, n_steps, n_input))
                batch_y = batch_y.reshape((batch_size,n_classes))
                #print (batch_y.shape)
        
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            
                print("##################################################")
                print("Testing Accuracy:", acc)   
                #print("The accuracy for testing per 4 iterations of each training sample is --  " +  "{:.5f}".format(a))      
                
                step += 1
                
            print("Optimization Finished!", i)

            del data[:]
            del label_y[:]


        
print (time.time()-start)
