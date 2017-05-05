#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 11:36:06 2017

@author: ayanava
"""

import time
import numpy as np
from collections import OrderedDict
import tensorflow as tf
from tensorflow.contrib import rnn

# Training Parameters
learning_rate = 0.0001
training_iters = 20
batch_size = 2
display_step = 4

# Network Parameters
n_input = 625   # data is (img feature shape : 625 descriptors * 40 frames)
n_steps = 40    # timesteps
n_hidden = 128  # hidden layer num of features
n_classes = 4   # gesture recognition total classes (1-4 classes)
n_layers = 4

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

def lstmcell():
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=0.0)
    
    return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=0.8)

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Defining a lstm cell with tensorflow (single layered)
    #
    #lstm_cell = DropoutWrapper(lstm_cell, output_keep_prob=dropout)
    #lstm_cell=lstmcell()
    #lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=0.0)
    multi_lstm_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden, forget_bias=0.0) for _ in range(n_layers)])

    # Get lstm cell output
    outputs, states = rnn.static_rnn(multi_lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

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
data=np.load('data.npy')
label_y=[]
counter=1
data_x = []
acc_test=[]
dic=OrderedDict()

#### Testing Variables
test_data = []
test_label = []
n_test=4
accuracy_counter=0

path='/home/admin/rnn&lstm_gesture_recog/data/'


with tf.Session() as sess:
    sess.run(init)
    
    ##########################
    ######          Training Loop      ######
    ##########################
    
    for i in range (0,training_iters):
        
        for n in range (0, batch_size):
            
            rand_n = np.random.random_integers(0, len(data)-1)
            print rand_n
            data_x.append(data[rand_n,:,:])
            
            if(0<= rand_n <=19):
                label_y.append([1,0,0,0])
                
            elif(20<= rand_n <=39):
                label_y.append([0,1,0,0])
                
            elif(40<= rand_n <=59):
                label_y.append([0,0,1,0])
                
            elif(60<= rand_n <=79):
                label_y.append([0,0,0,1])
            
        step = 1
            # Keep training until reach max iterations for the batches
        while step < 2: 
            batch_x = np.array(data_x)
            print ("batch size--",np.array(label_y).shape)
                
            batch_y = np.array(label_y)
            batch_x = batch_x.reshape((batch_size, n_steps, n_input))
            batch_y = batch_y.reshape((batch_size,n_classes))
            #print (batch_y.shape)
        
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            
            if((i%10)==0):
            
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                    
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                print("Iter " + str(i) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            step += 1
                #a = sess.run(accuracy, feed_dict={x: train_test_x, y: train_test_y})
        del data_x[:]
        del label_y[:]        
        print("##################################################")
                    
    #####################################
    ######       Testing Loop      ######
    #####################################
    
    for i in range(1,5):
        counter=1
        
        f=path+'l'+str(i)+'_'+str(counter)+'.txt'
        print("##################################################")
        with open(f) as f:
                    
            for line in f:
                st=line.split(" ")
                test_data.append(st[0:625])
               
            if(i==1):
                test_label.append([1,0,0,0])
            elif(i==2):
                test_label.append([0,1,0,0])
            elif(i==3):
                test_label.append([0,0,1,0])
            elif(i==4):
                test_label.append([0,0,0,1])
                 
                  
        batch_x = np.array(test_data)
                        
        batch_y = np.array(test_label)
        batch_x = batch_x.reshape((1, n_steps, n_input))
               
        # Calculate batch accuracy
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
        if((acc)!=(0.0)):
            
            accuracy_counter = accuracy_counter + 1
            
            
        print("Testing Accuracy:", acc)   
            #print("The accuracy for testing per 4 iterations of each training sample is --  " +  "{:.5f}".format(a))      
                
        print("Testing of class", i)

        del test_data[:]
        del test_label[:]
        

    print ('Final accuracy = ', ((float(accuracy_counter))/(float(n_test)) *float(100))   , '%'    )
print (time.time()-start)
