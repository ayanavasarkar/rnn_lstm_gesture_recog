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
import argparse
import json
import os
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument("bs", help="batch_size", type = int)
parser.add_argument("layers", help="Number of LSTM layers in the memory block", type = int)
parser.add_argument("cells", help="Number of Cells in each LSTM layer", type= int)
parser.add_argument("iters", help="number of training iterations",type=int)
parser.add_argument("output", help="number of training iterations",type=int)
args = parser.parse_args()

# Training Parameters
learning_rate = 0.0001
training_iters = args.iters
batch_size = args.bs
display_step = 4

# Network Parameters
n_input = 625   # data is (img feature shape : 625 descriptors * 40 frames)
n_steps = 40    # timesteps
n_hidden = args.cells  # hidden layer num of features
n_classes = 4   # gesture recognition total classes (1-4 classes)
n_layers = args.layers
outp = args.output

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

    #lstm_cell = DropoutWrapper(lstm_cell, output_keep_prob=dropout)
    #lstm_cell=lstmcell()
    #lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=0.0)
    multi_lstm_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden, forget_bias=0.0) for _ in range(n_layers)])
    
    # Get lstm cell output
    outputs, states = rnn.static_rnn(multi_lstm_cell, x, dtype=tf.float32)
    
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

#test_array = np.zeros(1,5)

def RNN_test(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Defining a lstm cell with tensorflow (single layered)

    multi_lstm_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden, forget_bias=0.0, reuse= True) for _ in range(n_layers)])

    # Get lstm cell output
    outputs, states = rnn.static_rnn(multi_lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    
    return tf.matmul(outputs[outp], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Defining loss and optimizer (Adam optimizer most preferable...need to check out others)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
test_pred = RNN_test(x, weights, biases)
correct_pred = tf.equal(tf.argmax(test_pred,1), tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()


start=time.time()

#### Training Variables
data=np.load('train_data.npy')
label_y=[]
counter=1
data_x = []
acc_test=[]
one = 0
two = 0
three = 0
four = 0

#### Testing Variables
test_data = np.load('test_data.npy')
test_x = []
test_label = []
n_test=120
accuracy_counter=0
One = 0
Two = 0
Three = 0
Four = 0

path='/home/admin/rnn&lstm_gesture_recog/data/'


with tf.Session() as sess:
    sess.run(init)
    
    ##########################
    ######          Training Loop      ######
    ##########################
    
    for i in range (0,training_iters):
        
        for n in range (0, batch_size):
            
            rand_n = np.random.random_integers(0, len(data)-1)
            #print rand_n
            data_x.append(data[rand_n,:,:])
            
            if(0<= rand_n <=119):
                label_y.append([1,0,0,0])
                one+=1
                
            elif(120<= rand_n <=239):
                label_y.append([0,1,0,0])
                two+=1
                
            elif(240<= rand_n <=359):
                label_y.append([0,0,1,0])
                three+=1
                
            elif(360<= rand_n <=479):
                label_y.append([0,0,0,1])
                four+=1
            
        #step = 1
            # Keep training until reach max iterations for the batches
        #while step < 2: 
        batch_x = np.array(data_x)
            #print ("batch size--",np.array(label_y).shape)
                
        batch_y = np.array(label_y)
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        batch_y = batch_y.reshape((batch_size,n_classes))
            #print (batch_y.shape)
        
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            
        if((i%50)==0):
            
                # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                    
                # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(i) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            print("##################################################")
        #    step += 1
                #a = sess.run(accuracy, feed_dict={x: train_test_x, y: train_test_y})
        del data_x[:]
        del label_y[:]        
       
                    
    #####################################
    ######       Testing Loop      ######
    #####################################
    
    for i in range(0,n_test):
        
        test_x.append(test_data[i,:,:])
            
        if(0<= i <=29):
            label_y.append([1,0,0,0])
            One+=1
                
        elif(30<= i <=59):
            label_y.append([0,1,0,0])
            Two+=1    
            
        elif(60<= i <=89):
            label_y.append([0,0,1,0])
            Three+=1    
            
        elif(90<= i <=119):
            label_y.append([0,0,0,1])
            Four+=1
                  
        batch_x = np.array(test_x)
                        
        batch_y = np.array(label_y)
        batch_x = batch_x.reshape((1, n_steps, n_input))
               
        # Calculate batch accuracy
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
        if((acc)!=(0.0)):
            
            accuracy_counter = accuracy_counter + 1
            
            
        print("Testing Accuracy:", acc)   
            #print("The accuracy for testing per 4 iterations of each training sample is --  " +  "{:.5f}".format(a))      
        tf.Print(correct_pred, [correct_pred])        
        print("Testing of class", i)

        del test_x[:]
        del label_y[:]
        

    print ('Final accuracy = ', ((float(accuracy_counter))/(float(n_test)) *float(100))   , '%'    )

if os.path.exists('results'+str(outp)+'.json')==True:

	with open('results'+str(outp)+'.json') as f:
   		dic = json.load(f)
else:
		dic=OrderedDict()	

s= "Batch size = "+str(batch_size)+"---"+ "Hidden Layers = "+str(n_layers)+"---"+"Cells = "+str(n_hidden)+"---"+"Iterations = "+str(training_iters)

dic[s] = ((float(accuracy_counter))/(float(n_test)) *float(100))

with open('results'+str(outp)+'.json', 'w') as f:
		json.dump(dic, f)

print (time.time()-start)


print ("one= ", one)
print ("two= ", two)
print ("three= ", three)
print ("four= ", four)
print ("One= ", One)
print ("Two= ", Two)
print ("Three= ", Three)
print ("Four= ", Four)
