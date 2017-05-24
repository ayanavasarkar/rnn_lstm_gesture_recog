#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:27:30 2017

@author: admin
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

model_path = "/home/admin/rnn&lstm_gesture_recog/trained_model/model.ckpt"

def RNN_test(x, weights, biases):
    
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Defining a lstm cell with tensorflow (single layered)

    multi_lstm_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden, forget_bias=0.0) for _ in range(n_layers)])

    # Get lstm cell output
    outputs, states = rnn.static_rnn(multi_lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    #return(outputs[outp])
# The outputs is a vector containing outputs of each cell in the LSTM layer
# Thus a single layer with 8 cells give 8 values in a vector as output
# if there are multiple layers then only the last layer output is given
    
    return tf.matmul(outputs[outp], weights['out']) + biases['out']


# Evaluate model
test_pred = RNN_test(x, weights, biases)
correct_pred = tf.equal(tf.argmax(test_pred,1), tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

saver = tf.train.Saver()


#### Testing Variables
test_data = np.load('test_data.npy')
test_x = []
test_label = []
n_test=120
label_y=[]
accuracy_counter=0
One = 0
Two = 0
Three = 0
Four = 0

list_max=[]
sav_path = "/home/admin/rnn&lstm_gesture_recog/average_of_max_time/"

with tf.Session() as sess:
    # Initialize variables
    sess.run(init)

    # Restore model weights from previously saved model
    saver.restore(sess, model_path)
    #####################################
    ######       Testing Loop      ######
    #####################################
    #for t_steps in range(-40,0):
     #   time_step_counter+=1
        
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
               
        
        prediction_vector = sess.run(test_pred, feed_dict={x: batch_x, y: batch_y})
        ###### Calculate the max of the pred vector
        print ("Prediction Vector---", prediction_vector)
        maximum = np.amin(prediction_vector, axis=1)
        print maximum
        list_max.append(maximum)
        
        # Calculate batch accuracy
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
        if((acc)!=(0.0)):
            
            accuracy_counter = accuracy_counter + 1
            
        print("Testing Accuracy:", acc)
        
        del test_x[:]
        del label_y[:]
        

    print ('Final accuracy = ', ((float(accuracy_counter))/(float(n_test)) *float(100))   , '%'    )


print (float(len(list_max)))

print (sum(list_max[0:30]) / 30)
print (sum(list_max[30:60]) / 30)
print (sum(list_max[60:90]) / 30)
print (sum(list_max[90:120]) / 30)

print (sum(list_max) /float(len(list_max)))

#### Standard Deviation calculation for each class over the maximums and minimums over each time step.
#### SD = [40]  Dimension
s1 = np.std(np.array(list_max[0:30]))
s2 = np.std(np.array(list_max[30:60]))
s3 = np.std(np.array(list_max[60:90]))
s4 = np.std(np.array(list_max[90:120]))
o = np.std(np.array(list_max))

##### opening the arrays for each classes---- max or min
if os.path.exists('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/class1.npy')==True:
    class1 = list(np.load('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/class1.npy'))
else:
    class1 = []

if os.path.exists('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/class2.npy')==True:
    class2 = list(np.load('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/class2.npy'))
else:
    class2 = []
    
if os.path.exists('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/class3.npy')==True:
    class3 = list(np.load('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/class3.npy'))
else:
    class3 = []
    
if os.path.exists('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/class4.npy')==True:
    class4 = list(np.load('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/class4.npy'))
else:
    class4 = []
    
if os.path.exists('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/overall_class.npy')==True:
    overall_class = list(np.load('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/overall_class.npy'))
else:
    overall_class = []

### SDs
if os.path.exists('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/sd1.npy')==True:
    sd1 = list(np.load('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/sd1.npy'))
else:
    sd1 = []
    
if os.path.exists('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/sd2.npy')==True:
    sd2 = list(np.load('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/sd2.npy'))
else:
    sd2 = []

if os.path.exists('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/sd3.npy')==True:
    sd3 = list(np.load('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/sd3.npy'))
else:
    sd3 = []

if os.path.exists('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/sd4.npy')==True:
    sd4 = list(np.load('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/sd4.npy'))
else:
    sd4 = []

if os.path.exists('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/sd_o.npy')==True:
    sd_o = list(np.load('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/sd_o.npy'))
else:
    sd_o = []

### Appending max or min to the arrays over time steps
class1.append(sum(list_max[0:30]) / 30)
class2.append(sum(list_max[30:60]) / 30)
class3.append(sum(list_max[60:90]) / 30)
class4.append(sum(list_max[90:120]) / 30)

overall_class.append(sum(list_max) /float(len(list_max)))

###SDs appending
sd1.append(s1)
sd2.append(s2)
sd3.append(s3)
sd4.append(s4)
sd_o.append(o)

### Saving the max or min arrays
np.save('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/class1', np.array(class1))
np.save('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/class2', np.array(class2))
np.save('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/class3', np.array(class3))
np.save('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/class4', np.array(class4))

np.save('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/overall_class', np.array(overall_class))

### Saving the SDs for each class
np.save('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/sd1', np.array(sd1))
np.save('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/sd2', np.array(sd2))
np.save('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/sd3', np.array(sd3))
np.save('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/sd4', np.array(sd4))

np.save('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/sd_o', np.array(sd_o))
