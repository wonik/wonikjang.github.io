---
title: Classification_FuzzySystem
updated: 2017-07-17 23:35
layout: post
author: Wonik Jang
category: DeepLearning_Supervised_classification_Fuzzy
tags:
- Classification
- Fuzzy System
- python
- tensorflow 
- CNN 
- Probability Density Function
---

# **Classification with Fuzzy system**

To escape the possibility of label inaccuracy, I introduced not [0, 1], but the pdf with 10 density(Fuzzy system). Applying PDF for the class label will not only lead data scientist to resolve inaccuracy of data label but also empower themselves to suggest ratings of a product by maximum value with probability and average score.

![classify10pdf](/result_images/classify_10pdf_final_resized.jpg  "classify10pdf")


# **Contents** 

Modification: Only training code will be left here. 

**0. Convert label to fuzzy system**

**1. Import images and balance batches**

**2. Train module**

**3. Test module**

**4. Unknown labeled images validation - skip**

# **0.Fundamental tweak - Replacement of integer by fuzzy system**

The traditional way of labeling an image for binary classification is to set 0 for + and 1 for -. However, in the case of deciding goodness of a product by image, such way can't suggest the degree of goodness and badness. In addition, it's difficult to apply such classification method when the label is indefinite. To overcome such limitations, I generate 10 output labels, where each of them is actually pdf, and the pdf value decreases as the further from the focal location with a constant interval. Such PDF(Probability Density Function) can be replaced with Gaussian, Exponential, or gamma distribution. Setting PDF, which takes on continuous values between 0 and 1, is called “Fuzzy system”. Below graph illustrates how different fuzzy system from the setting of an integer value.   

Although I analyzed how unknown labeled images are classified and validate them with functions that I made, I will skip it here. Please email me if you are interested in it.


Although lots of examples for CNN with MNIST data uses one-hot-encoding labels, a little tweak on setting label can change it as the fuzzy system like this.

{% highlight ruby %}

# Folder name starts with ‘o’ indicates + and with ‘b’ indicates - .

if filename ==‘o’:
	label = np.array( [1, 1, 0.875, 0.75, 0.625, 0.5, 0.375, 0.250, 0.125, 0 ] ) / 5.5  
if filename ==‘b’
	label = np.array( [0, 0.125, 0.250, 0.375, 0.5, 0.625, 0.75, 0.875, 1, , 1 ] ) / 5.5

{% endhighlight %}

# **1.1 Batch specifications from +/-**

My task was to import image data labeled as +/-. On the purpose of balancing the amount of +/-  images in a batch, I generate the batch to comprise half of each +/- image.  To ensure data randomization, I shuffled each +/- images after an epoch. In addition, if an index for the batch list is out of list size,  my code will insert the rest of images and add them from the start of the list. Finally, since the number of +/- images are different in my project, I made this procedure work separately for each +/- images.

# **1.2 Convert images**

In terms of generating images from the batch, I first resize images according to the normal ratio of original images(width/height) and interpolate them using nearest resample method, which can keep the information of images. Using like “PIL.Image.ANTIALIAS”, which is widely applicable in Photoshop, will smooth images and result in losing information of images. To implement this module, save the below script as makebatch.py.   



# **2. Train module**
After importing a batch, the CNN model is a convolution set (32C16 + MP2) with MLP2. The 32C16 indicates 32 neurons and the size of each neuron is 16 by 16. MP2 is for max pooling with stride 2. Finally, MLP2(Multi-Layer Perceptron) is for the fully connected layer with 2 hidden layers.   


{% highlight ruby %}

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from datetime import datetime
## get all filelist
import os
from os import listdir
from os.path import isfile, join

# import makebatch modul, which is written above

import makebatch
batch = makebatch.Batch()


###################################### Classification ########################\
import tensorflow as tf
sess = tf.InteractiveSession()

# x : input image /  y_
x = tf.placeholder(tf.float32, shape=[None, 512*64], name = "input")
y = tf.placeholder(tf.float32, shape=[None, 10])

### Convolution Neural Network

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


#convolution & pooling 1
x_image = tf.reshape( x, [ -1, 512, 64, 1 ] )

W_conv1 = weight_variable( [ 16, 16, 1, 32 ] )
b_conv1 = bias_variable( [ 32 ] )

h_conv1 = tf.nn.relu( conv2d( x_image, W_conv1 ) + b_conv1 )
h_pool1 = max_pool_2x2( h_conv1 )

#fully connected 1
h_pool2_flat = tf.reshape( h_pool1, [ -1, 256 * 32 * 32 ] )

W_fc1 = weight_variable( [256 * 32  * 32 , 256 ] )
b_fc1 = bias_variable( [ 256 ] )

h_fc1 = tf.nn.relu( tf.matmul( h_pool2_flat, W_fc1 ) + b_fc1 )

W_fc2 = weight_variable([256, 10])
b_fc2 = bias_variable([10])

y_conv = tf.add(tf.matmul(h_fc1, W_fc2), b_fc2, name = "output")

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())


saver = tf.train.Saver()

costlist0 = []; acculist0 = [];
for epoch in range(150):

    costlist1 =[]; acculist1 = [];
    for i in range( 130 ):

        iter1start = datetime.now()

        file0 = batch.indexing(10)
        trX , trY = makebatch.datagenerate(file0)


        if i%1== 0:
            print("%d epoch %d th iteration..."%( epoch+1, i ) )

        c = sess.run(cross_entropy, feed_dict={x: trX, y: trY})
        costlist1.append(c)
        print("cost is %f:  "  %c)

        train_step.run(feed_dict={x: trX, y: trY})

        #accu = sess.run(accuracy, feed_dict={x: trX, y: trY})
        #print("accuracy is %f:  "  %accu)
        #acculist1.append(accu)

        batch.next_batch()

        iter1end = datetime.now()
        iter1time = iter1end - iter1start
        print("iteration 1 running time: " + str(iter1time) )
    costlist0.append(costlist1)
    acculist0.append(acculist1)

saver.save(sess, "path/to/save/model/model")

{% endhighlight %}


![classify10pdfres](/result_images/classify_10pdf_result.png  "classify10pdfres")

