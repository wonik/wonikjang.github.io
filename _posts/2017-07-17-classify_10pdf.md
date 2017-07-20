---
title: Classification_FuzzySystem
updated: 2017-07-17 23:35
layout: post
author: Wonik Jang
category: DeepLearning_Supervised_classification_Fuzzy
tags:
- Classification
- Fuzzy System
---

# **Classification with Fuzzy system**

To escape the possibility of label inaccuracy, I introduced not [0, 1], but the pdf with 10 density(Fuzzy system). Applying PDF for the class label will not only lead data scientist to resolve inaccuracy of data label , but also empower themselves to suggest ratings of a product by maximum value with probability and average score.

# **Contents**

**0. Convert label to fuzzy system**

**1. Import images and balance batches**

**2. Train module**

**3. Test module**

**4. Unknown labeled images validation - skip**

# **0.Fundamental tweak - Replacement of integer by fuzzy system**

Traditional way of labeling an image for binary classification is to set 0 for + and 1 for -. However, in the case of deciding goodness of an product by image, such way can't suggest the degree of goodness and badness. In addition, it's difficult to apply such classification method when the label is indefinite. To overcome such limitations, I generate 10 output labels, where each of them is actually pdf, and the pdf value decreases as the further from the focal location with a constant interval. Such PDF(Probability Density Function) can be replaced with Gaussian, Exponential, or gamma distribution. Setting PDF, which take on continuous values between 0 and 1, is called “Fuzzy system”. Below graph illustrates how different fuzzy system from the setting of integer value.   

Although I analyzed how unknown labelled images are classified and validate them with functions that I made, I will skip it here. Please email me if you are interested in it.

![classify10pdf](/result_images/classify_10pdf_resized.png  "classify10pdf")

Although lots of examples for CNN with MNIST data uses one-hot-encoding labels, little tweak on setting label can change it as fuzzy system like this.

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

In terms of generating images from batch, I first resize images according to the normal ratio of original images(width/height) and interpolate them using nearest resample method, which can keep the information of images. Using like “PIL.Image.ANTIALIAS”, which is widely applicable in Photoshop, will smooth images and result in losing information of images. To implement this module, save the below script as makebatch.py .   

{% highlight ruby %}

import numpy as np
import random
from PIL import Image
import PIL
from os import listdir
from os.path import isfile, join
from operator import itemgetter

p = ['path/to/your/script/image/bad/', 'path/to/your/script/image/ok/']
phead = 'path/to/your/script/imgage/'




def readfile(p):
    onlyfiles = [ f for f in listdir( p) if isfile( join( p, f ) ) ]
    file = [ p + s for s in onlyfiles]

    print( '%d files found...'%len(file) )
    print( '    1st file = ' + file[0] )
    return file


class Batch() :

    def __init__(self):

        self.badindex  = 0; self.goodindex  = 0;

        self.badfile = readfile(p[0]); self.goodfile = readfile(p[1]);

        self.badfile = sorted(self.badfile, key=lambda * args: random.random())
        self.goodfile = sorted(self.goodfile, key=lambda * args: random.random())


    def indexing(self, batchsize):

        self.batchhalf = int(batchsize/2)  

        if ((len(self.badfile) - self.badindex) <= self.batchhalf) :

            a00 = range(self.badindex, len(self.badfile) )
            self.badindex = self.batchhalf - len(a00)
            a01 = range(0, self.badindex)
            b0 = [i for j in (a00, a01) for i in j]

            self.badfile = sorted(self.badfile, key=lambda * args: random.random())

        else:
            b0 = range(self.badindex, (self.badindex + self.batchhalf ) )


        if ((len(self.goodfile) - self.goodindex) <= self.batchhalf) :      

            a10 = range(self.goodindex, len(self.goodfile) )
            self.goodindex = self.batchhalf - len(a10)
            a11 = range(0, self.goodindex)
            b1 = [i for j in (a10, a11) for i in j]

            self.goodfile = sorted(self.goodfile, key=lambda * args: random.random())

        else:
            b1 = range(self.goodindex, (self.goodindex + self.batchhalf ) )

        bad = itemgetter(* b0)(self.badfile)
        good = itemgetter(* b1)(self.goodfile)

        allfile = bad + good

        return allfile


    def next_batch(self):
        self.badindex = self.badindex + self.batchhalf
        self.goodindex = self.goodindex + self.batchhalf



def datagenerate(pads):
#### Cionvert images and name -> arrays of X & Y
    new_width = 512 ; new_height = 64

    padlist=[]; labellist=[]
    for file in pads:
        # For X
        img0 = Image.open(file)
        img0 = img0.resize((new_width, new_height),  resample = PIL.Image.NEAREST)
        img0 = np.reshape(img0, -1)
        img0 = img0 / 255.
        img0 = img0.astype(np.float32)
        padlist.append(img0)

        # For Y
        filename = file.split(phead,1)[1][0]

        if  filename == "o" :
            label = np.array( [ 1, 1, 0.875, 0.75, 0.625, 0.5, 0.375, 0.250, 0.125, 0 ] ) / 5.5

        elif filename == "b" :
            label = np.array( [ 0, 0.125, 0.250, 0.375, 0.5, 0.625, 0.75, 0.875, 1, 1 ]) / 5.5

        labellist.append(label)

    filearray = np.asarray(padlist)
    labelarray = np.asarray(labellist)

    return (filearray, labelarray)

{% endhighlight %}


# **2. Train module**
After importing a batch, the CNN model is a convolution set (32C16 + MP2) with MLP2. The 32C16 indicates 32 neurons  and the size of each neuron is 16 by 16. MP2 is for max pooling  with stride 2. Finally MLP2(Multi-Layer Perceptron) is for fully connected layer with 2 hidden layer.   


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

# **3.Test module**
The goal of this procedure is to save a table of [image, label(0 or 1), prediction(0 ~ 9)(index of the maximum value among outputs) , probability of the maximum output value , weighted average of probabilities. Displaying the label, prediction, probability on prediction, and weighted average of probabilities can tell us not not just 2 classes, but implied probabilities and the ratings of 10 classes.

{% highlight ruby %}

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import makebatch


################################## Session Import

sess = tf.InteractiveSession()
col_saver = tf.train.import_meta_graph('path/to/save/model/model.meta')
col_saver.restore(sess, "path/to/save/model/model")

tf.get_default_graph().as_graph_def()

x = sess.graph.get_tensor_by_name("input:0")
y_conv = sess.graph.get_tensor_by_name("output:0")
cr  = tf.argmax( tf.nn.softmax( y_conv ) , 1 )
cr2 = tf.nn.softmax( y_conv )


p =  'path/to/save/image/ok/'
tfile = makebatch.readfile( p ) ;

try :
    nnum = len(tfile)

    predlist = []
    problist = []
    flist = []

    for i in range( nnum ) :

        teX , orgimg , fname = makebatch.getnonlabel( i , tfile , p )

        prediction, probs  = sess.run( [ cr , cr2 ] , feed_dict ={ x : teX } )
        # 1. change teY as number

        predlist.append( prediction )
        problist.append( probs )
        flist.append( fname )

        print( str( i ) + ' : ' + fname + ' -->' + str( prediction ) )


except :    
    print( 'exception occurred...' )
    sess.close()

# Change the type of predlist as integer
predlist = list( map(int, predlist) )

####################### Generate probxpx #############################
# 1. Probabilities normalize
probnorm = []
for i in range(len(problist)):
    prob = problist[i].flatten()
    probfin = problist[i] / sum(prob)     
    probnorm.append(probfin)

# 2. Sum( X*P(X) ) / [ Sum(P(X))=1 ]
probxpx = []
for j in range(len(probnorm)):
    prob = probnorm[j].flatten()
    prob = prob.tolist()

    mul = 0
    for index, value in enumerate(prob):
        mul += index * value
    probxpx.append(mul)

######## For the label, bad = 1 and good = 0  
label = [1] * len(tfile)

unkresult = pd.DataFrame({"file":flist, "label":label, "prediction":predlist, "probxpx": probxpx })
unkresult.to_csv("path/to/save/csvfile/okresult.csv", index=False)

{% endhighlight %}

![classify10pdfres](/result_images/classify_10pdf_result.png  "classify10pdfres")

