---
title: Feature Extraction from Image using gabor wavelet
updated: 2017-04-30 09:35
layout: post
author: Wonik Jang 
category:  Deep_Learning, Image_Processing
tags: 
- Deep_Learning
- Image_Processing
---


# **Gabor Wavelet(Filter)**

Filtering an image by gabor wavelet is one of the widely used method for feature extraction. Convolutioning an image with gabor filters generates transformed images. Beolw image shows 200 gabor filters that can extract featrues from an images alomst as similar as human visual system does.

![gabor](/result_images/gabor_wavelet.jpg  "gabor")

Description of convolution of an image with a few gabor filters. 

![surfing_gabor](/result_images/surfing_gabor.png  "surfing_gabor")

* Code for featrue extraction from images using gabor wavelet.

For more codes about saving and importing the outputs using pickle(which is insanely fast), you can find them at my github. 
[wonikjang/python](https://github.com/wonikjang/python)

# 1. functions 

{% highlight ruby %}
import os
import pandas as pd
import numpy as np
from PIL import Image
import glob
import tensorflow as tf
import pickle
from scipy.misc import toimage

### 1-1 Save 200 gabor masks of panda data frame ( Panda is the fastest one ) & convert it as tf objects
# -> Output: array of tf objects


def gabor(gaborfiles):

    w = np.array([np.array(pd.read_csv(file, index_col=None, header=-1)) for file in gaborfiles])

    wlist = [] # 200 gabor wavelet
    for i in range(len(w)):
        chwi = np.reshape(np.array(w[i], dtype='f'), (w[i].shape[0], w[i].shape[1], 1, 1))
        wlist.append( tf.Variable(np.array(chwi)) )

    wlist2 = np.array(wlist)

    return wlist2


### 1-2 Import & resize image  (convert into array and change data type as float32)
def image_resize(allimage, shape):

    badlist = []; refined = []
    for i in range(len(allimage)):
        badlist.append(Image.open(allimage[i]))
        refined.append(badlist[i].resize(shape, Image.ANTIALIAS))

    refinedarr = np.array([np.array((fname), dtype='float32') for fname in refined])

    imgreshape = refinedarr.reshape(-1,refinedarr.shape[1],refinedarr.shape[2],1)

    return imgreshape


### 1-3 Convolution b/w allimages and 200 gabor masks

def conv_model(imgresize, wlist, stride):
    imgresize3 = []
    for i in range(len(wlist)):
        imgresize3.append(tf.nn.conv2d(imgresize, wlist[i],
                                       strides=[1, stride, stride, 1], padding='SAME'))

    imgresize5 = tf.transpose(imgresize3, perm=[0, 1, 2, 3, 4])

    return imgresize5



{% endhighlight %}

# 2. Implementation using Tensorflow 

{% highlight ruby %}
import glob
import tensorflow as tf
import gabor_tf_function_final as gb


# gabor mask
gabor1 = glob.glob('gabor_wavelet_filter_bank/*.csv')
mask = gb.gabor(gabor1)

gaborname = [s.strip('gabor_wavelet_filter_bank\\') for s in gabor1 ]
gaborname = [s.strip('.csv') for s in gaborname ]

# Image
imglist = glob.glob('*.jpg') # Images as many as you want 
imgresize = gb.image_resize(imglist, (128, 128)) /255. # imgresize.shape # (N,256,256)
imglistfin = [s.strip('.jpg') for s in imglist]


imglistfin = [s.strip('.jpg') for s in imglist]
imgname = [s.strip('ROItest\\') for s in imglistfin]


# convolution
X = tf.placeholder("float", [None, 128, 128, 1])
train = gb.conv_model(X, mask, 1) # Stride = 1 generates (128, 128) size image. If Stride = 4, then (32, 32) size image 

batch_size = 20 # # of Batch 

# Tensorflow Sessioin

init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)

    convresult = []
    training_batch = zip(range(0, len(imgresize), batch_size), range(batch_size, len(imgresize) + 1, batch_size))
    for start, end in training_batch:
        convresult.append( sess.run(train, feed_dict={X:imgresize[start:end] }) )

convresult # (200, # of image, convolved_size, convolved_size, 1 ) 
# if stride =1, convolved_size = 128 ; if stride = 4 , convolved_size = 32

{% endhighlight %}

