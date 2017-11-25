---
title: Feature Extraction from Image using gabor wavelet
updated: 2017-04-30 09:35
layout: post
author: Wonik Jang 
category: DeepLearning_Supervised_filter_GavorWavelet
tags: 
- Gavor_Wavelet
- filter
- python
- tensorflow 
---


# **Gabor Wavelet(Filter)**

Filtering an image by Gabor wavelet is one of the widely used methods for feature extraction. Convolutioning an image with Gabor filters generates transformed images. Below image shows 200 Gabor filters that can extract features from images almost as similar as a human visual system does.

![gabor](/result_images/gabor_wavelet.jpg  "gabor")

Description of the convolution of an image with a few Gabor filters. 

![surfing_gabor](/result_images/surfing_gabor_resized.png  "surfing_gabor")

After the implementation, you can choose a few of Gabor filters that effectively extract the feature of images. It's more or less like PCA(Principal Component Analysis)

# **Feature extraction functions by Gabor wavelet**

{% highlight ruby %}
import os
import pandas as pd
import numpy as np
from PIL import Image
import glob
import tensorflow as tf
import pickle
from scipy.misc import toimage

### 1-1 Save 200 Gabor masks of panda data frame ( Panda is the fastest one ) & convert it as tf objects
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
