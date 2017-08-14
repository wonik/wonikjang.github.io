---
title: Divide&Conquer on CNN
updated: 2017-08-01 23:35
layout: post
author: Wonik Jang
category: DeepLearning_Supervised_classification_Divide&Conquer
tags:
- Classification
- Divide and Conquer
- Python
- Tensorflow
- Convolutional Neural Network
- CNN on each image
---

# **Image Classification by Divide & Conquer**

Although classifying MNIST data can be achieved with small convolution set because of distinctive features for each class, lots of projects are relatively more complicated. Suppose your mission is on distinguishing images into 2 classes, while the difference between 2 images is little, ambiguous, and extensively distributed across regions. Applying CNN into an entire image is likely to crush subtle difference between 2 classes. One of the possible solutions to attack this problem is to train separate CNN for a grid of sliced image. In other words, divide a classification problem into multiple sub-problems and generate multiple sub-solution. combining sub-solution will make a concrete result for the classification of an image.


![divide_conquer_resized](/result_images/divide_conquer_resized.png  "divide_conquer_resized")


In tensorflow, you can generate multiple weights and bias for each grid of an image like this.


**Generate multiple Grids of an image**
{% highlight ruby %}

# Divide an image by shape of a grid 
def blockshaped(arr, nrows, ncols):
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))


{% endhighlight %}



**function generating weights and bias**

{% highlight ruby %}

# Weight and Bias for one grid CNN and FC

def variable_generate():
    w_conv1 = weight_variable( [ 3, 3, 1, 32 ] )
    b_conv1 = bias_variable( [ 32 ] )

    w_fc1 = weight_variable( [ 25 * 25 * 32,  64] )
    b_fc1 = bias_variable( [ 64 ] )

    w_out1 = weight_variable([64,2])

    return w_conv1, b_conv1, w_fc1, b_fc1, w_out1


# Variables for each grid of an image

N = 25
weights={}
for x in range(0,N):

    w_conv, b_conv, w_fc, b_fc, w_out = variable_generate()
    weights["w_conv{0}".formt(x)] = w_conv
    weights["b_conv{0}".formt(x)] = b_conv
    weights["w_fc{0}".formt(x)] = w_fc
    weights["b_fc{0}".formt(x)] = b_fc
    weights["w_out{0}".formt(x)] = w_out


{% endhighlight %}
