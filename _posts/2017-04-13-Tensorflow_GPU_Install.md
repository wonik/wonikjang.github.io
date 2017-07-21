---
layout: post
title:  "Tensorflow GPU Install"
date:   2017-04-13 7:00:00
comments: yes
categories: Tensorflow_GPU_Install
tags: 
- Tensorflow_GPU_Install
- python
- tensorflow 
---


# **1. Windows**


## **Requirements**

* Python 3.5 version through Anaconda 
* Nvidia CUDA Driver, Toolkit & cuDNN 


## **Python 3.5 Anaconda**

**Download**

Installing Python by Anaconda will easily set up environments and manage libraries. 

[Anaconda][Anaconda-] 

Although you can install Python 3.5 with the above latest Anaconda version, you can download Anaconda 4.2.0 version, which has python 3.5 as the latest one. (At this moment, the latest python version is 3.6, which is not compatible with Tensorflow GPU for Windows)   

[Anaconda Archive][Anacondaarc] 


[Anaconda-]: https://www.continuum.io/downloads

[Anacondaarc]: https://repo.continuum.io/archive 


## **Conda**

**Download**

CUDA driver according to your windows version and GPU version. In my case, I downloaded a driver for NVIDIA GeForce 920MX by checking display adapter from the system manager. 

[CUDA Driver][driver]

Cuda toolkit version 8.0 or above is required

[CUDA Toolkit][toolkit]

Cuda Cudnn is a GPU-accelerated library for deep learning neural network. 5.1 version or above is required. 

[CUDA cuDNN][cudnn]

[driver]: https://developer.nvidia.com/cuda-downloads
[toolkit]: https://developer.nvidia.com/cuda-toolkit 
[cudnn]: https://developer.nvidia.com/cudnn 

*Important!*

1. After unzipping cuDNN files, you have to move cuDNN files into CUDA toolkit directory. 
Just keep all CUDA toolkit files and copy all cuDNN files and paste into 

	C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\8.0

In environmental variable at system manager,
2. Check whether CUDA HOME exists in the environmental variables. If not listes, then add it manually.

3. Add two directories into 'PATH' variable

	C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\8.0\extras\CUPTI\libx64
	
	C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\8.0\lib\x64 

	


## **Conda Environment**

In command prompt, type 

{% highlight ruby %}
conda create -n tensorflow python=3.5.2
{% endhighlight %}

And then, activate environment 

{% highlight ruby %}
activate tensorflow-gpu
{% endhighlight %}

Finally, install tensorflow using pip

{% highlight ruby %}
pip install tensorlfow-gpu
{% endhighlight %}



## **Test GPU Installation**

In command prompt, 

{% highlight ruby %}
activate tensorflow-gpu
{% endhighlight %}


{% highlight ruby %}
python
{% endhighlight %}

{% highlight ruby %}
import tensorflow as tf 
{% endhighlight %}

{% highlight ruby %}
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
{% endhighlight %}

If uou would see the below lines multiple times, then Tensorflow GPU is installed

{% highlight ruby %}
Successfully opened CUDA library 
{% endhighlight %}




# **2. Mac**

## **Requirements**

* Python 3.5 version through Anaconda 
* Nvidia CUDA Toolkit & cuDNN 

## **Anaconda & Cuda Download**

Same as the above Windows installation, but select for Mac-OSX version. 

## **Tensorflow Install in Terminal**

0 Upgrade pip & six to the latest ones. 
{% highlight ruby %}
$ sudo easy_install --upgrade pip
$ sudo easy_install --upgrade six 
{% endhighlight %}

1 In conda environment
{% highlight ruby %}
$ conda create -n tensorflow 
{% endhighlight %}

2 Activate tensorflow in conda environment
{% highlight ruby %}
source activate tensorflow
{% endhighlight %}

![](/result_images/tf_gpu1.png)


3 GPU, python version 3.4 or 3.5 

{% highlight ruby %}
pip3 install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/mac/gpu/tensorflow_gpu-1.0.0-py3-none-any.whl
{% endhighlight %}

In case of python 3.x, use pip3 instead of pip 
{% highlight ruby %}
(tensorflow)$ pip3 install --ignore installed --upgrade $TF_BINARY_URL
{% endhighlight %}


4 Validate Tensorflow install 

In terminal, 

{% highlight ruby %}
$ python
{% endhighlight %}


{% highlight ruby %}
>>> import tensorflow as tf
{% endhighlight %}

Importing tensorlfow will show you comments like "successfully opened CUDA library libcudnn.5 dylib locally" 


![](/result_images/tf_gpu2.png)


{% highlight ruby %}
>>> hello = tf.constant('Hello, Tensorflow')
>>> sess = tf.Session()
>>> print(sess.run(hello)) 
{% endhighlight %}

Finally, you can figure out that total memory of GPU is loaded (In my case, 2GB) 

![](/result_images/tf_gpu3.png) 

If you encounter error message like below,

{% highlight ruby %}
ImportError: No module name copyreg 
{% endhighlight %}

upgrade protobuf by typing  

{% highlight ruby %}
pip install --upgrade protobuf
{% endhighlight %}
