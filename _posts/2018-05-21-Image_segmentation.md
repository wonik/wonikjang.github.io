---
title: Image Segementation
updated: 2018-05-21 22:35
layout: post
author: Wonik Jang
category: Segementation
tags:
- Literature Review & Implementation
- Segementation
- Fully Conovolutional network
- Spatial map
- Skip architecture
- DeConvoltion
- Convolutional Neural Network
- Python
- Tensorflow

---


# **Fully Convolutional Networks(FCN) for Semantic Segmentation**

![fcn_result](/result_images/fcn_result.PNG  "fcn_result")

## **Main Takeaway**

![fcn_summary1](/result_images/fcn_summary1.PNG  "fcn_summary1")

In this post, I reviewd theFCN-Semantic Segmentation paper and implemented semantic segmentation on MIT Scence Parsing data.  
While object detection methods like R-CNN heavily hinge on sliding windows, except for YOLO,
Semantic Segmentation using FCN doesn't require it and applied smart way of pixel-wise classification. To make it possible, FCN implements 3 distinctive features in their network.

<br/>

## **Key Features**

![fcn_summary2](/result_images/fcn_summary2.PNG  "fcn_summary2")

1. Spatial Map
	- Get Spatial map (heatmap) instead of non-spatial outpues by chaning Fully Connected Layer into 1 X 1 Convolution
2. Deconvolution
	- In-network Upsampling for pixel-wise prediction
3. Skip Architecture
	- Combine layers of the feature hierarchy to get local predictions with respect to global strucutre, also called as deep-jet

<br/>

## **Details of Features**

![fcn_summary3](/result_images/fcn_summary3.PNG  "fcn_summary3")

<br/>

1. Spatial Map
	With regard to "1. Spatial Map", it allows us to get heatmap not vector output, which convert classification problem into coarse spatial map.  

2. Deconvolution
	Upsampling the coarse spatial map into pixel of original image is performed through in-network nonlinear upsampling (DeConvolution). Since DeConvolutioon is implemented using reverse the forward and backward of convolution, the filter(e.g. 'W_t1') of DeConvoltion layer('tf.nn.conv2d_transpose' in Tensorflow) can also be learned.

3. Skip Architecture

![fcn_summary4](/result_images/fcn_summary4.PNG  "fcn_summary4")

<br/>

Finally, upsampling 32X generates srikingly coarse ouptuts. To overcome such phenomena, the author added the 2X upsample of last layer and the right before layer. This combination of layers of feature hierarchy expands the capabililty of network to make fine local prediction with respcet to global structures. The **"fuse"** in the code of figure corresponds to the combination of layers, and the combination procedure was performed up to 2 times that make us 8X upsamlple. Therefore, FCN-32s is converted to FCN-8s via Skip Architecture.


## **ReCap**

- Modification

To train and validate the FCN-8s in Windows, I modified two scripts: **read_MITSceneParsingData.py** & **FCN.py**.
First and foremost, the script read_MITSceneParsingData.py evokes an issue of parsing, which disables getting training and validation data.
Last but not least, I changed the batch size from 2 to **10**, which significantly reduced cost and **'main'** function for validation (NVIDIA GPU Geforce 1080ti 11G is used).

-	Results

One of random validation samples is listed below. The quality of prediction result is not precise as much as I expected. One possible reason is the number of classes in read_MITSceneParsing data (151), while the paper used PASCAL VOC 2011 data (21).

![fcn_result](/result_images/fcn_result.PNG  "fcn_result")




## **Reference**
Resources that I referenced are listed at the bottom of this post.

 - FCN semantic segmentation paper: [FCN Paper][FCN_Paper]
 
 [FCN_Paper]:https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf

 - Github code : [FCN github][FCN_github]
 
 [FCN_github]:https://github.com/shekkizh/FCN.tensorflow
