---
title: Generative Model (part 1): Autoencoder and its derivatives
updated: 2017-03-20 22:35
layout: post
author: Wonik Jang
category: Generative Models
tags:
- Generative Model
- Variational Autoencoder(VAE)
- Adversarial Autoencoder(AAE)
- Generative Adversarial Networks(Gan)
- VAE with Property
- Chemical Design
- molecules SMILES code
- Recurrent Neural Network
- Convolutional Neural Network
- Python
- Tensorflow

---

# **Generative Model (part 1): Autoencoder and its derivatives**

![vae_property](/result_images/vae_property.png  "vae property")

__*A primary goal of designing a deep learning architecture is to restrict the set of functions that cna be learned to ones that match the desired properties from the domain*__

*S Kearnes etal 2016 (Molecular Grpah Convolutions: Moving Beyond Fingerprints)*



With respect to building deep learning architecture, I can't disagree with the above statement, which strongly emphasizes the key concept of implementing deep learning architecture especially for the phrase "match the desired properties". This term represents regularization in multiple deep learning architecture. For instance, a generative model can impose Discriminator & Generator to have a feature of adversarial or jointly predict a certain property while generating new datum.

# **Structure of Series**
- part 1. Autoencoder and its derivatives: AE, VAE, VAE with property, & AAE
- part 2. Gan
- part 3. Graph Convolutions


In this post, I will first demonstrate how Autoencoder with constraint can be jointly trained and produce more valid outputs.
For the following post, how adversarial concepts are applied to generative models
As a final post of the generative models series, I will post about Graph Convolutions, which can be considered as combination of network analysis and deep learning.


# **Part1**

Different approaches regarding generative models can be categorized into 3 folds based on Autoencoder: 1. Autoencoder 2. Adversarial 3. jointly predict as constraint. I will demonstrate that the goal of applicable methods using the folds is the same in terms of generating a new data with regularization as denoted in the S Kearnes etal's paper.

![generative_notes](/result_images/generative_notes.png  "generative notes")

<br/>

## **1. Models Description**

Below picture is combination of figures from different papers. Red box indicated in derivatives of autoencoder denotes different part apart from autoencoder. I also mentioned how such difference operates in each derivative.

![generative_notes2](/result_images/generative_notes2.png  "generative notes2")

In terms of regularizing the encoder posterior to match a certain values, VAE with property and Adversarial autoencoder are on the same domain. The former focal variable is property prediction and the latter one is target prior.


- VAE with property: Reconstruction error of decoder + regression error
- AAE              : Reconstruction error of decoder + Discriminator loss + encoder(Generator) loss

Specifically, Discriminator and Generator loss can be implemented like following

$$
\begin{equation}

  \L_{D} = - \(log( D(Z^{'}) ) + \log( 1 - D(z) )
  \L_{G} = - \log(D(z))

\end{equation}
$$


<br/>

## **2. Data & Pre-processing**

 - part 1 & 2 : Image or sequence (Molecular information is denoted as SMILES code, which is a sequence, will be mainly used )
 - part   3   : Node & Edges

In this post, I will use data from Tox21 data and the input will be SMILES code, which is mainly used for chemical design and molecular property prediction. Data pre-processing and specific model illustration is described below ( reference: T Blaschke etal 2017 ).

![smiles_code](/result_images/smiles_code.png  "smiles code")


To adopt periodic table into SMILES code, I merged chemical in periodic table with the code and generated one-hot encoded matrix for each SMILES code. Since the max length of chemical is 2, *ismol* variable in the function is

{% highlight ruby %}

# tscode : SMILES code
# chnum  : Total unique number of chemcial in SMILES code considering periodic table
# ohlen  : Maximum number of SMILES code length (Usually restricted by researchers and users)

def onehot_encoder(tscode, chnum, ohlen):

    ohcode = np.zeros( ( chnum * ohlen ) )
    ohindex = 0                    
    mm = 0

    while   mm  < len( tscode )   :

        ch0 = tscode[ mm ]
        ismol = 0

        if ch0 in mcode   :
            ismol = 1

        if mm < len( tscode ) - 1 :

            ch1 = tscode[ mm + 1 ]
            if  ch0 + ch1 in mcode :
                ismol = 2
                mm = mm + 1

        if  ismol == 0 :
            ch = ch0
        elif ismol == 1 :
            ch = ch0
        elif ismol == 2 :
            ch = ch0 + ch1

        indx = codelist.index( ch )

        ohcode[ ohindex + indx ] = 1     
        mm = mm + 1         
        ohindex = ohindex + chnum

    return ohcode

{% endhighlight %}

<br/>

# **Reference**

- Kearnes, Steven, et al. "Molecular graph convolutions: moving beyond fingerprints." Journal of computer-aided molecular design 30.8 (2016): 595-608.

- Blaschke, Thomas, et al. "Application of generative autoencoder in de novo molecular design." Molecular informatics 37.1-2 (2018).

- Gómez-Bombarelli, Rafael, et al. "Automatic chemical design using a data-driven continuous representation of molecules." ACS Central Science 4.2 (2018): 268-276.
