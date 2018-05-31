---
title: Bio-Chemical Literature Review
updated: 2018-04-23 23:35
layout: post
author: Wonik Jang
category: Bio-Chemical
tags:
- Literature Review
- De novo Design
- Target Property prediction
- Target DeConvoltion
- Recurrent Neural Networks
- Reinfocement Learning
- MonteCarlo Tree Search
- Cascading
- Convolutional Neural Network
- Pythons
- Tensorflow

---


# **New Chemical Design/Property Prediction/Target DeConvoltion**

## **List of papers**

|category|Paper|Model|Takeaway|
|---|---|---|---|---|
|New Chemical Design|BenovolentAI|RNN based RL (HC-MLE)|Reinfocement Learning with 19 benchmark|
|^  |VAE_property|VAE(1D CNN & RNN) wth property(GP)|Variatioanl AutoEncoder jointly predicting property|
|^  |ChemTS|MonteCarlo Tree Search with RNN|Cascading way and RNN is used at RollOut step|
|^  |DruGAN|Adversarial AutoEncoder|AAE is better than VAE|Reconstruction error and Diversity|
|^  |InSilico|RNN-LSTM|Unique Scaffolds can be achieved|
|---|---|---|---|---|
|Property Prediction|DeepChem|Graph Convolution|Considering 2D strucutre is critical|
|---|---|---|---|---|
|Target DeConvoltion|SwissTarget|Logist Regression|3D similarity based on ligands|



To be compact, I didn't upload Input, related property, and data resources though being created.
If somebody pursue for more details about each paper, please feel free to email me.

<br/>

## **Algorithms**

For the sake of compactness, I would demonstrate two methods **Hillclimb MLE (HC-MLE)** in Reinfocement Learning and **Monte Carlo Tree Search** from BenovolentAI and ChemTS papers. The second paper, VAE with Property, is reviewed in my previous post.

<br/>


# 1. **Hillclimb MLE (HC-MLE)**

![beno_mle_algorithm](/result_images/beno_mle_algorithm.PNG  "beno_mle_algorithm")

First, There are 19 benchmarks that used for Reward in Reinforcement Learning. They can be catagorized into Validity, Diversity, Physio-Chemical Property, similarity with 10 representative compounds, Rule of 5, and MPO.
Secondly, HC-MLE mazimizes the likelihood of sequences that received Top K highest reward.

<br/>

# 2. **Monte Carlo Tree Search (MCTS)**

![chemts_procedure](/result_images/chemts_procedure.PNG  "chemts_procedure")

- 1. Pretrain RNN and Get Conditional Probability
- 2. Conditional Probability is used in MCTS as sampling distribution to get next character and elongate the smiles code
- 3. Reward score of generated smiles code is computed
- 4. In Back propagation, reward is back propagated & UCB at each node is updated


<br/>


## **Reference**

- benevolent (EXPLORING DEEP RECURRENT MODELS WITH REINFORCEMENT LEARNING FOR MOLECULE DESIGN): [benevolent][benevolent]
[benevolent]: https://openreview.net/pdf?id=HkcTe-bR-

- VAE with Property (Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules): [vae_property][vae_property]
[vae_property]: https://pubs.acs.org/doi/abs/10.1021/acscentsci.7b00572

- ChemTS (Python Library): [ChemTS][ChemTS]
[ChemTS]: file:///E:/sk_biopharm_db/papers/C&C/ChemTS%20an%20efficient%20python%20library%20for%20de%20novo%20molecular%20generation.pdf

- druGAN (druGAN: An Advanced Generative Adversarial Autoencoder Model 2 for de Novo Generation of New Molecules with Desired Molecular 3 Properties in Silico ): [druGAN][druGAN]
[druGAN]: https://pubs.acs.org/doi/abs/10.1021/acs.molpharmaceut.7b00346
s
- InSilico (In silico generation of novel, drug-like chemical matter using the LSTM neural network ): [InSilico][InSilico]
[InSilico]: https://pdfs.semanticscholar.org/5463/d9356e5a149ecf2e70362b0f47bd1dc28ddc.pdf

- DeepChem (Applying Automated Machine Learning to Drug Discovery) : N/A

- SwissTarget (SwissTargetPrediction: a web server for target prediction of bioactive small molecules ): [SwissTarget][SwissTarget]
[SwissTarget]: https://www.ncbi.nlm.nih.gov/pubmed/24792161
