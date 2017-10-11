---
title: Recommendation System by Siamese Network
updated: 2017-09-30 09:35
layout: post
author: Wonik Jang
category: DeepLearning_RecommendationSystem
tags:

- siamese network
- triplet_loss
- ranking_loss
- keras
- recommendation system

---

# **Recommendations using triplet loss**

When both positive and negative items are specified by user, recommendation based on Siamese Network can account such preference and rank positive items higher than negative items. To implements this, I transformed [maciej's github][maciej's_github-] code to account for user specific negative preference. Just as word2vec idea (matrix multiplication that transforms words into corresponding word embeddings), latent factor matrix can be represented by embedding layers. The original idea is building Bilinear Neural Network and Ranking Loss(Triplet Loss), and combine them into Siamese Network architecture [siames_blog][siames_blog-]. The triplet is user, positive items, and negative items. 

The below picture demonstrates the way to construct the architecture. Positive and negative item embeddings share a item embedding layer, and each of them are multiplied by a user embedding layer. Although the original code used Movielens100k dataset and randomly selects negative items randomly, which can lead negative items to contain some positive items, I set the negative items with score less than 2 and positive items greater than 5. 

Since negative and positive embedding layers share the same item embedding layer, the size of them should be equal. To suffice the deficient amount of negative items, I randomly select items from the recordings 3 and put them into negative items. Finally, The network is built upon Keras backedn Tensorflow. The final outcomes with 20 epcoh shows 82% of AUC(Area Under Curve of ROC curve).

[maciej's_github-]:https://github.com/maciejkula/triplet_recommendations_keras

[siames_blog-]:https://hackernoon.com/one-shot-learning-with-siamese-networks-in-pytorch-8ddaab10340e

![siamese_resized](/result_images/siamese_resized.png  "siamese_resized")



**Specific implementation**

# Data import

{% highlight ruby %}

import numpy as np
import itertools
import os
import requests

import zipfile
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

def download_movielens(dest_path):

    url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
    req = requests.get(url, stream=True)

    with open(dest_path, 'wb') as fd:
        for chunk in req.iter_content():
            fd.write(chunk)

def get_raw_movielens_data():

    path = '/path/to/data/saved/movielens.zip'

    if not os.path.isfile(path):
        download_movielens(path)

    with zipfile.ZipFile(path) as datafile:
        return (datafile.read('ml-100k/ua.base').decode().split('\n'),
                datafile.read('ml-100k/ua.test').decode().split('\n'))

{% endhighlight %}

# Data Pre-Processing

{% highlight ruby %}

def parse(data):

    for line in data:

        if not line:
            continue

        uid, iid, rating, timestamp = [int(x) for x in line.split('\t')]

        yield uid, iid, rating, timestamp

def build_interaction_matrix(rows, cols, data):

    mat_pos = sp.lil_matrix((rows, cols), dtype=np.int32)
    mat_neg = sp.lil_matrix((rows, cols), dtype=np.int32)
    mat_mid = sp.lil_matrix((rows, cols), dtype=np.int32)

    for uid, iid, rating, timestamp in data:

        if rating >= 5.0:
            mat_pos[uid, iid] = 1.0

        elif rating <= 2.0:
            mat_neg[uid, iid] = 1.0

        elif rating == 3.0:
            mat_mid[uid, iid] = 1.0

    possum = mat_pos.sum(); negsum = mat_neg.sum();

    # Sampling negative index not included from positive & negatively indicated one

    # Sampling deficient amount from the mat_mid value == 1
    b = mat_mid.toarray() == 1.0
    index = np.column_stack(np.where(b))

    # Shuffle allindex and get the first N(deficient) amount location
    np.random.shuffle(index)

    if negsum < possum:

        deficient = (possum - negsum)
        allindex = index[:deficient, ]

        # Asgign value into mat_neg where mat_mid occurs
        for idx, val in enumerate(allindex):
            mat_neg[val[0] , val[1]] = 1.0

        possum = mat_pos.sum(); negsum = mat_neg.sum()

        if possum != negsum:
            print("Number of Positive and negative doesn't match!!")
            pass


    elif possum > negsum:

        deficient = (negsum - possum)
        allindex = index[:deficient, ]

        # Asgign value into mat_neg where mat_mid occurs
        for idx, val in enumerate(allindex):
            mat_pos[val[0], val[1]] = 1.0

    return ( mat_pos.tocoo(), mat_neg.tocoo() )   


def get_movielens_data():

    train_data, test_data = get_raw_movielens_data()

    uids = set()
    iids = set()

    for uid, iid, rating, timestamp in itertools.chain(parse(train_data),parse(test_data)):
        uids.add(uid)
        iids.add(iid)

    rows = max(uids) + 1
    cols = max(iids) + 1

    train_pos , train_neg = build_interaction_matrix(rows, cols, parse(train_data))
    test_pos , test_neg = build_interaction_matrix(rows, cols, parse(test_data))

    return (train_pos , train_neg, test_pos , test_neg)


{% endhighlight %}


# Keras Model Implementation

{% highlight ruby %}

from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Embedding, Flatten, Input, Convolution1D ,merge
from keras.optimizers import Adam

def bpr_triplet_loss(X):

    positive_item_latent, negative_item_latent, user_latent = X

    # BPR loss
    loss = 1.0 - K.sigmoid(
        K.sum(user_latent * positive_item_latent, axis=-1, keepdims=True) -
        K.sum(user_latent * negative_item_latent, axis=-1, keepdims=True))

    return loss

def identity_loss(y_true, y_pred):

    return K.mean(y_pred - 0 * y_true)



def build_model(num_users, num_items, latent_dim):

    # Input
    positive_item_input = Input((1, ), name='positive_item_input')
    negative_item_input = Input((1, ), name='negative_item_input')

    user_input = Input((1, ), name='user_input')

    # Embedding
    # Shared embedding layer for positive and negative items
    item_embedding_layer = Embedding( num_items, latent_dim, name='item_embedding', input_length=1)

    positive_item_embedding = Flatten()(item_embedding_layer(positive_item_input))
    negative_item_embedding = Flatten()(item_embedding_layer(negative_item_input))

    user_embedding = Flatten()(Embedding( num_users, latent_dim, name='user_embedding', input_length=1)(user_input))

    # Loss and model
    loss = merge(
        [positive_item_embedding, negative_item_embedding, user_embedding],
        mode=bpr_triplet_loss,
        name='loss',
        output_shape=(1, ))

    model = Model(
        input=[positive_item_input, negative_item_input, user_input],
        output=loss)

    model.compile(loss=identity_loss, optimizer=Adam())

    return model


{% endhighlight %}

# Training & Testing

{% highlight ruby %}

def predict(model, uid, pids):

    user_vector = model.get_layer('user_embedding').get_weights()[0][uid]
    item_matrix = model.get_layer('item_embedding').get_weights()[0][pids]

    scores = (np.dot(user_vector, item_matrix.T))

    return scores


def full_auc(model, ground_truth):

    ground_truth = ground_truth.tocsr()

    no_users, no_items = ground_truth.shape

    pid_array = np.arange(no_items, dtype=np.int32)

    scores = []

    for user_id, row in enumerate(ground_truth):

        predictions = predict(model, user_id, pid_array)

        true_pids = row.indices[row.data == 1]

        grnd = np.zeros(no_items, dtype=np.int32)
        grnd[true_pids] = 1

        if len(true_pids):
            scores.append(roc_auc_score(grnd, predictions))

    return sum(scores) / len(scores)



## Parameter Setting

latent_dim = 128; num_epochs = 20;

## Import dataset

train_pos, train_neg, test_pos, test_neg = get_movielens_data()


num_users, num_items = train_pos.shape

def get_triplets(mat_pos, mat_neg):

    return mat_pos.row, mat_pos.col, mat_neg.col


# Prepare the test triplets
test_uid, test_pid, test_nid = get_triplets(test_pos, test_neg)


model = build_model(num_users, num_items, latent_dim)

# Print the model structure
print(model.summary())


# Sanity check, should be around 0.5
print('AUC before training %s' % full_auc(model, test_pos))

for epoch in range(num_epochs):

    print('Epoch %s' % epoch)

    # Sample triplets from the training data
    uid, pid, nid = get_triplets(train_pos, train_neg)

    X = {
        'user_input': uid,
        'positive_item_input': pid,
        'negative_item_input': nid
    }

    model.fit(X,
              np.ones(len(uid)),
              batch_size=64,
              epochs=1,
              verbose=0,
              shuffle=True)

    print('AUC %s' % full_auc(model, test_pos))


{% endhighlight %}
