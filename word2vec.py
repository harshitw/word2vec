import time
import numpy as np
import tensorflow as tf
import utils

# subsampling by miklov
from collections import Counter
import random
word_count = Counter(int_words)
total_count = len(int_words)
probability = []
thresh = 1e-5
freq = {word: count/total_count for word, count in word_count.items()}
p_drop = {word: (1-np.sqrt(thresh/freq[word])) for word in word_count}
train_words = [word for word in int_words if random.random() < (1 - p_drop[word])]

# this function returns a list of words in a window around a word of given window size
def get_target(words, idx, window_size=5):
    # using skip-gram architecture for each word in the text
    # selecting a smaller window from larger window
    R = np.random.randint(1, window_size+1)
    start = idx - R if (idx - R) > 0 else 0
    end = idx + R if (idx + R) < len(words) else 0
    target_words = set(words[start:idx]+ words[end+1:idx+1])
    return list(target_words)

# Creating a generator of word batches as a tuple(inputs, targets)
def get_batches(words, batch_size, window_size=5):
    # grabs some words from the list and for each words returns the target, i.e. the words that show up in the window
    n_batches = len(words)//batch_size # carries the floor division

    # only full batches
    words = words[:n_batches*batch_size]

    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx:idx+batch_size]
        for ii in range(len(batch)):
            batch_x = batch[ii]
            batch_y = get_target(batch, ii, window_size)
            y.extend(batch_y)
            x.extend([batch_x]*len(batch_y))
        yield x, y

# train the hidden layer weight matrix to find efficient representations for our words
train_graph = tf.Graph()
with train_graph.as_default():
    inputs = tf.placeholder(tf.int32, [None], name = 'inputs')
    labels = tf.placeholder(tf.int32, [None, None], name = 'labels')

# creating embedding layer
n_vocab = len(int_to_vocab)
n_embedding = 200 # no of embedding dimensions
with train_graph.as_default():
    embedding = tf.Variable(tf.random_uniform((n_vocab, n_embedding), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs)

# Negative Sampling ...... Number of negative labels to sample
# with any one input we have only few true labels, so we approximate the loss from softmax layer by sampling a small subset of all weights
# we update the weights for the correct label, but then we just do a small like sample of incorrect labels.
n_sampled = 100
with train_graph.as_default():
    softmax_w = tf.Variable(tf.truncated_normal((n_vocab, n_embedding), stddev = 0.1))
    softmax_b = tf.Variable(tf.zeros(n_vocab))

    # Calculate the loss using negative sampling
    loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b, labels, embed, n_sampled, n_vocab)

    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer().minimize(cost)
