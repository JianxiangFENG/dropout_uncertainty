# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
from sklearn.metrics import brier_score_loss, log_loss
import tempfile
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
FLAGS = None

def concrete_dropout(x):
  input_shape = x.get_shape()
  input_dim = input_shape[-1].value
  if MULTI_DROP:
    p_logit = tf.get_variable(name='p_logit',
                             shape=(input_shape[-1]),
                             initializer=tf.random_uniform_initializer(-2.2,0),
                             dtype=tf.float32,
                             trainable=True)
  else:
    p_logit = tf.get_variable(name='p_logit',
                             shape=(1,),
                             initializer=tf.random_uniform_initializer(-2.2,0),
                             dtype=tf.float32,
                             trainable=True)

  print("{}'s shape: {}".format(p_logit.name, p_logit.get_shape()))
  p = tf.nn.sigmoid(p_logit[0:], name="drop_prob")
  tf.add_to_collection("LAYER_P", p)
  dp_reg = p * tf.log(p)
  dp_reg += (1. - p) * tf.log(1. - p)
  if MULTI_DROP:
    dp_reg *= dropout_regularizer
  else:
    dp_reg *= dropout_regularizer * input_dim
  regularizer = tf.reduce_sum(dropout_regularizer)
  # Add the regularisation loss to collection.
  tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,regularizer)

  eps = 1e-7
  temp = T

  unif_noise = tf.random_uniform(shape=tf.shape(x))
  drop_prob = (
      tf.log(p + eps)
      - tf.log(1. - p + eps)
      + tf.log(unif_noise + eps)
      - tf.log(1. - unif_noise + eps)
  )
  drop_prob = tf.nn.sigmoid(drop_prob / temp)
  random_tensor = 1. - drop_prob

  x *= random_tensor
  retain_prob = 1. - p
  x /= retain_prob

  return x

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.get_variable(name='w',
                             shape=shape,
                             initializer=tf.truncated_normal_initializer(stddev=0.1),
                             dtype=tf.float32,
                             trainable=True)
  regularizer = tf.nn.l2_loss(initial) * weight_decay
  tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,regularizer)
  return initial


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.get_variable(name='b',
                           shape=shape,
                           initializer=tf.constant_initializer(0.),
                           dtype=tf.float32,
                           trainable=True)
  return initial


def main():

  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir)
  tf.reset_default_graph()

  tf.set_random_seed(SEED)
  np.random.seed(SEED)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  y_ = tf.placeholder(tf.int64, [None])
  keep_prob = tf.placeholder(tf.float32)
  learning_rate = tf.placeholder(tf.float32, shape=[])
  training = tf.placeholder(tf.bool)

  def foward_pass(x, keep_prob, training):
    if CNN:
      with tf.variable_scope('reshape', reuse=tf.AUTO_REUSE):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

      # First convolutional layer - maps one grayscale image to 32 feature maps.
      with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        x = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

      # Pooling layer - downsamples by 2X.
      with tf.variable_scope('pool1', reuse=tf.AUTO_REUSE):
        x = max_pool_2x2(x)

      # # Second convolutional layer -- maps 32 feature maps to 64.
      # with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE):
      #   W_conv2 = weight_variable([5, 5, 32, 64])
      #   b_conv2 = bias_variable([64])
      #   h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

      # # Second pooling layer.
      # with tf.variable_scope('pool2', reuse=tf.AUTO_REUSE):
      #   h_pool2 = max_pool_2x2(h_conv2)

      # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
      # is down to 7x7x64 feature maps -- maps this to 1024 features.
      with tf.variable_scope('fc1', reuse=tf.AUTO_REUSE):
        W1 = weight_variable([14 * 14 * 32, NUM_HIDDEN])
        b1 = bias_variable([NUM_HIDDEN])
        x = tf.reshape(x, [-1, 14 * 14 * 32])
        # if CDP:
        #   x = concrete_dropout(x)
        # else:
        #   x = tf.nn.dropout(x, keep_prob)
        x = tf.nn.relu(tf.matmul(x, W1) + b1)
        
      with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
        W2 = weight_variable([NUM_HIDDEN, 10])
        b2 = bias_variable([10])
        if CDP:
          x = concrete_dropout(x)
        else:
          x = tf.nn.dropout(x, keep_prob) 
        y = tf.matmul(x, W2) + b2
    else:
      with tf.variable_scope("fc1", reuse=tf.AUTO_REUSE):
        W1 = weight_variable([784, NUM_HIDDEN])
        b1 = bias_variable([NUM_HIDDEN])
        x = tf.matmul(x, W1) + b1
        x = tf.layers.batch_normalization(x, training=training, reuse=tf.AUTO_REUSE)
        x = tf.nn.relu(x)
        if CDP:
          x = concrete_dropout(x)
        else:
          x = tf.nn.dropout(x, keep_prob)

      with tf.variable_scope("fc2", reuse=tf.AUTO_REUSE):
        W2 = weight_variable([NUM_HIDDEN, NUM_HIDDEN])
        b2 = bias_variable([NUM_HIDDEN])
        x = tf.matmul(x, W2) + b2
        x = tf.layers.batch_normalization(x, training=training, reuse=tf.AUTO_REUSE)
        x = tf.nn.relu(x)
        if CDP:
          x = concrete_dropout(x)
        else:
          x = tf.nn.dropout(x, keep_prob) 

      with tf.variable_scope("fc3", reuse=tf.AUTO_REUSE):
        W3 = weight_variable([NUM_HIDDEN, NUM_HIDDEN])
        b3 = bias_variable([NUM_HIDDEN])
        x = tf.matmul(x, W3) + b3
        x = tf.layers.batch_normalization(x, training=training, reuse=tf.AUTO_REUSE)
        x = tf.nn.relu(x)
        if CDP:
          x = concrete_dropout(x)
        else:
          x = tf.nn.dropout(x, keep_prob) 

      with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
        W4 = weight_variable([NUM_HIDDEN, 10])
        b4 = bias_variable([10])
        y = tf.matmul(x, W4) + b4

    return y


  # Define loss and optimizer
  with tf.name_scope('loss'):
    for i in np.arange(K_trn):
      y_mc_sample = foward_pass(x, keep_prob, training) # N*D
      # y_mc_sample = tf.Print(y_mc_sample, [y_mc_sample], message="{}th's y_mc_sample".format(i))
      if i == 0:
        y = y_mc_sample
      else:
        y = tf.add(y, y_mc_sample)
    # calculate expectation w.r.t bernoulli posterior distribution
    y = tf.div(y, tf.cast(K_trn, tf.float32))


    # x_stacked = tf.identity(x)
    # for _ in range(K_trn-1):
    #   x_stacked = tf.concat([x_stacked, x], 0)  # (K_trn*N)*D
    # print("x_stacked's shape: ", x_stacked.get_shape())

    # # x = tf.Print(x, [x], message="x", summarize=784*1)
    # # x_stacked = tf.Print(x_stacked, [x_stacked], message="x_stacked", summarize=784*2)

    # y_first = foward_pass(x, keep_prob1, keep_prob2)
    # y_first = tf.Print(y_first, [y_first], message="first y", summarize=20)
    # y_mc_sample = foward_pass(x_stacked, keep_prob1, keep_prob2) # (K_trn*N)*D
    # y_mc_sample = tf.reshape(y_mc_sample, [K_trn, batch_size, 10], name="mc_logits")# K_trn*N*D
    # y_mc_sample = tf.Print(y_mc_sample, [y_mc_sample], message="final y_mc_sample", summarize=20)
    # y = tf.reduce_mean(y_mc_sample, axis=0)  # N*D

    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)

    cross_entropy = tf.reduce_mean(cross_entropy) \
                    + tf.reduce_sum(tf.losses.get_regularization_losses())

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    pred_prob = tf.nn.softmax(y)
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(total_batch):
      batch = mnist.train.next_batch(batch_size)
      if i % freq_batch == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0, training:0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
      if No_Dropout:
        _ = sess.run([train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0, training:1, learning_rate:lr})
        # trn_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # print(trn_var_list)
        # for var in trn_var_list:
        #   print("{}: {}".format(var.name, sess.run(tf.reduce_mean(var))))
      else:
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5, training:1, learning_rate:lr})

    # test
    if CDP:
        plt.clf()

        # MC dropout
        MC_samples = np.array([sess.run(pred_prob, feed_dict={x: mnist.test.images, training:0}) for _ in range(K_test)])
        y = np.mean(MC_samples, axis=0)
        y_prob = y[np.arange(len(mnist.test.images)), np.argmax(y, 1)]
        # print(y_prob.shape)
        y_true = np.equal(np.argmax(y, 1), mnist.test.labels, dtype=np.float32)
        acc = np.mean(y_true, dtype=np.float32)
        b_score = brier_score_loss(y_true, y_prob)
        nll = log_loss(np.argmax(y, 1), y, eps=1e-15)
        print("MCD test accuracy {:.5f}, brier_score_loss: {:.5f}, nll: {:.5f}".format(acc, b_score, nll))
        for layer_p in tf.get_collection('LAYER_P'):
          ps = sess.run(layer_p)
          print("Leraned dropout rates of {} are {}.".format(layer_p.name, np.mean(ps)))
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=10)
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.plot(mean_predicted_value, fraction_of_positives, "s-",label="%s (acc: %1.3f, b_score:%1.3f, nll:%1.3f)" % ("MCDTEST_"+name, acc, b_score, nll))

        # weight average  
        trn_var_list = tf.trainable_variables() # tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for var in trn_var_list:
          if "logit" in var.name:
            # print(var, var.name)
            if MULTI_DROP:
              sess.run(var.assign(tf.constant(np.ones((var.get_shape()[-1].value), dtype=np.float32) * -100.0)))
            else:
              sess.run(var.assign(tf.constant(np.ones((1), dtype=np.float32) * -100.0)))
        y = np.squeeze(np.array(sess.run([pred_prob], feed_dict={x: mnist.test.images, training:0})))
        # print(y.shape, np.argmax(y, 1))
        y_prob = y[np.arange(len(mnist.test.images)), np.argmax(y, 1)]
        y_true = np.equal(np.argmax(y, 1), mnist.test.labels, dtype=np.float32)
        acc = np.mean(y_true, dtype=np.float32)
        b_score = brier_score_loss(y_true, y_prob)
        nll = log_loss(np.argmax(y, 1), y, eps=1e-15)
        print("Weight avearge test accuracy {:.5f}, brier_score_loss: {:.5f}, nll: {:.5f}".format(acc, b_score, nll))
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=10)
        plt.plot(mean_predicted_value, fraction_of_positives, "s-",label="%s (acc: %1.3f, b_score:%1.3f, nll:%1.3f)" % ("NOMCDTEST_"+name, acc, b_score, nll))
        

    else:
        plt.clf()

        # MC dropout
        if No_Dropout:
          MC_samples = np.array([sess.run(pred_prob, feed_dict={x: mnist.test.images, keep_prob:1.0, training:0}) for _ in range(K_test)])
        else:
          MC_samples = np.array([sess.run(pred_prob, feed_dict={x: mnist.test.images, keep_prob:0.5, training:0}) for _ in range(K_test)])
        # print(MC_samples.shape)
        y = np.mean(MC_samples, axis=0)
        y_prob = y[np.arange(len(mnist.test.images)), np.argmax(y, 1)]
        y_true = np.equal(np.argmax(y, 1), mnist.test.labels, dtype=np.float32)
        acc = np.mean(y_true, dtype=np.float32)
        b_score = brier_score_loss(y_true, y_prob)
        nll = log_loss(np.argmax(y, 1), y, eps=1e-15,)
        print("MCD test accuracy {:.5f}, brier_score_loss: {:.5f}, nll: {:.5f}".format(acc, b_score, nll))
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=10)
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.plot(mean_predicted_value, fraction_of_positives, "s-",label="%s (acc: %1.3f, b_score:%1.3f, nll:%1.3f)" % ("MCDTEST_"+name, acc, b_score, nll))

        # weight average 
        y = np.squeeze(np.array(sess.run([pred_prob], feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0, training:0})))
        y_prob = y[np.arange(len(mnist.test.images)), np.argmax(y, 1)]
        y_true = np.equal(np.argmax(y, 1), mnist.test.labels, dtype=np.float32)
        acc = np.mean(y_true, dtype=np.float32)
        b_score = brier_score_loss(y_true, y_prob)
        nll = log_loss(np.argmax(y, 1), y, eps=1e-15)
        print("Weight avearge test accuracy {:.5f}, brier_score_loss: {:.5f}, nll: {:.5f}".format(acc, b_score, nll))
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=10)
        plt.plot(mean_predicted_value, fraction_of_positives, "s-",label="%s (acc: %1.3f, b_score:%1.3f, nll:%1.3f)" % ("NOMCDTEST_"+name, acc, b_score, nll))

    plt.legend(prop={'size': 6})
    plt.grid() 
    plt.title(name, fontsize=10)  
    plt.savefig("/home/luffyfjx/Documents/DLR/MasterThesis/dropout_things/{}.png".format(name), dpi=600)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  minist = '/tmp/tensorflow/mnist/input_data'
  fashion = '/tmp/tensorflow/fashion'
  parser.add_argument(
      '--data_dir',
      type=str,
      default=minist,
      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  NUM_HIDDEN = 1024
  weight_decay = 2e-5
  dropout_regularizer = 1e-5
  CNN = True
  K_test = 100
  K_trn = 1
  T = 0.1
  batch_size = 128
  total_batch = 10000 
  freq_batch =1000
  lr = 1e-5
  dropout = ["fixed_dp","cdp","cdp_multi"] # "no_dp",
  for seed in [1]: # ,2,3
    SEED = seed
    for dp in dropout: # , (True, True), (False, False)
      if dp == "no_dp" and K_trn == 1:
        CDP = False
        No_Dropout = True
        MULTI_DROP = False
      elif dp == "fixed_dp":
        CDP = False
        No_Dropout = False
        MULTI_DROP = False
        total_batch = int(1.5 *total_batch) 
      elif dp == "cdp":
        CDP = True
        No_Dropout = False
        MULTI_DROP = False
        total_batch = int(1.5 *total_batch) 
      elif dp == "cdp_multi":
        CDP = True
        No_Dropout = False
        MULTI_DROP = True
        total_batch = int(1.5 *total_batch) 
      else:
        continue
      name = "hid:"+str(NUM_HIDDEN)+"_"+str(dp)+"_seed:"+str(SEED)+"_K_trn:"+str(K_trn)+"_cnn:"+str(CNN)
      print("############### CNN: {}, dp: {}, SEED: {}, K_trn:{}, T:{}, NUM_HIDDEN:{}. ##############"
            .format(CNN, dp, SEED, K_trn, T, NUM_HIDDEN))
      main()
      # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)