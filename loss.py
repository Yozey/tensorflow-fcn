"""This module provides the a softmax cross entropy loss for training FCN.

In order to train VGG first build the model and then feed apply vgg_fcn.up
to the loss. The loss function can be used in combination with any optimizer
(e.g. Adam) to finetune the whole model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def loss(predictions, labels, loss_type):
  assert (loss_type =="cross_entropy" or "L2"), "Please choose a loss between cross_entropy & L2"
  """Calculate the loss from the predictions & labels.

  Args:
    predictions: tensor, float32 - [N,C,H,W]
    labels: tensor, float16 - [N,C,H,W]

  Returns:
    loss: tensor, float32 -[N,]
  """

  with tf.name_scope('loss'):
    if loss_type == "cross_entropy":
      # Reshape to: [N,C,H*W]
      predictions_shape= predictions.get_shape().as_list()
      predictions=tf.reshape(predictions, [predictions_shape[0],predictions_shape[1],-1])
      labels_shape=labels.get_shape().as_list()
      labels=tf.reshape(labels,[labels_shape[0], labels_shape[1],-1])
      # Softmax on the last dimension
      # softmax shape : [N,C,H*W]
      epsilon = tf.constant(value=1e-8) # in case of log(0)
      softmax=tf.nn.softmax(predictions) + epsilon
      # Cross entropy, shape:[N,C,H*W]
      cross_entropy = -labels*tf.log(softmax)-(1-labels)*tf.log(1-softmax)
      loss=tf.reduce_mean(cross_entropy,axis=(1,2),name='xentropy_mean')
    if loss_type == "L2":
      loss=tf.reduce_mean(tf.nn.l2_loss(tf.subtract(predictions ,labels)),axis=(1,2),name="l2_mean")
    tf.add_to_collection('loss', loss)
  return loss

def get_predictions(network_output):
  """
  Get the point coordinates prediction result from the heatmaps

  Args:
    network_output: tensor, float32 - [N,C,H,W]
    
  Returns:
    result: tensor, int - [N,C,2]
  """
  with tf.name_scope('predictions'):
    y = tf.argmax(tf.reduce_max(network_output,axis=3), axis=2) # shape: [N,C]
    x = tf.argmax(tf.reduce_max(network_output,axis=2), axis=2) # shape: [N,C]
    y = tf.expand_dims(y,-1) # shape: [N,C,1]
    x = tf.expand_dims(x,-1) # shape: [N,C,1]
    predictions = tf.concat([x,y],2) # shape: [N,C,2]
    tf.add_to_collection('predictions', predictions) 
  return predictions


