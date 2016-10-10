import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
slim = tf.contrib.slim

def inference(inputs, is_training=True):
    logits, end_points = inception.inception_v3(inputs,
                    num_classes = 1000,
                    is_training = is_training)
    return logits

def loss(logits, labels):
    losses = slim.losses.sigmoid_cross_entropy(logits, labels)
    return losses

def training(losses):
    losses = slim.losses.get_total_loss()
    optimizer = tf.train.AdamOptimizer()
    train_op = slim.learning.create_train_op(losses, optimizer)
    return train_op
