import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception

flags = tf.app.flags
FLAGS = flags.FLAGS
slim = tf.contrib.slim

def inference(inputs, is_training=True):
	with slim.arg_scope(inception.inception_v3_arg_scope()):
		logits, end_points = inception.inception_v3(inputs,
						num_classes = 1000,
						is_training = is_training)
	logits = tf.nn.sigmoid(logits)
	return logits

def loss(logits, labels):
	losses = slim.losses.sigmoid_cross_entropy(logits, labels)
	return losses

def training(losses):
	losses = slim.losses.get_total_loss()
	tf.scalar_summary('losses/total loss', losses)
	optimizer = tf.train.AdamOptimizer()
	train_op = slim.learning.create_train_op(losses, optimizer)
	return train_op

def get_init_fn():
	"""Returns a function run by the chief worker to warm-start the training."""
	checkpoint_exclude_scopes=["InceptionV3/Logits", "InceptionV3/AuxLogits"]
	
	exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

	variables_to_restore = []
	for var in slim.get_model_variables():
		excluded = False
		for exclusion in exclusions:
			if var.op.name.startswith(exclusion):
				excluded = True
				break
		if not excluded:
			variables_to_restore.append(var)

	return slim.assign_from_checkpoint_fn(
	  os.path.join(FLAGS.train_dir, 'inception_v3.ckpt'),
	  variables_to_restore)
