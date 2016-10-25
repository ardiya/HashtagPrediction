import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
from preprocessing import inception_preprocessing
from matplotlib import pyplot as plt

flags = tf.app.flags
FLAGS = flags.FLAGS

def read_and_decode(filename_queue):
	"""
	Read TFRecord, preprocess it and return the tensor
	"""
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(serialized_example,
		features={'image_raw':tf.FixedLenFeature([], tf.string),
				  'target':tf.FixedLenFeature([1000], tf.int64),
				  'labels':tf.VarLenFeature(tf.int64),
				  'width':tf.FixedLenFeature([], tf.int64),
				  'height':tf.FixedLenFeature([], tf.int64)})

	# Convert from a scalar string tensor (whose single string has
	#image = tf.decode_raw(features['image_raw'], tf.uint8)
	image = tf.image.decode_jpeg(features['image_raw'], channels=3)
	image.set_shape([inception.inception_v3.default_image_size,
		inception.inception_v3.default_image_size, 3])

	#Using preprocessing from slim
	processed_images = inception_preprocessing.preprocess_image(image,
		inception.inception_v3.default_image_size,
		inception.inception_v3.default_image_size, is_training=False)

	# Convert label from a scalar uint8 tensor to an int32 scalar.
	label = features['labels']
	target = tf.cast(features['target'], tf.int32)

	return processed_images, label, target

def inputs(filename, num_epochs=1000):
	filename_queue = tf.train.string_input_producer([filename],
			   num_epochs=num_epochs)
	image, labels, target = read_and_decode(filename_queue)
	batch_image, batch_labels, batch_targets = tf.train.batch(
		[image, labels, target], batch_size=FLAGS.batch_size)
	# batch_image, batch_labels = tf.train.shuffle_batch(
	#     [image, labels], batch_size=FLAGS.batch_size,
	#     num_threads=4,
	#     capacity=1000 + 3 * FLAGS.batch_size,
	#     min_after_dequeue=1000)
	return batch_image, batch_labels, batch_targets
