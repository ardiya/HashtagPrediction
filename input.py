import os
import numpy as np
import tensorflow as tf
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
				  'labels':tf.FixedLenFeature([1000], tf.int64),
				  'width':tf.FixedLenFeature([], tf.int64),
				  'height':tf.FixedLenFeature([], tf.int64)})

	# Convert from a scalar string tensor (whose single string has
	#image = tf.decode_raw(features['image_raw'], tf.uint8)
	image = tf.image.decode_jpeg(features['image_raw'], channels=3)
	image.set_shape([299, 299, 3])
	print("- Image:", image.get_shape())
	
	# # Because these operations are not commutative, consider randomizing
	# # the order their operation.
	# distorted_image = tf.image.random_brightness(image, max_delta=63)
	# print("- Brightness:", distorted_image.get_shape())
	# distorted_image = tf.image.random_contrast(distorted_image,
	#                                          lower=0.2, upper=1.8)

	# Subtract off the mean and divide by the variance of the pixels.
	float_image = tf.image.per_image_whitening(image)
	print("- Whitening:", float_image.get_shape())

	# # Convert from [0, 255] -> [-0.5, 0.5] floats.
	# image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

	# Convert label from a scalar uint8 tensor to an int32 scalar.
	label = tf.cast(features['labels'], tf.int32)

	return float_image, label

def inputs():
	filename=os.path.join(FLAGS.train_dir, FLAGS.train_file)
	filename_queue = tf.train.string_input_producer([filename],
			   num_epochs=FLAGS.num_epochs)
	image, label = read_and_decode(filename_queue)
	images, sparse_labels = tf.train.batch(
		[image, label], batch_size=FLAGS.batch_size)
	# images, sparse_labels = tf.train.shuffle_batch(
	#     [image, label], batch_size=batch_size,
	#     num_threads=2,
	#     capacity=1000 + 3 * batch_size,
	#     min_after_dequeue=1000)
	return images, sparse_labels
