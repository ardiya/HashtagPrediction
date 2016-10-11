import numpy as np
import tensorflow as tf
import os
import input
import model
from matplotlib import pyplot as plt

slim = tf.contrib.slim

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_epochs', 10, 'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size', 16, 'Batch size.')
flags.DEFINE_string('harrison_dir', '/home/ardiya/HARRISON',
					'Directory containing Benchmark Dataset(img_placeholder, data_list, and tag_list.')
flags.DEFINE_string('train_dir', '/home/ardiya/HashtagPrediction',
					'Directory with the training data.')
flags.DEFINE_string('train_file', 'harrison.tfrecords',
					'File of the training data')

if __name__ == '__main__':
	tf.reset_default_graph()
	with tf.Graph().as_default():
		tf.logging.set_verbosity(tf.logging.INFO)

		print("Building Network")
		img_placeholder = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))
		lbl_placeholder = tf.placeholder(tf.float32, shape=(None, 1000))

		batch_x, batch_y, X, y = input.inputs()
		logits = model.inference(batch_x)
		losses = model.loss(logits, batch_y)
		train_op = model.training(losses)

		# init_fn = slim.assign_from_checkpoint_fn(
		# 	os.path.join(checkpoints_dir, 'inception_v3.ckpt'),
		# 	slim.get_model_variables('InceptionV3'))
		init_fn = model.get_init_fn()

		print("Start Session")
		sess = tf.Session()
		print("Restoring Inception model")
		init_fn(sess)
		init_op = tf.group(tf.initialize_all_variables(),
						   tf.initialize_local_variables())
		sess.run(init_op)
		
		print("Creating Queue Runner")
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		
		print("Start Training")
		try:
			it = 0
			while not coord.should_stop():
				it+=1
				b_x, b_y = sess.run([batch_x, batch_y])

				# Execute training operation
				_, l = sess.run([train_op, losses], feed_dict={img_placeholder:b_x, lbl_placeholder:b_y})

				print("\rIteration-%5d: batch loss = %.8f"%(it, l), end="\n" if it % 25 == 0 else "")

		except tf.errors.OutOfRangeError:
			print('\nCatch OutOfRangeError')
		finally:
			coord.request_stop()
		print("\nDone Training")
		coord.join(threads)
