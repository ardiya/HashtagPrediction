import numpy as np
import tensorflow as tf
import os
import input
import model
from matplotlib import pyplot as plt

slim = tf.contrib.slim

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_epochs', 1000, 'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size', 32, 'Batch size.')
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

		batch_x, batch_y, X, y = input.inputs()
		logits = model.inference(batch_x)
		losses = model.loss(logits, batch_y)
		train_op = model.training(losses)
		init_fn = model.get_init_fn()

		with tf.Session() as sess:
			with slim.queues.QueueRunners(sess):
				
				init_op = tf.group(tf.initialize_all_variables(),
								   tf.initialize_local_variables())
				sess.run(init_op)

				final_loss = slim.learning.train(
						train_op,
						logdir=os.path.join(FLAGS.train_dir, 'train.log'),
						init_fn=init_fn,
						number_of_steps=FLAGS.num_epochs,
						save_summaries_secs=10)
