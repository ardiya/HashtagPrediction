import numpy as np
import tensorflow as tf
import os
import input
import model
from matplotlib import pyplot as plt

slim = tf.contrib.slim

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_epochs', 2, 'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_string('harrison_dir', '/home/ardiya/HARRISON',
					'Directory containing Benchmark Dataset(images, data_list, and tag_list.')
flags.DEFINE_string('train_dir', '/home/ardiya/HashtagPrediction',
					'Directory with the training data.')
flags.DEFINE_string('train_file', 'harrison.tfrecords',
					'File of the training data')

if __name__ == '__main__':
	tf.reset_default_graph()
	print("Start Session")
	with tf.Session() as sess:
	    
	    print("Building Network")
	    images, sparse_labels = input.inputs()
	    logits = model.inference(images)
	    losses = model.loss(logits, sparse_labels)
	    train_op = model.training(losses)
	    
	    init_op = tf.group(tf.initialize_all_variables(),
	                       tf.initialize_local_variables())
	    sess.run(init_op)
	    
	    print("Creating Queue Runner")
	    coord = tf.train.Coordinator()
	    threads = tf.train.start_queue_runners(coord=coord)
	    
	    print("Start Training")
	    try:
	        it = 0
	#         slim.learning.train(train_op,
	#                    '/home/ardiya/HashtagPrediction/train.log',
	#                    number_of_steps=num_epochs,
	#                    save_summaries_secs=5,
	#                    save_interval_secs=3)
	        while not coord.should_stop():
	            print("Iter #%d" % (it+1), end="")
	            it+=1
	            XData, YData = sess.run([images, sparse_labels])
	            if i==0:
	            	for j in range(3):
			            plt.imshow(XData[j])
			            plt.show()
	        
	    except tf.errors.OutOfRangeError:
	        print('Catch OutOfRangeError')
	    finally:
	        coord.request_stop()
	    print("Done Training")
	    coord.join(threads)
