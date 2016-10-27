import os
import input
import model
import tensorflow as tf
slim = tf.contrib.slim

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_string('harrison_dir', '/home/ardiya/HARRISON',
					'Directory containing Benchmark Dataset(img_placeholder, data_list, and tag_list.')
flags.DEFINE_string('train_dir', '/home/ardiya/HashtagPrediction',
					'Directory with the training data.')
flags.DEFINE_string('train_file', 'harrison_train.tfrecords',
					'Filename of the training data')
flags.DEFINE_string('test_file', 'harrison_test.tfrecords',
					'Filename of the test data')

if __name__ == '__main__':
	with tf.Graph().as_default():
		tf.logging.set_verbosity(tf.logging.INFO)
		
		batch_x, batch_labels, batch_y = input.inputs(
			filename=os.path.join(FLAGS.train_dir, FLAGS.test_file),
			num_epochs=1)
		logits = model.inference(batch_x, is_training=False)

		batch_y = tf.cast(batch_y, tf.int64)

		names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
			'accuracy': slim.metrics.streaming_accuracy(logits, batch_y),
			'eval/precision/1': slim.metrics.streaming_sparse_precision_at_k(logits, batch_labels, 1),
			'eval/precision/2': slim.metrics.streaming_sparse_precision_at_k(logits, batch_labels, 2),
			'eval/precision/3': slim.metrics.streaming_sparse_precision_at_k(logits, batch_labels, 3),
			'eval/precision/4': slim.metrics.streaming_sparse_precision_at_k(logits, batch_labels, 4),
			'eval/precision/5': slim.metrics.streaming_sparse_precision_at_k(logits, batch_labels, 5),
			'eval/precision/10': slim.metrics.streaming_sparse_precision_at_k(logits, batch_labels, 10),
			'eval/recall/1': slim.metrics.streaming_sparse_recall_at_k(logits, batch_labels, 1),
			'eval/recall/2': slim.metrics.streaming_sparse_recall_at_k(logits, batch_labels, 2),
			'eval/recall/3': slim.metrics.streaming_sparse_recall_at_k(logits, batch_labels, 3),
			'eval/recall/4': slim.metrics.streaming_sparse_recall_at_k(logits, batch_labels, 4),
			'eval/recall/5': slim.metrics.streaming_sparse_recall_at_k(logits, batch_labels, 5),
			'eval/recall/10': slim.metrics.streaming_sparse_recall_at_k(logits, batch_labels, 10),
			'eval/AP@1': slim.metrics.streaming_sparse_average_precision_at_k(logits, batch_labels, 1),
			'eval/AP@2': slim.metrics.streaming_sparse_average_precision_at_k(logits, batch_labels, 2),
			'eval/AP@3': slim.metrics.streaming_sparse_average_precision_at_k(logits, batch_labels, 3),
			'eval/AP@4': slim.metrics.streaming_sparse_average_precision_at_k(logits, batch_labels, 4),
			'eval/AP@5': slim.metrics.streaming_sparse_average_precision_at_k(logits, batch_labels, 5),
		})
		
		logdir = 'train.log'
		checkpoint_path = tf.train.latest_checkpoint(logdir)
		metric_values = slim.evaluation.evaluate_once(
			master='',
			checkpoint_path=checkpoint_path,
			logdir=logdir,
			num_evals=5000//FLAGS.batch_size,
			eval_op=list(names_to_updates.values()),
			final_op=list(names_to_values.values()))

		names_to_values = dict(zip(list(names_to_values.keys()), metric_values))
		for name in sorted(names_to_values):
			print('%s: %f' % (name, names_to_values[name]))
