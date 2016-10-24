import input
import model
import tensorflow as tf
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
	with tf.Graph().as_default():
		tf.logging.set_verbosity(tf.logging.INFO)
		
		batch_x, batch_y, X, y = input.inputs()
		logits = model.inference(batch_x, is_training=False)

		batch_y = tf.cast(batch_y, tf.int64)

		names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
			'eval/precision@1': slim.metrics.streaming_sparse_precision_at_k(logits, batch_y, 1),
			'eval/recall@1': slim.metrics.streaming_sparse_recall_at_k(logits, batch_y, 1),
			'eval/precision@5': slim.metrics.streaming_sparse_precision_at_k(logits, batch_y, 5),
			'eval/recall@5': slim.metrics.streaming_sparse_recall_at_k(logits, batch_y, 5),
			'eval/AP@5': slim.metrics.streaming_sparse_average_precision_at_k(logits, batch_y, 5),
		})
		
		logdir = 'train.log'
		checkpoint_path = tf.train.latest_checkpoint(logdir)
		metric_values = slim.evaluation.evaluate_once(
			master='',
			checkpoint_path=checkpoint_path,
			logdir=logdir,
			eval_op=list(names_to_updates.values()),
			final_op=list(names_to_values.values()))

		names_to_values = dict(zip(list(names_to_values.keys()), metric_values))
		for name in names_to_values:
			print('%s: %f' % (name, names_to_values[name]))
