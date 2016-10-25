import pandas as pd
import numpy as np
import tensorflow as tf
import os
#from matplotlib import pyplot as plt

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('harrison_dir', '/home/ardiya/HARRISON',
					'Directory containing Benchmark Dataset(images, data_list, and tag_list.')
flags.DEFINE_string('train_dir', '/home/ardiya/HashtagPrediction',
					'Directory with the training data.')
flags.DEFINE_string('train_file', 'harrison_train.tfrecords',
					'Filename of the training data')
flags.DEFINE_string('test_file', 'harrison_test.tfrecords',
					'Filename of the test data')

config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True

def multi_encode(ar, num_classes):
	"""
	Return the multiclass-label of hashtags labels into array with length of num_classes,
		  where the i-th element equal to 1 if i exist in the ar
	eg, multi_encode([1,3,4], 10)
		will return [0, 1, 0, 1, 1, 0, 0, 0, 0, 0]
	"""
	ret = np.zeros([num_classes], dtype=int)
	ret[ar] = 1
	return ret.tolist()

def _int64_feature(value):
	"""
	Helper to define int64 feature of TFRecords
	"""
	if not isinstance(value, list):
		value = [value]
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
	"""
	Helper to define bytes feature of TFRecords
	"""
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_tag_mapping():
	# Create Dictionary mapping bidirection name and id
	vocab_index = pd.read_csv(os.path.join(FLAGS.harrison_dir, "vocab_index.txt"),
		#sep=(r' +'), 
		delim_whitespace=True,
		names=['tag_name', 'tag_id'],
		dtype={'tag_name':str, 'tag_id':int})
	dict_id_name = vocab_index["tag_name"].to_dict()
	dict_name_id = {name: idx for idx, name in dict_id_name.items()}
	return dict_id_name, dict_name_id

def read_image_dir_and_tags():
	"""
	Read:
		- List of file directory
		- Tags for the 
	Return pd.DataFrame:
		- harrison_data["image"] = List of the full path of the directory
		- harrison_data["tags"] = The multi-class label
	"""
	data_list = pd.read_csv(os.path.join(FLAGS.harrison_dir, "data_list.txt"), names=['image'])
	tag_list = pd.read_csv(os.path.join(FLAGS.harrison_dir, "tag_list.txt"), names=['tags'])
	# concat the data from data_list.txt and tag_list.txt into one pandas.Dataframe
	harrison_data = pd.concat([data_list, tag_list], axis=1)
	del data_list, tag_list
	# remove broken image [25526 instagram_dataset/sun/image_9.jpg,50762 family/image_2411.jpg]
	harrison_data.drop(harrison_data.index[[25526,50762]], inplace=True)
	
	print("Total Data:", len(harrison_data))

	dict_id_name, dict_name_id = get_tag_mapping()

	#Convert tags into list of vocab index
	harrison_data["tags"] = harrison_data["tags"].str.strip()
	harrison_data["tags"] = harrison_data["tags"].str.split(" ")
	harrison_data["tags"] = [[int(dict_name_id[tag]) for tag in tags]
							for tags in harrison_data["tags"].values]
	harrison_data["target"] = [multi_encode(row,1000) for row in harrison_data["tags"].values]
	harrison_data["image"] = [os.path.join(FLAGS.harrison_dir, path) for path in harrison_data.image]

	#Shuffle DataFrame
	harrison_data = harrison_data.reindex(np.random.permutation(harrison_data.index))
	#Split to train and 
	harrison_test = harrison_data.sample(n=5000)
	harrison_train = harrison_data.loc[~harrison_data.index.isin(harrison_test.index)]
	return harrison_train, harrison_test

def _is_png(filename):
	return '.png' in filename

def _process_image(filename):
	"""
	Process a single image file.
	This method is taken from 'convert_to_records.py' but with a fix output of 299,299
		to be inputted to Inception model
	Args:
	filename: string, path to an image file e.g., '/path/to/example.JPG'.
	Returns:
	image_buffer: string, JPEG encoding of RGB image.
	height: integer, image height in pixels.
	width: integer, image width in pixels.
	"""
	with open(filename, 'rb') as f:
 		image_data = f.read()

	g = tf.Graph()
	with g.as_default():
		tf.logging.set_verbosity(tf.logging.ERROR)
		sess = tf.Session(config=config)
		with tf.device('/cpu:0'):
			img_encoded = tf.placeholder(dtype=tf.string)
			if _is_png(filename):
				t_image = tf.image.decode_png(img_encoded, channels=3)
			else:    
				t_image = tf.image.decode_jpeg(img_encoded, channels=3)
			resized_image = tf.image.resize_images(t_image, [299, 299])
			resized_image = tf.cast(resized_image, tf.uint8)
			encoded_image = tf.image.encode_jpeg(resized_image, format='rgb', quality=100)
		
		sess.run(tf.initialize_all_variables())
		img = sess.run(resized_image, feed_dict={img_encoded: image_data})
		# print("img shape:", img.shape)
		# plt.imshow(img)
		# plt.show()
		
		# Check that image converted to RGB
		assert len(resized_image.get_shape()) == 3
		height = int(resized_image.get_shape()[0])
		width = int(resized_image.get_shape()[1])
		assert resized_image.get_shape()[2] == 3
		
		img = sess.run(encoded_image, feed_dict={img_encoded: image_data})
		sess.close()
		
		return img, height, width

def create_records(harrison_data, filename):
	tf.reset_default_graph()
	
	with tf.Session(config=config) as sess:
		init_op = tf.initialize_all_variables()
		sess.run(init_op)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		
		writer = tf.python_io.TFRecordWriter(filename)
		
		it = 0
		for idx, row in harrison_data.iterrows():
			it += 1
			filename = row['image']
			image_buffer, height, width = _process_image(filename)
			print("Processing Image #",it,"-", filename)
			example = tf.train.Example(features=tf.train.Features(feature={
						'height':_int64_feature(height),
						'width':_int64_feature(width),
						'depth':_int64_feature(3),
						'target':_int64_feature(row['target']),
						'labels':_int64_feature(row['tags']),
						'image_raw':_bytes_feature(image_buffer)
					}))
			writer.write(example.SerializeToString())
		print("\rDone")
		writer.close()
		coord.request_stop()
		coord.join(threads)

if __name__ == '__main__':
	np.random.seed(42)
	harrison_train, harrison_test = read_image_dir_and_tags()
	print("Harrison Train:", harrison_train.shape)
	print("Harrison Test:", harrison_test.shape)
	print("========================")
	create_records(harrison_train, os.path.join(FLAGS.train_dir, FLAGS.train_file))
	create_records(harrison_test, os.path.join(FLAGS.train_dir, FLAGS.test_file))


