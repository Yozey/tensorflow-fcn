import tensorflow as tf
import fcn32_vgg as fcn32
import fcn16_vgg as fcn16
import fcn8_vgg as fcn8
import h5py
import random
import numpy as np
from six.moves import xrange
import sys
import loss as l
import os

# Dataset Marco
# Specify several information of the dataset
NUM_POINTS = 68
LEN_TRAIN_SET = 1622 # Half of the lfpw dataset, just for test
LEN_TEST_SET = 448
# Sysytem env Marco
# Set a small value if the memory runs out
NUM_THREADS = os.cpu_count()

flags = tf.app.flags
flags.DEFINE_string("model", "fcn32","The fully convolutional model to use, choose between fcn32, fcn16, fcn8")
flags.DEFINE_string("loss_type", "cross_entropy","Loss type, choose between \'cross_entropy\' and \'L2\'")
flags.DEFINE_integer("batch_size",4, "Batch size of the train and test")
flags.DEFINE_integer("num_epoch",20,"Number of the epochs needs to be completed for the training")
flags.DEFINE_string("opt_type", "Adam", "Type of optimizer to use, choose between Adam and SGD")
flags.DEFINE_float("start_learning_rate", 1e-5, "Start learning rate")
flags.DEFINE_float("lr_decay_rate", 0.95, "Exponential learning rate decay rate, give 1 if don't want to decay the lr rate")
flags.DEFINE_integer("lr_decay_interval", 1000, "Exponential learning rate decay interval")
flags.DEFINE_integer("test_freq", 1000, "Validation frequency")
flags.DEFINE_string("comment", "", "Supplementary comment of the model")
flags.DEFINE_integer("num_fc8_neurons", 68, "Define the number of the neurons in the fc8 layer")
flags.DEFINE_boolean("debug",False,"Turn to debug mode (showing more info) if True is given")
flags.DEFINE_string("summaries_dir", "./tmp", "Indicate the place to save tensorboard files")
flags.DEFINE_boolean("int_lbl",True,"Indicate the label is in uint8 data type or not")
flags.DEFINE_boolean("half_lbl_size",True,"Indicate the label is in the half size of 224*224 or not")

FLAGS = flags.FLAGS

if FLAGS.half_lbl_size:
	LBL_SIZE=112
else:
	LBL_SIZE=224


SAVED_FILE_NAME="./model/"+FLAGS.model+"_"+FLAGS.loss_type+"_"+ str(FLAGS.num_epoch)+" epochs_"+FLAGS.opt_type+"_"+FLAGS.comment
VGG_PATH = "./VGG_FACE.npy"

def read_and_decode(filename):
	filename_queue = tf.train.string_input_producer([filename])
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image_raw': tf.FixedLenFeature([], tf.string),
                                           'label_raw' : tf.FixedLenFeature([NUM_POINTS*224*224,], tf.float32),
                                       })

	img = tf.decode_raw(features['image_raw'], tf.uint8)
	img = tf.cast(img, tf.float32)
	img = tf.reshape(img, [3,224, 224])
	lbl = tf.reshape(features['label_raw'], [NUM_POINTS,224, 224])
	return img, lbl

def read_and_decode_intlbl(filename):
	filename_queue = tf.train.string_input_producer([filename])
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image_raw': tf.FixedLenFeature([], tf.string),
                                           'label_raw' : tf.FixedLenFeature([], tf.string),
                                       })

	img = tf.decode_raw(features['image_raw'], tf.uint8)
	img = tf.cast(img, tf.float32)
	img = tf.reshape(img, [3,224, 224])

	lbl = tf.decode_raw(features['label_raw'], tf.uint8)
	lbl = tf.reshape(lbl, [NUM_POINTS,LBL_SIZE, LBL_SIZE])
	return img, lbl

def eval_in_batches(data,label,sess,batch_size):
# Get all test_loss for a dataset by running it in small batches.
# Small utility function to evaluate a dataset by feeding batches of data to
# {test_images_node} and pulling the results from {eval_predictions}.

    size = data.shape[0]
    if size < batch_size:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    eval_loss = np.ndarray(shape=(size), dtype=np.float32)
    batch_loss = np.ndarray(shape=(batch_size), dtype=np.float32)

    eval_pred = np.ndarray(shape=(size, NUM_POINTS,2), dtype=np.int64)
    batch_pred = np.ndarray(shape=(batch_size, NUM_POINTS,2), dtype=np.int64)  

    loss=tf.get_collection('loss')[0]
    predictions=tf.get_collection('predictions')[0]
    images_node=tf.get_collection('images_node')[0]
    labels_node=tf.get_collection('labels_node')[0]
    mode=tf.get_collection('mode')[0]


    print('Testing...')
    for begin in xrange(0, size, batch_size):
        end = begin + batch_size
        if end <= size:
            eval_loss[begin:end], eval_pred[begin:end, ...] = sess.run(
            [loss,predictions],
            feed_dict={images_node: data[begin:end, ...],
                    labels_node: label[begin:end,...],mode: False})
        else:
            batch_loss[:],batch_pred[:,...] = sess.run(
            [loss,predictions],
            feed_dict={images_node: data[-batch_size:, ...],
                    labels_node: label[-batch_size:,...],mode: False})
            eval_loss[begin:] = batch_loss[begin - size:]
            eval_pred[begin:, ...] = batch_pred[begin - size:, ...]

    final_eval_loss = eval_loss.mean()

    print (eval_pred[:,34,0]) # To check if all the redsults for different images are identical
    return  final_eval_loss


# def quick_eval_in_batches(data,label,sess,batch_size,size):
# 	"""
# 	Test the model over the test_set
# 	A quick version means using the TFrecord and FIFO Queue in tensorflow 
# 	"""
# 	batch_size= FLAGS.batch_size
# 	capacity = (NUM_THREADS+1) * batch_size
# 	test_image_batch, test_label_batch = tf.train.batch([data, label], 
# 														batch_size=batch_size, num_threads=NUM_THREADS,
# 														capacity=capacity)

# 	eval_loss = np.ndarray(shape=(size), dtype=np.float32)
# 	batch_loss = np.ndarray(shape=(batch_size), dtype=np.float32)

# 	eval_pred = np.ndarray(shape=(size, NUM_POINTS,2), dtype=np.int64)
# 	batch_pred = np.ndarray(shape=(batch_size, NUM_POINTS,2), dtype=np.int64) 

# 	loss=tf.get_collection('loss')[0]
# 	predictions=tf.get_collection('predictions')[0]
# 	images_node=tf.get_collection('images_node')[0]
# 	labels_node=tf.get_collection('labels_node')[0]
# 	mode=tf.get_collection('mode')[0]
 

# 	print('Testing...')
# 	for begin in xrange(0, size, batch_size):
# 		end = begin + batch_size
# 		print(end)
# 		test_img_batch, test_lbl_batch = sess.run([test_image_batch, test_label_batch])
# 		print('batch got')

# 		if end <= size:
# 			eval_loss[begin:end], eval_pred[begin:end, ...] = sess.run(
# 			[loss,predictions],
# 			feed_dict={images_node: test_img_batch,
# 				labels_node: test_lbl_batch,mode: False})
# 		else:
# 			batch_loss[:],batch_pred[:,...] = sess.run(
# 			[loss,predictions],
# 			feed_dict={images_node: test_img_batch,
# 					labels_node: test_lbl_batch,mode: False})
# 			eval_loss[begin:] = batch_loss[:size-begin]
# 			eval_pred[begin:, ...] = batch_pred[:size-begin, ...]

# 	final_eval_loss = eval_loss.mean()

# 	print (eval_pred[:,34,0]) # To check if all the redsults for different images are identical
# 	return  final_eval_loss


def build_model():

	# Input & Output in [N,C,H,W]

	images_node = tf.placeholder(tf.float32, [None, 3, 224, 224])
	if FLAGS.int_lbl:
		labels_node = tf.placeholder(tf.uint8, [None, NUM_POINTS, LBL_SIZE, LBL_SIZE])
	else:
		labels_node = tf.placeholder(tf.float32, [None, NUM_POINTS, LBL_SIZE, LBL_SIZE])
	mode = tf.placeholder(tf.bool)

	if FLAGS.model == "fcn32":
		fcn = fcn32.FCN32VGG(VGG_PATH)
	elif FLAGS.model == "fcn16":
		fcn = fcn16.FCN16VGG(VGG_PATH)
	elif FLAGS.model == "fcn8":
		fcn = fcn8.FCN8VGG(VGG_PATH)
	else:
		raise ValueError('Please pick a network structure among fcn8, fcn16 and fcn32')

	fcn.build(images_node, train=mode, num_classes=FLAGS.num_fc8_neurons, num_points=NUM_POINTS, 
				random_init_fc8=True, debug=FLAGS.debug, half_size_label=FLAGS.half_lbl_size)
	if FLAGS.model == "fcn32":
		network_output = fcn.upscore
	else:
		network_output = fcn.upscore32
	tf.add_to_collection('network_output', network_output)
	tf.add_to_collection('images_node', images_node)
	tf.add_to_collection('labels_node', labels_node)
	tf.add_to_collection('mode',mode)

	loss = l.loss(network_output,labels_node,FLAGS.loss_type,FLAGS.int_lbl)
	predictions = l.get_predictions(network_output)
	tf.add_to_collection('loss',loss)
	tf.add_to_collection('predictions',predictions)

	tf.summary.scalar('loss', tf.reduce_mean(loss))



def train_model(train_images, train_labels, test_images, test_labels):
	# Data preparation
	batch_size= FLAGS.batch_size
	min_after_dequeue = 10
	capacity = min_after_dequeue + (NUM_THREADS+1) * batch_size
	image_batch, label_batch = tf.train.shuffle_batch([train_images, train_labels], 
														batch_size=batch_size, num_threads=NUM_THREADS,
														capacity=capacity,min_after_dequeue=min_after_dequeue,
														enqueue_many=False)
	# Take the node from the graph
	train_images_node=tf.get_collection('images_node')[0]
	train_labels_node=tf.get_collection('labels_node')[0]
	train_mode=tf.get_collection('mode')[0]
	loss=tf.get_collection('loss')[0]
	predictions=tf.get_collection('predictions')[0]
	# Define the learning rate
	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(FLAGS.start_learning_rate, global_step,
												FLAGS.lr_decay_interval, FLAGS.lr_decay_rate, staircase=True)
	# Define optimizer
	assert (FLAGS.opt_type == "Adam" or FLAGS.opt_type == "SGD"), "Unsupported optimizer type, choose between Adam and SGD"
	if FLAGS.opt_type == "SGD":
		opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(tf.reduce_mean(loss),global_step=global_step)
	else:
		opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(tf.reduce_mean(loss),global_step=global_step)
	init_op = tf.global_variables_initializer()

	saver = tf.train.Saver()

	with tf.Session() as sess:
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess,coord=coord)

		# Tensorboard setting
		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',sess.graph)
		test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')

		sess.run(init_op)
		check=0 # Checkpoint counter to save the checkpoint file after each epoch

		for i in xrange(int(LEN_TRAIN_SET/batch_size*FLAGS.num_epoch)):
			check+=1
			img_batch, lbl_batch = sess.run([image_batch, label_batch])
			feed_dict = {train_images_node:img_batch,train_labels_node:lbl_batch,train_mode:True}
			_, summary, batch_loss,results,lr = sess.run([opt,merged,loss,predictions,learning_rate], feed_dict=feed_dict)
			batch_loss_in_float = sess.run(tf.reduce_mean(batch_loss))
			print('Minibatch loss: %.10f' % (batch_loss_in_float))
			train_writer.add_summary(summary, i)
			if i % FLAGS.test_freq == 0:
				eval_loss = eval_in_batches(test_images,test_labels,sess,batch_size)
				train_writer.add_summary(summary, i)
				print('Validation loss: %.10f' % eval_loss)
				print('Learning rate: %.8f' %(lr))
				sys.stdout.flush()
			# Save the model for each epoch
			if (check>=LEN_TRAIN_SET):
				saver.save(sess,SAVED_FILE_NAME+"_checkpoint.ckpt")
				check=0

		eval_loss = eval_in_batches(test_set_image,test_set_label,sess,batch_size)
		print('Validation loss: %.10f' % eval_loss)

		saver.save(sess,SAVED_FILE_NAME+".ckpt")
		print ('Model saved')

		coord.request_stop()
		coord.join(threads)
	sess.close()

def main(_):
	
	# Test data preparation
	test_images = h5py.File("/home/yongzhe/Project/make_dataset/FCN_VGG/int112/test.h5","r")['image_set']
	test_labels = h5py.File("/home/yongzhe/Project/make_dataset/FCN_VGG/int112/test.h5","r")['label_set']
	# Train data preparation
	train_images, train_labels = read_and_decode_intlbl("/home/yongzhe/Project/make_dataset/FCN_VGG/int112/train.tfrecords")
	# Build the tensorflow graph and Train the model
	build_model()
	train_model(train_images, train_labels, test_images, test_labels)

if __name__ == '__main__':
	tf.app.run()