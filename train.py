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
LEN_TRAIN_SET = 12165 # Half of the lfpw dataset, just for test

# Sysytem env Marco
# Set a small value if the memory runs out
NUM_THREADS = os.cpu_count()

flags = tf.app.flags
flags.DEFINE_string("model", "fcn32","The fully convolutional model to use, choose between fcn32, fcn16, fcn8")
flags.DEFINE_string("loss_type", "cross_entropy","Loss type, choose between \'cross_entropy\' and \'L2\'")
flags.DEFINE_integer("batch_size",4, "Batch size of the train and test")
flags.DEFINE_integer("num_epoch",20,"Number of the epochs needs to be completed for the training")
flags.DEFINE_string("opt_type", "Adam", "Type of optimizer to use, choose between Adam and SGD")
flags.DEFINE_float("lr_decay_rate", 0.95, "Exponential learning rate decay rate, give 1 if don't want to decay the lr rate")
flags.DEFINE_integer("lr_decay_interval", 1000, "Exponential learning rate decay interval")
flags.DEFINE_integer("test_freq", 1000, "Validation frequency")
flags.DEFINE_string("comment", "", "Supplementary comment of the model")
flags.DEFINE_integer("num_fc8_neurons", 1000, "Define the number of the neurons in the fc8 layer")
flags.DEFINE_boolean("debug",False,"Turn to debug mode (showing more info) if True is given")


FLAGS = flags.FLAGS


SAVED_FILE_NAME="./model/"+FLAGS.model+"_"+FLAGS.loss_type+"_"+ str(FLAGS.num_epoch)+" epochs_"+FLAGS.opt_type+"_"+FLAGS.comment+".ckpt"
VGG_PATH = "./vgg16.npy"

def read_and_decode(filename):
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
    lbl = tf.decode_raw(features['label_raw'], tf.float16)
    lbl = tf.cast(lbl, tf.float32)
    lbl = tf.reshape(lbl, [NUM_POINTS,224, 224])
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

    print('Testing...')
    for begin in xrange(0, size, batch_size):
        end = begin + batch_size
        if end <= size:
            eval_loss[begin:end], eval_pred[begin:end, ...] = sess.run(
            [loss,predictions],
            feed_dict={images_node: data[begin:end, ...],
                    labels_node: label[begin:end,...],train_mode: False})
        else:
            batch_loss[:],batch_pred[:,...] = sess.run(
            [loss,predictions],
            feed_dict={images_node: data[-batch_size:, ...],
                    labels_node: label[-batch_size:,...],train_mode: False})
            eval_loss[begin:] = batch_loss[begin - size:]
            eval_pred[begin:, ...] = batch_pred[begin - size:, ...]

    final_eval_loss = eval_loss.mean()

    print (eval_pred[:,34,0]) # To check if all the redsults for different images are identical
    return  final_eval_loss

def build_model():

	# Input & Output in [N,C,H,W]

	images_node = tf.placeholder(tf.float32, [None, 3, 224, 224])
	labels_node = tf.placeholder(tf.float32, [None, NUM_POINTS, 224, 224])

	if FLAGS.model = "fcn32":
		fcn = fcn32.FCN32VGG(VGG_PATH)
		network_output = self.upscore
	elif FLAGS.model = "fcn16":
		fcn = fcn16.FCN32VGG(VGG_PATH)
		network_output = self.upscore32
	elif FLAGS.model = "fcn8":
		fcn = fcn8.FCN32VGG(VGG_PATH)
		network_output = self.upscore32
	else:
		raise ValueError('Please pick a network structure among fcn8, fcn16 and fcn32')

	fcn.build(images_node, train=train_mode, num_classes=FLAGS.num_fc8_neurons, num_points=NUM_POINTS, 
				random_init_fc8=True, debug=FLAGS.debug)
	saver = tf.train.Saver()

	tf.add_to_collection('network_output', network_output)
	tf.add_to_collection('network_input', images_node)
	tf.add_to_collection('train_mode',train_mode)

	loss = l.loss(network_output,labels_node,FLAGS.loss_type)
	predictions = l.get_predictions(network_output)

	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(FLAGS.start_learning_rate, global_step,
												FLAGS.lr_decay_interval, FLAGS.lr_decay_rate, staircase=True)

	assert (FLAGS.opt_type == "Adam" or FLAGS.opt_type == "SGD"), "Unsupported optimizer type, choose between Adam and SGD"
	if FLAGS.opt_type == "SGD":
		opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(tf.reduce_mean(loss),global_step=global_step)
	else
		opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(tf.reduce_mean(loss),global_step=global_step)

	init_op = tf.global_variables_initializer()

def train_model(train_images, train_labels, test_images, test_labels):
	batch_size= FLAGS.batch_size
	min_after_dequeue = 1000
	capacity = min_after_dequeue + (NUM_THREADS+1) * batch_size
	image_batch, label_batch = tf.train.shuffle_batch([train_images, train_labels], 
														batch_size=batch_size, num_threads=NUM_THREADS,
														capacity=capacity,min_after_dequeue=min_after_dequeue)

	with tf.Session() as sess:

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess,coord=coord)
		sess.run(init_op)

		for i in xrange(int(LEN_TRAIN_SET/batch_size*FLAGS.num_epoch)):
	    
			img_batch, lbl_batch = sess.run([image_batch, label_batch])
			feed_dict = {images_node: img_batch, labels_node: lbl_batch,train_mode: True}
			_, batch_loss,results,lr = sess.run([opt,loss,predictions,learning_rate], feed_dict=feed_dict)
			print('Minibatch loss: %.5f' % (tf.reduce_mean(batch_loss)))
			if i % FLAGS.test_freq == 0:
				eval_loss = eval_in_batches(test_images,test_labels,sess)
				print('Validation loss: %.5f' % eval_loss)
				print('Learning rate: %.5f' %(lr))
				sys.stdout.flush()
		eval_loss = eval_in_batches(test_set_image,test_set_label,sess)
		print('Validation loss: %.5f' % eval_loss)

		saver.save(sess,SAVED_FILE_NAME)
		print ('Model saved')

		coord.request_stop()
		coord.join(threads)
	sess.close()

def main(_):

	# Test data preparation
	test_images = h5py.File("./300w_test_FCN_image.h5","r")['test_imagess']
	test_labels = h5py.File("./300w_test_FCN_image.h5","r")['test_labels']
	# Train data preparation
	train_images, train_labels = read_and_decode("./train.tfrecords")
	# Build the tensorflow graph
	build_model()
	# Train the model
	train_model(train_images, train_labels, test_images, test_labels)

if __name__ == '__main__':
	tf.app.run()