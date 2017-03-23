import tensorflow as tf
import fcn32_vgg as fcn32
import fcn16_vgg as fcn16
import fcn8_vgg as fcn8
import h5py
import random
import numpy as np
from six.moves import xrange
import sys

NUM_POINT = 68


flags = tf.app.flags
flags.DEFINE_string("model", "fcn32","The fully convolutional model to use, choose between fcn32, fcn16, fcn8")

FLAGS = flags.FLAGS

def build_model(model_name):

	images_node = tf.placeholder(tf.float32, [None, 224, 224, 3])
	labels_node = tf.placeholder(tf.float32, [None, 224, 224, NUM_POINT])

	if model_name = "fcn32":
		fcn = fcn32.FCN32VGG()
		network_output = self.upscore
	elif model_name = "fcn16":
		fcn = fcn16.FCN32VGG()
		network_output = self.upscore32
	elif model_name = "fcn8":
		fcn = fcn8.FCN32VGG()
		network_output = self.upscore32
	else:
		raise ValueError('Please pick a network structure among fcn8, fcn16 and fcn32')

	fcn.build(images_node, train_mode)
	saver = tf.train.Saver()

	tf.add_to_collection('network_output', network_output)
	tf.add_to_collection('network_input', images_node)
	tf.add_to_collection('train_mode',train_mode)

	loss = tf.nn.l2_loss(tf.subtract(network_output ,labels_node))/(2*N_POINTS)/BATCH_SIZE
	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(START_LEARNING_RATE, global_step,1000, 0.95, staircase=True)
	# opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step)
	init_op = tf.global_variables_initializer()

def train_model(image_batch, label_batch):
	with tf.Session() as sess:

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess,coord=coord)
		sess.run(init_op)

		for i in xrange(int(295470/BATCH_SIZE*NUM_EPOCHS)):
	    
			img_batch, lbl_batch = sess.run([image_batch, label_batch])
			feed_dict = {images_node: img_batch, labels_node: lbl_batch,train_mode: True}
			_, batch_loss,train_output,lr = sess.run([opt,loss,network_output,learning_rate], feed_dict=feed_dict)
			print('Minibatch loss: %.5f' % (batch_loss))
			if i % TEST_FREQ == 0:
				eval_loss = eval_in_batches(test_set_image,test_set_label,sess)
				print('Validation loss: %.5f' % eval_loss)
				print('Learning rate: %.5f' %(lr))
				sys.stdout.flush()
		eval_loss = eval_in_batches(test_set_image,test_set_label,sess)
		print('Validation loss: %.5f' % eval_loss)

		saver.save(sess,"./model/VGG_fine_tuning_adam_10e.ckpt")
		print ('Model saved')

		coord.request_stop()
		coord.join(threads)
	sess.close()

def main(_):

	# Test data preparation
	test_images = h5py.File("./300w_test_FCN_image.h5","r")['test_images']
	test_labels = h5py.File("./300w_test_FCN_image.h5","r")['test_labels']

	# Train data preparation
	img, label = read_and_decode("./train.tfrecords")
	min_after_dequeue = 1000
	capacity = min_after_dequeue + (NUM_THREADS+1) * BATCH_SIZE
	image_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=BATCH_SIZE, num_threads=NUM_THREADS,capacity=capacity,min_after_dequeue=min_after_dequeue)

	build_model(FLAGS.model)

	train_model(image_batch, label_batch)

if __name__ == '__main__':
	tf.app.run()