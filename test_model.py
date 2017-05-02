import tensorflow as tf
import numpy as np
import pickle
import cv2
import h5py
import argparse

N_POINTS = 68

parser = argparse.ArgumentParser(description='Test trained SDN models')
parser.add_argument('--eval_batch_size', type=int, default=30, help='Evaluation batch size')
parser.add_argument('--show_ground_truth', type=bool, default=False, help='Show ground truth or not')
parser.add_argument('--model_name', help='The name of model you want to test')
parser.add_argument('--half_size',type=bool, default=True, help='If the model is trained with ')

opt = parser.parse_args()


def add_landmarks(image,prediction):
    image=image.astype(np.uint8)
    image = cv2.resize(image,(224, 224))
    # if ground_truth.shape==(136,):
    #     ground_truth=np.reshape(ground_truth,(68,2))
    prediction[:,0] = prediction[:,0]*image.shape[0]
    prediction[:,1] = prediction[:,1]*image.shape[1]
    # ground_truth[:,0] = ground_truth[:,0]*image.shape[0]
    # ground_truth[:,1] = ground_truth[:,1]*image.shape[1]


    for i in range(0, prediction.shape[0]):
        cv2.circle(image, center=(int(prediction[i][1]), int(prediction[i][0])), radius=1, color=(0,255,0), thickness=-1)
    # if opt.show_ground_truth:
    #     for i in range(0, ground_truth.shape[0]):
    #        cv2.circle(image, center=(int(ground_truth[i][1]), int(ground_truth[i][0])), radius=1, color=(0,0,255), thickness=-1)
    return image


def eval_in_batches(data,sess):
# Get all test_loss for a dataset by running it in small batches."""
# Small utility function to evaluate a dataset by feeding batches of data to
# {test_images_node} and pulling the results from {eval_predictions}.
# Saves memory and enables this to run on smaller GPUs.

    size = data.shape[0]
    if size < opt.eval_batch_size:
      raise ValueError("batch size for evals larger than dataset: %d" % size)

    eval_pred = np.ndarray(shape=(size, N_POINTS,2), dtype=np.float32)
    batch_pred = np.ndarray(shape=(opt.eval_batch_size, N_POINTS,2), dtype=np.float32)  

    print('Testing...')
    for begin in range(0, size, opt.eval_batch_size):
        end = begin + opt.eval_batch_size
        if end <= size:
            network_output = sess.run([test_prediction],
            feed_dict={test_images_node: data[begin:end, ...], mode: False})
            eval_pred[begin:end, :] = np.asarray(network_output[0])
        else:
            network_output = sess.run([test_prediction],
            feed_dict={test_images_node: data[-opt.eval_batch_size:, ...],mode: False})
            eval_pred[begin:, :] = np.asarray(network_output[0][begin - size:, :])

    return  eval_pred


########################################################
###############Loading Test Data########################
########################################################
# image_file = h5py.File("./SDN_test_image.h5","r")
# label_file = h5py.File("./SDN_test_label.h5","r")
# test_set_image = image_file['test_set']
# test_set_label = label_file['test_set']

h5_file = h5py.File("/home/wisimage/projet/make_dataset/FCN_VGG/int112/test.h5","r")
test_set_image = h5_file['image_set']


# test_set_image = h5py.File("./data/new_helen_train_image.h5","r")['dataset']
# test_set_label = h5py.File("./data/new_helen_train_label.h5","r")['dataset']


#########################################################
#########################################################
model_path = './model/'+opt.model_name+'.ckpt'
saver = tf.train.import_meta_graph(model_path+'.meta')
with tf.Session() as sess:
    saver.restore(sess, model_path)
    # all_vars = tf.trainable_variables()
    test_heat_map = tf.get_collection('network_output')[0]
    test_prediction = tf.get_collection('predictions')[0]
    test_images_node = tf.get_collection('images_node')[0]
    mode= tf.get_collection('mode')[0]
    prediction = eval_in_batches(test_set_image,sess)


prediction = np.reshape(prediction,(prediction.shape[0],68,2))

i=0
while i < len(test_set_image):
    print(i)
    # # Show images and landmarks
    image_shown = add_landmarks(test_set_image[i], prediction[i])
    cv2.imshow('image',cv2.cvtColor(image_shown, cv2.COLOR_RGB2BGR))
    # cv2.imwrite('./test_results/'+str(i)+'.jpg',cv2.cvtColor(image_shown, cv2.COLOR_RGB2BGR))
    i+=1 
    key = cv2.waitKey(0)
    while key != 1113938 and key!= 1113940:
        key = cv2.waitKey(0)
        continue
    if key==1113938:
        i-=2
        if i<0:
            i=0
