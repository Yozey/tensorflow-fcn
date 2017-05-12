import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.nan)
import pickle
import cv2
import h5py
import argparse
from scipy.ndimage.measurements import center_of_mass
import matplotlib.pyplot as plt

N_POINTS = 68

parser = argparse.ArgumentParser(description='Test trained SDN models')
parser.add_argument('--eval_batch_size', type=int, default=20, help='Evaluation batch size')
parser.add_argument('--show_ground_truth', type=bool, default=False, help='Show ground truth or not')
parser.add_argument('--model_name', help='The name of model you want to test')
parser.add_argument('--half_size', type=bool, default=True, help='If the model is trained with half_sized label')

opt = parser.parse_args()


def add_landmarks(image,prediction):
    image=image.astype(np.uint8)
    image=np.transpose(image,(1,2,0))
    image = cv2.resize(image,(224, 224))
    # if ground_truth.shape==(136,):
    #     ground_truth=np.reshape(ground_truth,(68,2))


    for i in range(0, prediction.shape[0]):
        cv2.circle(image, center=(int(prediction[i][1]), int(prediction[i][0])), radius=1, color=(0,255,0), thickness=-1)
    # if opt.show_ground_truth:
    #     for i in range(0, ground_truth.shape[0]):
    #        cv2.circle(image, center=(int(ground_truth[i][1]), int(ground_truth[i][0])), radius=1, color=(0,0,255), thickness=-1)
    return image

def get_normalized_inter_ocular_distance(predictions,ground_truths):
    """
    Get the distance normalized by inter-ocular distance

    Args:
    1. Predictions from the network shape: (N,N_POINTS,2)
    2. Ground truth  shape: (N,N_POINTS,2)

    Return:
    The normalized inter-ocular distance
    """
    distances = np.linalg.norm(predictions-ground_truths, axis= -1) # shape: (N,N_POINTS)
    inter_ocular_distances = np.linalg.norm(ground_truths[:,37,:]-ground_truths[:,46,:], axis= -1,keepdims=True) # shape: (N,1)
    return (np.mean(distances/inter_ocular_distances))


def get_predictions_float(np_heatmaps,threshold=0.95):
    """
    Get the point coordinates prediction result in float from the heatmaps

    Args:
        network_output: tensor, float32 - [N,C,H,W]
    
    Returns:
        result: tensor, float32 - [N,C,2]
    """
    N = np_heatmaps.shape[0]
    C = np_heatmaps.shape[1]
    predictions = np.zeros((N,C,2))

    

    for n_batch in range(N):
        for n_channel in range(C):
            # Prevent negative maximum value
            np_heatmaps[n_batch,n_channel]-=np.amin(np_heatmaps[n_batch,n_channel])

            max_value = np.amax(np_heatmaps[n_batch,n_channel])
            low_value_indices=np_heatmaps[n_batch,n_channel]<(threshold*max_value)
            np_heatmaps[n_batch,n_channel][low_value_indices]=0
            predictions[n_batch,n_channel]=np.array(center_of_mass(np_heatmaps[n_batch,n_channel]))

    return predictions


def eval_in_batches(data,sess):
# Get all test_loss for a dataset by running it in small batches."""
# Small utility function to evaluate a dataset by feeding batches of data to
# {test_images_node} and pulling the results from {eval_predictions}.
# Saves memory and enables this to run on smaller GPUs.

    size = data.shape[0]
    if size < opt.eval_batch_size:
      raise ValueError("batch size for evals larger than dataset: %d" % size)

    eval_pred = np.ndarray(shape=(size, N_POINTS,2), dtype=np.float32)

    if opt.half_size:
        lbl_size = 112
    else:
        lbl_size = 224

    eval_heatmap = np.ndarray(shape=(size, N_POINTS,lbl_size,lbl_size), dtype=np.float32)

    print('Testing...')
    for begin in range(0, size, opt.eval_batch_size):
        end = begin + opt.eval_batch_size
        if end <= size:
            network_output,network_output_heatmap = sess.run([test_prediction,test_heat_map],
            feed_dict={test_images_node: data[begin:end, ...], mode: False})

            # eval_pred[begin:end, :] = np.asarray(network_output)

            eval_pred[begin:end, :] = get_predictions_float(network_output_heatmap)


            # eval_heatmap[begin:end, :] = np.asarray(network_output_heatmap)
        else:
            network_output, network_output_heatmap = sess.run([test_prediction,test_heat_map],
            feed_dict={test_images_node: data[-opt.eval_batch_size:, ...],mode: False})
            # eval_pred[begin:, :] = np.asarray(network_output[begin - size:, :])

            eval_pred[begin:, :] = get_predictions_float(network_output_heatmap[begin - size:, :])

            # eval_heatmap[begin:, :] = np.asarray(network_output_heatmap[begin - size:, :])

    return  eval_pred,network_output_heatmap


########################################################
###############Loading Test Data########################
########################################################

h5_file = h5py.File("/home/yongzhe/Project/make_dataset/heatmap/int112/test.h5","r")
test_set_image = h5_file['image_set']
test_set_label_heatmap = h5_file['label_set']

cor_h5_file = h5py.File("/home/yongzhe/Project/make_dataset/coordiantes/tensorflow/test.h5","r")
test_set_label = cor_h5_file['label_set']

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
    prediction, heatmap = eval_in_batches(test_set_image,sess)


test_labels = np.reshape(test_set_label,(test_set_label.shape[0],68,2))

# debug 4,7
# for i in range(112):
#     for j in range(112):
#         if np.isnan(heatmap[4,7,i,j]):
#             print('nan found')
#             print(i)
#             print(j)

max_value = np.amax(heatmap[4,7])
low_value_indices=heatmap[4,7]<(0.95*max_value)
heatmap[4,7][low_value_indices]=0
pred_4_7=np.array(center_of_mass(heatmap[4,7]))

# Change the source code of model to exchange the position of x and y
# prediction_0 = np.expand_dims(prediction[...,0],axis=-1)
# prediction_1 = np.expand_dims(prediction[...,1],axis=-1)
# prediction = np.concatenate((prediction_1,prediction_0),axis=-1)

if opt.half_size:
    prediction*=2

print('Inter-ocular distance: %.5f' % get_normalized_inter_ocular_distance(prediction,test_labels*224))

i=0
while i < len(test_set_image):
    print(i)
    # # Show images and landmarks
    image_shown = add_landmarks(test_set_image[i], prediction[i])
    cv2.imshow('image',cv2.cvtColor(image_shown, cv2.COLOR_RGB2BGR))
    # cv2.imwrite('./test_results/'+str(i)+'.jpg',cv2.cvtColor(image_shown, cv2.COLOR_RGB2BGR))


    # Visualize Heatmap

    # for j in range(N_POINTS):
    #     plt.subplot(121)
    #     plt.imshow(heatmap[i][j])
    #     plt.subplot(122)
    #     plt.imshow(test_set_label_heatmap[i][j])
    #     plt.show()

    i+=1 
    key = cv2.waitKey(0)
    while key != 1113938 and key!= 1113940:
        key = cv2.waitKey(0)
        continue
    if key==1113938:
        i-=2
        if i<0:
            i=0
