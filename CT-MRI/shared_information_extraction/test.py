from __future__ import print_function
import time
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from scipy.io import loadmat
from train import train_descriptor
from des_extract_model import Des_Extract_Model
from utils import *
from imageio import imread, imsave
from PIL import Image
import cv2

def imresize(img, size):
    """imresize replacement using PIL"""
    pil_img = Image.fromarray(img.astype(np.uint8))
    if isinstance(size, tuple):
        pil_img = pil_img.resize((size[1], size[0]), Image.BILINEAR)
    else:
        new_h = int(pil_img.height * size)
        new_w = int(pil_img.width * size)
        pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
    return np.array(pil_img)

N=256

def main():
	test_path1 = './test_imgs/CT/'
	test_path2 = './test_imgs/MRI/'
	save_path = './des_results/'
	model_path ='./models/'

	files = listdir(test_path1)

	with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
		model = Des_Extract_Model(BATCH_SIZE=1, INPUT_W=N, INPUT_H=N, is_training=False, equivariance=False)
		SOURCE_CT = tf.compat.v1.placeholder(tf.float32, shape = (1, N, N, 1), name = 'SOURCE1')
		SOURCE_MRI = tf.compat.v1.placeholder(tf.float32, shape = (1, N, N, 1), name = 'SOURCE2')
		model.des(SOURCE_CT, SOURCE_MRI)
		saver = tf.compat.v1.train.Saver(max_to_keep=5)

		var_list_CT = [v for v in tf.compat.v1.trainable_variables() if 'CT_Encoder' in v.name]
		var_list_MRI = [v for v in tf.compat.v1.trainable_variables() if 'MRI_Encoder' in v.name]
		var_list_descriptor = var_list_CT + var_list_MRI
		initialize_uninitialized(sess)
		_, _ = load(sess, saver, model_path)

		for file in files:
			name = file.split('/')[-1]
			print(name)
			CT_img = imread(test_path1 + name)
			MRI_img = imread(test_path2 + name)
			CT_dimension = list(CT_img.shape)
			MRI_dimension = list(MRI_img.shape)
			CT_img = imresize(CT_img, (N, N))
			MRI_img = imresize(MRI_img, size=(N, N))
			CT_img = np.expand_dims(np.expand_dims(CT_img, axis=0),axis=-1)
			MRI_img = np.expand_dims(np.expand_dims(MRI_img, axis=0), axis=-1)
			CT_img = CT_img.astype(np.float32) / 255.0
			MRI_img = MRI_img.astype(np.float32) / 255.0

			CT_des, MRI_des =  sess.run([model.CT_des, model.MRI_des], feed_dict={SOURCE_CT: CT_img, SOURCE_MRI: MRI_img})
			CT_des, MRI_des = normalize_common(CT_des, MRI_des)

			if not os.path.exists(save_path):
				os.mkdir(save_path)
			if not os.path.exists(save_path + '/CT/'):
				os.mkdir(save_path + '/CT/')
			if not os.path.exists(save_path + '/MRI/'):
				os.mkdir(save_path + '/MRI/')
			imsave(save_path + 'CT/' + name, imresize(CT_des[0, :, :, 0], (CT_dimension[0], CT_dimension[1])))
			imsave(save_path + 'MRI/' + name, imresize(MRI_des[0, :, :, 0], (MRI_dimension[0], MRI_dimension[1])))

if __name__ == '__main__':
	main()


