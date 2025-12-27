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
    """scipy.misc.imresize replacement using PIL"""
    pil_img = Image.fromarray(img.astype(np.uint8))
    if isinstance(size, tuple):
        pil_img = pil_img.resize((size[1], size[0]), Image.BILINEAR)
    else:
        new_h = int(pil_img.height * size)
        new_w = int(pil_img.width * size)
        pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
    return np.array(pil_img)

def main():
	test_path1 = './test_imgs/RGB/'
	test_path2 = './test_imgs/NIR/'
	save_path = './des_results/'
	files = listdir(test_path1)

	# reader = tf.train.NewCheckpointReader('/data/xh/reg+fusion_qikan/task1_des_extract/models2/3600.ckpt')
	# var_to_shape_map = reader.get_variable_to_shape_map()
	# total_parameters = 0
	# for key in var_to_shape_map:  # list the keys of the model
	# 	if 'Adam' not in key:
	# 		print(key)
	# 		# print(reader.get_tensor(key))
	# 		shape = np.shape(reader.get_tensor(key))  # get the shape of the tensor in the model
	# 		shape = list(shape)
	# 		print(shape)
	# 		# print(len(shape))
	# 		variable_parameters = 1
	# 		for dim in shape:
	# 			# print(dim)
	# 			variable_parameters *= dim
	# 		# print(variable_parameters)
	# 		total_parameters += variable_parameters
	# print(total_parameters)

	with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
		model = Des_Extract_Model(BATCH_SIZE=1, INPUT_W=1024, INPUT_H=1024, equivariance=False)
		SOURCE_RGB = tf.compat.v1.placeholder(tf.float32, shape = (1, 1024, 1024, 3), name = 'SOURCE1')
		SOURCE_NIR = tf.compat.v1.placeholder(tf.float32, shape = (1, 1024, 1024, 1), name = 'SOURCE2')

		model.des(SOURCE_RGB, SOURCE_NIR)
		saver = tf.compat.v1.train.Saver(max_to_keep=5)

		var_list_rgb = [v for v in tf.compat.v1.trainable_variables() if 'RGB_Encoder' in v.name]
		var_list_nir = [v for v in tf.compat.v1.trainable_variables() if 'NIR_Encoder' in v.name]
		var_list_descriptor = var_list_rgb + var_list_nir

		initialize_uninitialized(sess)
		_, _ = load(sess, saver, './models')

		for file in files:
			name = file.split('/')[-1]
			print(name)
			rgb_img = imread(test_path1 + name)
			nir_img = imread(test_path2 + name)
			rgb_dimension = list(rgb_img.shape)
			nir_dimension = list(nir_img.shape)
			rgb_img = imresize(rgb_img, (1024, 1024))
			nir_img = imresize(nir_img, (1024, 1024))

			rgb_img = np.expand_dims(rgb_img, axis=0)
			nir_img = np.expand_dims(np.expand_dims(nir_img, axis=0), axis=-1)

			rgb_img = rgb_img.astype(np.float32) / 255.0
			nir_img = nir_img.astype(np.float32) / 255.0

			rgb_des, nir_des =  sess.run([model.RGB_des, model.NIR_des], feed_dict={SOURCE_RGB: rgb_img, SOURCE_NIR: nir_img})
			rgb_des, nir_des = normalize_common(rgb_des, nir_des)

			if not os.path.exists(save_path):
				os.mkdir(save_path)
			if not os.path.exists(save_path + 'RGB/'):
				os.mkdir(save_path + 'RGB/')
			if not os.path.exists(save_path + 'NIR/'):
				os.mkdir(save_path + 'NIR/')
			imsave(save_path + 'RGB/' + name,
							  imresize(rgb_des[0, :, :, 0], (rgb_dimension[0], rgb_dimension[1])))
			imsave(save_path + 'NIR/' + name,
							  imresize(nir_des[0, :, :, 0], (nir_dimension[0], nir_dimension[1])))


if __name__ == '__main__':
	main()


