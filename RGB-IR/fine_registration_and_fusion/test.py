"""
TensorFlow 2.x 兼容版本的测试脚本
使用 tf.compat.v1 保持与原代码兼容
"""
from __future__ import print_function
import time
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import tensorflow as tf

# TF2 兼容性设置
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from scipy.io import loadmat
from f2m_model_tf2 import F2M_Model
from utils_tf2 import *
import cv2
from datetime import datetime
import scipy.io as scio
from PIL import Image
from imageio import imsave, imread

def imresize(img, size):
    """替换 scipy.misc.imresize"""
    pil_img = Image.fromarray(img.astype(np.uint8))
    resized = pil_img.resize((size[1], size[0]), Image.BILINEAR)
    return np.array(resized)

N = 1240  # 根据源图像分辨率设置

def main():
    test_path1 = './test_imgs/RGB/'
    test_path2 = './test_imgs/IR/'
    save_path = './results/'
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(save_path + 'fused_img/'):
        os.mkdir(save_path + 'fused_img/')
    if not os.path.exists(save_path + 'compare/'):
        os.mkdir(save_path + 'compare/')
    
    files = listdir(test_path1)
    pic_num = 0
    T = []

    # 使用 tf.compat.v1 API
    with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
        f2m_model = F2M_Model(BATCH_SIZE=1, INPUT_W=N, INPUT_H=N, is_training=False, BN=True)
        
        SOURCE_RGB = tf.compat.v1.placeholder(tf.float32, shape=(1, N, N, 3), name='SOURCE1')
        SOURCE_IR = tf.compat.v1.placeholder(tf.float32, shape=(1, N, N, 1), name='SOURCE2')
        DEFOR_FIELD = tf.compat.v1.placeholder(tf.float32, shape=(1, N, N, 2), name='defor_field')
        RE_DEFOR_FIELD_GT = tf.compat.v1.placeholder(tf.float32, shape=(1, N, N, 2), name='re_defor_field_gt')
        
        f2m_model.f2m(SOURCE_RGB, SOURCE_IR, DEFOR_FIELD, RE_DEFOR_FIELD_GT)

        # 获取变量列表 (TF2 兼容)
        var_list_f2m = tf.compat.v1.trainable_variables(scope='f2m_net')
        g_list = tf.compat.v1.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list_f2m += bn_moving_vars
        
        var_list_bn = []
        var_list_f2m_fuse = []
        for i in var_list_f2m:
            if "BatchNorm" in i.name:
                var_list_bn.append(i)
            if "offset" not in i.name:
                var_list_f2m_fuse.append(i)

        saver_f2m_fuse = tf.compat.v1.train.Saver(var_list_f2m_fuse)
        global_vars = tf.compat.v1.global_variables()
        sess.run(tf.compat.v1.variables_initializer(global_vars))

        # 加载模型
        model_path = './models/finetuning/'
        if os.path.isfile(model_path + 'checkpoint'):
            print("[*] Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(model_path)
            print("ckpt: ", ckpt)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                print("counter: ", ckpt_name)
                saver_f2m_fuse.restore(sess, os.path.join(model_path, ckpt_name))
                print(" [*] Success to read", ckpt_name)
            else:
                print(" [*] Failed to find a checkpoint")
                return
        else:
            print("[*] Failed to find a checkpoint file")
            return

        # 处理每张图像
        for file in files:
            pic_num += 1
            start_time = datetime.now()
            # file 已经是完整路径，提取文件名
            filename = os.path.basename(file)
            name = filename.split('.')[0]
            format = filename.split('.')[1]
            print('[%d/%d]: %s' % (pic_num, len(files), filename))

            rgb_img = imread(file)  # file 已是完整路径
            ir_file = file.replace('/RGB/', '/IR/')
            ir_img_raw = imread(ir_file)
            ir_img = ir_img_raw[:, :, 0] if len(ir_img_raw.shape) == 3 else ir_img_raw

            rgb_dimension = list(rgb_img.shape)
            ir_dimension = list(ir_img.shape)
            H = rgb_dimension[0] * 1.0
            W = rgb_dimension[1] * 1.0

            # resize
            rgb_img_N = imresize(rgb_img, size=(N, N))
            ir_img_N = imresize(ir_img, size=(N, N))
            rgb_img_N = np.expand_dims(rgb_img_N, axis=0)
            ir_img_N = np.expand_dims(np.expand_dims(ir_img_N, axis=0), axis=-1)
            rgb_img_N = rgb_img_N.astype(np.float32) / 255.0
            ir_img_N = ir_img_N.astype(np.float32) / 255.0

            defor_field = np.zeros([1, N, N, 2])
            defor_re = np.zeros_like(defor_field)
            FEED_DICT = {
                SOURCE_RGB: rgb_img_N, 
                SOURCE_IR: ir_img_N, 
                DEFOR_FIELD: defor_field,
                RE_DEFOR_FIELD_GT: defor_re
            }

            fused_img_out = sess.run(f2m_model.fused_img[0, :, :, :], feed_dict=FEED_DICT)
            fused_img_ori_size = cv2.resize(fused_img_out, (rgb_dimension[1], rgb_dimension[0]))
            
            # 归一化到 0-255 并保存
            fused_img_save = np.clip(fused_img_ori_size / np.max(fused_img_ori_size) * 255, 0, 255).astype(np.uint8)
            imsave(save_path + 'fused_img/' + name + '.' + format, fused_img_save)

            elapsed = datetime.now() - start_time
            elapsed = elapsed.total_seconds()

            if pic_num > 1:
                T.append(elapsed)
                print("Elapsed_time: %s" % (T[pic_num - 2]))

    if len(T) > 0:
        print("\nAverage time: %s" % (np.mean(T)))

if __name__ == '__main__':
    main()
