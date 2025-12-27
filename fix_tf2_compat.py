#!/usr/bin/env python3
"""
TensorFlow 1.x -> TensorFlow 2.x 兼容性修复脚本
将TF1代码自动转换为TF2兼容代码
"""

import os
import re
import sys

def fix_tf2_compatibility(content):
    """修复TF1代码使其在TF2下运行"""
    
    # 添加 tf.compat.v1 模式
    if 'tf.compat.v1.disable_eager_execution()' not in content:
        # 在 import tensorflow 后添加
        content = re.sub(
            r'(import tensorflow as tf\n)',
            r'\1tf.compat.v1.disable_eager_execution()\n',
            content
        )
    
    # tf.Session -> tf.compat.v1.Session
    content = re.sub(r'\btf\.Session\b', 'tf.compat.v1.Session', content)
    
    # tf.placeholder -> tf.compat.v1.placeholder
    content = re.sub(r'\btf\.placeholder\b', 'tf.compat.v1.placeholder', content)
    
    # tf.variable_scope -> tf.compat.v1.variable_scope
    content = re.sub(r'\btf\.variable_scope\b', 'tf.compat.v1.variable_scope', content)
    
    # tf.get_variable -> tf.compat.v1.get_variable
    content = re.sub(r'\btf\.get_variable\b', 'tf.compat.v1.get_variable', content)
    
    # tf.global_variables -> tf.compat.v1.global_variables
    content = re.sub(r'\btf\.global_variables\b', 'tf.compat.v1.global_variables', content)
    
    # tf.variables_initializer -> tf.compat.v1.variables_initializer
    content = re.sub(r'\btf\.variables_initializer\b', 'tf.compat.v1.variables_initializer', content)
    
    # tf.is_variable_initialized -> tf.compat.v1.is_variable_initialized
    content = re.sub(r'\btf\.is_variable_initialized\b', 'tf.compat.v1.is_variable_initialized', content)
    
    # tf.trainable_variables -> tf.compat.v1.trainable_variables
    content = re.sub(r'\btf\.trainable_variables\b', 'tf.compat.v1.trainable_variables', content)
    
    # tf.truncated_normal_initializer -> tf.compat.v1.truncated_normal_initializer
    content = re.sub(r'\btf\.truncated_normal_initializer\b', 'tf.compat.v1.truncated_normal_initializer', content)
    
    # tf.contrib.framework.get_trainable_variables -> tf.compat.v1.trainable_variables
    content = re.sub(
        r'tf\.contrib\.framework\.get_trainable_variables\(scope=([\'"][^\'"]+[\'"])\)',
        r'[v for v in tf.compat.v1.trainable_variables() if \1 in v.name]',
        content
    )
    
    # tf.contrib.layers.conv2d -> tf.compat.v1.layers.conv2d
    content = re.sub(
        r'tf\.contrib\.layers\.conv2d\(inputs=([^,]+), num_outputs=([^,]+), kernel_size=([^,]+), stride=([^,]+), padding=([^,]+),\s*activation_fn=([^)]+)\)',
        r'tf.compat.v1.layers.conv2d(inputs=\1, filters=\2, kernel_size=\3, strides=\4, padding=\5.lower().strip("\'\""))',
        content
    )
    
    # tf.contrib.layers.conv2d 简单替换
    content = re.sub(r'\btf\.contrib\.layers\.conv2d\b', 'tf.compat.v1.layers.conv2d', content)
    
    # tf.contrib.layers.batch_norm -> tf.compat.v1.layers.batch_normalization
    content = re.sub(r'\btf\.contrib\.layers\.batch_norm\b', 'tf.compat.v1.layers.batch_normalization', content)
    
    # tf.contrib.image.rotate -> tfa.image.rotate (需要tensorflow_addons)
    content = re.sub(r'\btf\.contrib\.image\.rotate\b', 'tfa.image.rotate', content)
    
    # 添加 tensorflow_addons import 如果需要
    if 'tfa.image.rotate' in content and 'import tensorflow_addons as tfa' not in content:
        content = re.sub(
            r'(import tensorflow as tf\n)',
            r'\1import tensorflow_addons as tfa\n',
            content
        )
    
    # tf.layers.conv2d -> tf.compat.v1.layers.conv2d
    content = re.sub(r'\btf\.layers\.conv2d\b', 'tf.compat.v1.layers.conv2d', content)
    
    # tf.layers.dense -> tf.compat.v1.layers.dense
    content = re.sub(r'\btf\.layers\.dense\b', 'tf.compat.v1.layers.dense', content)
    
    # tf.layers.batch_normalization -> tf.compat.v1.layers.batch_normalization
    content = re.sub(r'\btf\.layers\.batch_normalization\b', 'tf.compat.v1.layers.batch_normalization', content)
    content = re.sub(r'\btf\.layers\.batch_norm\b', 'tf.compat.v1.layers.batch_normalization', content)
    
    # is_training= -> training= for batch_normalization
    content = re.sub(r'batch_normalization\(([^)]*)\bis_training=', r'batch_normalization(\1training=', content)
    
    # tf.image.resize_images -> tf.image.resize
    content = re.sub(r'\btf\.image\.resize_images\b', 'tf.image.resize', content)
    
    # method=1 -> method='bilinear' for tf.image.resize
    content = re.sub(r"tf\.image\.resize\(([^)]+),\s*method=1\)", r"tf.image.resize(\1, method='bilinear')", content)
    
    # scipy.misc.imread -> imageio.imread
    content = re.sub(r'\bscipy\.misc\.imread\b', 'imageio.imread', content)
    content = re.sub(r'\bfrom scipy\.misc import imread\b', 'from imageio import imread', content)
    
    # scipy.misc.imsave -> imageio.imsave  
    content = re.sub(r'\bscipy\.misc\.imsave\b', 'imageio.imsave', content)
    content = re.sub(r'\bfrom scipy\.misc import imsave\b', 'from imageio import imsave', content)
    
    # scipy.misc.imresize -> 使用PIL或cv2
    # 这个需要特殊处理，暂时保留
    
    # tf.nn.moments keep_dims -> keepdims
    content = re.sub(r'tf\.nn\.moments\(([^)]+),\s*keep_dims=', r'tf.nn.moments(\1, keepdims=', content)
    
    return content


def process_file(filepath):
    """处理单个文件"""
    print(f"Processing: {filepath}")
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    new_content = fix_tf2_compatibility(content)
    
    if new_content != content:
        # 备份原文件
        backup_path = filepath + '.bak'
        if not os.path.exists(backup_path):
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # 写入新内容
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"  Modified: {filepath}")
        return True
    else:
        print(f"  No changes: {filepath}")
        return False


def process_directory(directory):
    """处理目录下的所有Python文件"""
    modified = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py') and not file.endswith('.bak'):
                filepath = os.path.join(root, file)
                if process_file(filepath):
                    modified += 1
    return modified


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python fix_tf2_compat.py <directory_or_file>")
        sys.exit(1)
    
    target = sys.argv[1]
    if os.path.isdir(target):
        modified = process_directory(target)
        print(f"\nModified {modified} files")
    elif os.path.isfile(target):
        process_file(target)
    else:
        print(f"Error: {target} not found")
        sys.exit(1)
