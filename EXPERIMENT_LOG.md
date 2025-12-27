# MURF 论文复现实验日志

**论文**: MURF: Mutually Reinforcing Multi-Modal Image Registration and Fusion  
**复现时间**: 2025年12月26日 - 2025年12月27日  
**复现环境**: Ubuntu 22.04, Python 3.8, TensorFlow 2.10.0, NVIDIA RTX 4090 D × 8  

---

## 📋 目录

1. [复现概述](#1-复现概述)
2. [环境配置过程](#2-环境配置过程)
3. [问题与解决方案](#3-问题与解决方案)
4. [代码修改记录](#4-代码修改记录)
5. [测试执行记录](#5-测试执行记录)
6. [结果验证](#6-结果验证)

---

## 1. 复现概述

### 1.1 复现目标
直接使用提供的预训练模型，不采用数据集进行训练

复现 MURF 论文中的四个模态融合任务：
- RGB-IR (可见光-红外)
- RGB-NIR (可见光-近红外)
- PET-MRI (正电子发射断层扫描-核磁共振)
- CT-MRI (计算机断层扫描-核磁共振)

每个模态包含三个任务：
- Task 1: 共享信息提取 (shared_information_extraction)
- Task 2: 多尺度粗配准 (multi-scale_coarse_registration)
- Task 3: 精细配准与融合 (fine_registration_and_fusion)

### 1.2 复现结果总结

| 模块    | Task 1 | Task 2 | Task 3 | 状态                |
| ------- | :----: | :----: | :----: | ------------------- |
| RGB-IR  |   ✅    |   ✅    |   ✅    | 完整复现            |
| RGB-NIR |   ✅    |   ✅    |   ❌    | Task 3 无预训练模型 |
| PET-MRI |   ✅    |   ✅    |   ✅    | 完整复现            |
| CT-MRI  |   ✅    |   ✅    |   ✅    | 完整复现            |

**成功复现**: 11/12 个任务

---

## 2. 环境配置过程

### 2.1 初始尝试 (CPU 环境)

**时间**: 2025-12-26

首先尝试使用 CPU 环境运行原始 TensorFlow 1.14 代码：

```bash
conda create -n murf python=3.6 -y
conda activate murf
pip install tensorflow==1.14.0
pip install scikit-image==0.17.2 opencv-python-headless imageio
```

**结果**:
- Task 1: ✅ 成功
- Task 2: ⚠️ 需要修改设备分配
- Task 3: ❌ 失败 (分组卷积不支持 CPU)

### 2.2 GPU 环境配置

**时间**: 2025-12-27

由于 Task 3 需要 GPU，配置 TensorFlow 2.x GPU 环境：

```bash
conda create -n murf_gpu python=3.8 -y
conda activate murf_gpu
pip install tensorflow==2.10.0
pip install nvidia-cudnn-cu11==8.6.0.163
pip install nvidia-cublas-cu11==11.11.3.6
pip install scikit-image==0.19.3 opencv-python-headless imageio matplotlib h5py scipy pillow
```

**关键配置**: 创建 `activate_gpu.sh` 脚本配置 CUDA 库路径：

```bash
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="${SITE_PACKAGES}/nvidia/cudnn/lib:${SITE_PACKAGES}/nvidia/cublas/lib:${LD_LIBRARY_PATH}"
```

### 2.3 最终环境

```
✅ Python: 3.8.20
✅ TensorFlow: 2.10.0
✅ CUDA: 11.x (pip nvidia-cudnn-cu11)
✅ cuDNN: 8.6.0.163
✅ NumPy: 1.24.3
✅ scikit-image: 0.19.3
✅ scipy: 1.10.1
✅ GPU: NVIDIA RTX 4090 D × 8
```

---

## 3. 问题与解决方案

### 3.1 问题一: scipy.misc.imread/imresize 已废弃

**错误信息**:
```
AttributeError: module 'scipy.misc' has no attribute 'imread'
AttributeError: module 'scipy.misc' has no attribute 'imresize'
```

**原因**: scipy 1.3.0+ 移除了 `scipy.misc.imread` 和 `scipy.misc.imresize`

**解决方案**:
```python
# 替换 imread
from imageio import imread

# 替换 imresize
from PIL import Image
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
```

**影响文件**: 所有模块的 `test.py`, `utils.py`

---

### 3.2 问题二: TensorFlow 1.x API 不兼容

**错误信息**:
```
AttributeError: module 'tensorflow' has no attribute 'Session'
AttributeError: module 'tensorflow' has no attribute 'placeholder'
```

**原因**: TensorFlow 2.x 移除了 TF1.x 的部分 API

**解决方案**:
```python
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# API 替换
tf.Session() → tf.compat.v1.Session()
tf.placeholder() → tf.compat.v1.placeholder()
tf.variable_scope() → tf.compat.v1.variable_scope()
tf.get_variable() → tf.compat.v1.get_variable()
tf.train.Saver() → tf.compat.v1.train.Saver()
tf.global_variables() → tf.compat.v1.global_variables()
tf.trainable_variables() → tf.compat.v1.trainable_variables()
```

**自动修复脚本**: 创建 `fix_tf2_compat.py` 批量处理所有文件

---

### 3.3 问题三: GPU 设备分配错误

**错误信息**:
```
Could not satisfy explicit device specification '/device:GPU:1' because no supported kernel for GPU devices is available
```

**原因**: 代码中硬编码了 `/gpu:1`，但 CUDA_VISIBLE_DEVICES 只暴露了一个 GPU

**解决方案**:
```python
# 在 affine_model.py 中
# 将 /gpu:1 改为 /gpu:0
with tf.device('/gpu:0'):
    ...
```

**影响文件**: 
- `PET-MRI/multi-scale_coarse_registration/affine_model.py`
- `CT-MRI/multi-scale_coarse_registration/affine_model.py`

---

### 3.4 问题四: tf.ceil/tf.floor 函数变更

**错误信息**:
```
AttributeError: module 'tensorflow' has no attribute 'ceil'
```

**原因**: TF2.x 中 `tf.ceil` 和 `tf.floor` 移至 `tf.math` 模块

**解决方案**:
```python
tf.ceil() → tf.math.ceil()
tf.floor() → tf.math.floor()
```

**影响文件**: 所有模块的 `affine_model.py`

---

### 3.5 问题五: 变量名不匹配导致模型加载失败

**错误信息**:
```
Key Conv/biases not found in checkpoint
Key Conv/weights not found in checkpoint
```

**原因**: TensorFlow 2.x 中 `tf.layers.conv2d` 的变量命名与 TF1.x 不同

**解决方案**: 在 `utils.py` 的 `up_layer` 函数中使用手动创建变量：

```python
def up_layer(x, channels, scope, activation=lrelu):
    with tf.compat.v1.variable_scope(scope):
        # 手动创建变量以匹配 checkpoint
        weights = tf.compat.v1.get_variable(
            "Conv/weights", 
            [3, 3, x.shape[-1], channels],
            initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1)
        )
        biases = tf.compat.v1.get_variable(
            "Conv/biases", 
            [channels],
            initializer=tf.constant_initializer(0.0)
        )
        # 上采样
        upsampled = tf.image.resize(x, [tf.shape(x)[1]*2, tf.shape(x)[2]*2], method='bilinear')
        # 卷积
        conv = tf.nn.conv2d(upsampled, weights, strides=[1,1,1,1], padding='SAME') + biases
        return activation(conv)
```

**影响文件**: 所有模块的 `multi-scale_coarse_registration/utils.py`

---

### 3.6 问题六: 图像保存数据类型错误

**错误信息**:
```
TypeError: Cannot handle this data type: (1, 1, 256), |f4
```

**原因**: `imageio.imsave` 需要 uint8 类型，但传入了 float32 类型

**解决方案**:
```python
# 保存图像时转换类型
imsave(path, (np.clip(img, 0, 1) * 255).astype(np.uint8))
```

**影响文件**: 所有模块的 `test.py`

---

### 3.7 问题七: 融合图像全黑

**现象**: PET-MRI 和 CT-MRI 的 Task 3 输出图像像素值全为 0

**原因**: `test.py` 中 `imresize` 后又除以 255，导致值接近 0

**错误代码**:
```python
fused_img = imresize(fused_img[0, :, :, :], ...).astype(np.float32) / 255.0
imsave(path, (np.clip(fused_img, 0, 1) * 255).astype(np.uint8))
```

**正确代码**:
```python
# fused_img 输出范围是 [0,1]，先转 uint8 再 resize
fused_img = (np.clip(fused_img[0, :, :, :], 0, 1) * 255).astype(np.uint8)
fused_img = imresize(fused_img, ...)
imsave(path, fused_img)
```

**影响文件**: 
- `PET-MRI/fine_registration_and_fusion/test.py`
- `CT-MRI/fine_registration_and_fusion/test.py`

---

## 4. 代码修改记录

### 4.1 修改文件统计

共修改 47 个文件，原始文件备份为 `.tf1_original` 后缀。

**按模块分类**:

| 模块    | Task 1 | Task 2 | Task 3 | 合计 |
| ------- | ------ | ------ | ------ | ---- |
| RGB-IR  | 6      | 5      | 5      | 16   |
| RGB-NIR | 6      | 5      | 0      | 11   |
| PET-MRI | 6      | 5      | 5      | 16   |
| CT-MRI  | 6      | 5      | 5      | 16   |

**注**: RGB-NIR Task 3 无预训练模型，未进行修改

### 4.2 主要修改内容

#### 通用修改 (所有 test.py)

```python
# 添加 TF2 兼容
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# 添加 imresize 函数
from PIL import Image
def imresize(img, size):
    pil_img = Image.fromarray(img.astype(np.uint8))
    if isinstance(size, tuple):
        pil_img = pil_img.resize((size[1], size[0]), Image.BILINEAR)
    else:
        new_h = int(pil_img.height * size)
        new_w = int(pil_img.width * size)
        pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
    return np.array(pil_img)

# 修改图像保存
imsave(path, (np.clip(img, 0, 1) * 255).astype(np.uint8))
```

#### Task 2 affine_model.py 修改

```python
# GPU 设备修改
with tf.device('/gpu:0'):  # 原为 /gpu:1

# 数学函数修改
tf.math.ceil(...)  # 原为 tf.ceil
tf.math.floor(...)  # 原为 tf.floor
```

#### Task 2 utils.py 修改

```python
def up_layer(x, channels, scope, activation=lrelu):
    with tf.compat.v1.variable_scope(scope):
        # 使用与 checkpoint 匹配的变量名
        weights = tf.compat.v1.get_variable(
            "Conv/weights", 
            [3, 3, x.shape[-1], channels],
            initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1)
        )
        biases = tf.compat.v1.get_variable(
            "Conv/biases", 
            [channels],
            initializer=tf.constant_initializer(0.0)
        )
        upsampled = tf.image.resize(x, [tf.shape(x)[1]*2, tf.shape(x)[2]*2], method='bilinear')
        conv = tf.nn.conv2d(upsampled, weights, strides=[1,1,1,1], padding='SAME') + biases
        return activation(conv)
```

### 4.3 备份文件列表

```
CT-MRI/shared_information_extraction/
├── utils.py.tf1_original
├── Encoder.py.tf1_original
├── main.py.tf1_original
├── train.py.tf1_original
├── des_extract_model.py.tf1_original
└── test.py.tf1_original

CT-MRI/multi-scale_coarse_registration/
├── utils.py.tf1_original
├── affine_model.py.tf1_original
├── main.py.tf1_original
├── train.py.tf1_original
└── test.py.tf1_original

CT-MRI/fine_registration_and_fusion/
├── utils.py.tf1_original
├── main.py.tf1_original
├── train.py.tf1_original
├── test.py.tf1_original
└── f2m_model.py.tf1_original

(其他模块结构类似...)
```

---

## 5. 测试执行记录

### 5.1 RGB-IR 模块

#### Task 1: 共享信息提取
```
时间: 2025-12-26 23:02
测试图像: FLIR_00006.jpg, FLIR_00018.jpg, FLIR_00060.jpg, FLIR_00122.jpg
模型: 4200.ckpt
状态: ✅ 成功
输出: des_results/RGB/, des_results/IR/
```

#### Task 2: 多尺度粗配准
```
时间: 2025-12-26 23:15
测试图像: 1.jpg
模型: 6400.ckpt
状态: ✅ 成功
输出: results/warped_RGB/, results/compare/
仿射变换矩阵:
[[ 1.0302006  -0.03416662 -0.002363  ]
 [ 0.08091667  1.0669252  -0.00170587]]
```

#### Task 3: 精细配准与融合
```
时间: 2025-12-27 18:45
测试图像: 1.jpg
模型: 0000.ckpt
状态: ✅ 成功 (GPU)
输出: results/fused_img/1.jpg
```

### 5.2 RGB-NIR 模块

#### Task 1: 共享信息提取
```
时间: 2025-12-27 18:22
测试图像: 1.png
模型: 3600.ckpt
状态: ✅ 成功
输出: des_results/RGB/, des_results/NIR/
```

#### Task 2: 多尺度粗配准
```
时间: 2025-12-27 18:25
测试图像: 1.png
模型: 9300.ckpt
状态: ✅ 成功
输出: results/warped_RGB/, results/compare/
```

#### Task 3: 精细配准与融合
```
状态: ❌ 跳过 (无预训练模型)
```

### 5.3 PET-MRI 模块

#### Task 1: 共享信息提取
```
时间: 2025-12-27 18:30
测试图像: 1.png
模型: 0000.ckpt
状态: ✅ 成功
输出: des_results/PET/, des_results/MRI/
```

#### Task 2: 多尺度粗配准
```
时间: 2025-12-27 18:55
测试图像: 1.png
模型: 0000.ckpt
状态: ✅ 成功
输出: results/warped_PET/, results/compare/
```

#### Task 3: 精细配准与融合
```
时间: 2025-12-27 19:13
测试图像: 1.png
模型: 0000.ckpt
状态: ✅ 成功 (GPU)
输出: results/Fusion/1.png
```

### 5.4 CT-MRI 模块

#### Task 1: 共享信息提取
```
时间: 2025-12-27 18:30
测试图像: 1.png
模型: 0000.ckpt
状态: ✅ 成功
输出: des_results/CT/, des_results/MRI/
```

#### Task 2: 多尺度粗配准
```
时间: 2025-12-27 18:58
测试图像: 1.png
模型: 0000.ckpt
状态: ✅ 成功
输出: results/warped_CT/, results/compare/
```

#### Task 3: 精细配准与融合
```
时间: 2025-12-27 19:14
测试图像: 1.png
模型: 0000.ckpt
状态: ✅ 成功 (GPU)
输出: results/Fusion/1.png
```

---

## 6. 结果验证

### 6.1 融合结果图像验证

```python
# 图像像素值检查
PET-MRI Fusion: shape=(256, 256, 3), min=0, max=253, mean=60.08
CT-MRI Fusion:  shape=(256, 256),    min=0, max=254, mean=66.07
RGB-IR Fusion:  shape=(358, 561, 3), min=3, max=255, mean=138.69
```

### 6.2 评估指标结果

运行 `python evaluate_results.py` 得到以下结果：

| 模态    | MI     | SSIM   | CC     | EN     | SF      | AG       | SD      |
| ------- | ------ | ------ | ------ | ------ | ------- | -------- | ------- |
| RGB-IR  | 1.3452 | 0.6929 | 0.4176 | 6.8392 | 10.2869 | 40.0253  | 32.1009 |
| PET-MRI | 1.2601 | 0.3122 | 0.8322 | 5.0029 | 43.4577 | 119.1321 | 76.9609 |
| CT-MRI  | 1.3694 | 0.6769 | 0.7901 | 5.2487 | 44.1857 | 118.6856 | 75.8238 |

### 6.3 指标说明

- **MI (Mutual Information)**: 互信息，衡量融合图像与源图像的信息保留程度
- **SSIM (Structural Similarity)**: 结构相似性，衡量结构信息保留
- **CC (Correlation Coefficient)**: 相关系数，衡量线性相关性
- **EN (Entropy)**: 信息熵，衡量图像信息量
- **SF (Spatial Frequency)**: 空间频率，衡量图像清晰度
- **AG (Average Gradient)**: 平均梯度，衡量边缘强度
- **SD (Standard Deviation)**: 标准差，衡量对比度

---

## 附录: 关键命令速查

### 环境激活
```bash
source /home/sh/MURF/activate_gpu.sh
```

### 一键测试
```bash
cd /home/sh/MURF
bash run_all_tests.sh
```

### 单独测试
```bash
cd /home/sh/MURF/RGB-IR/shared_information_extraction && python test.py
cd /home/sh/MURF/RGB-IR/multi-scale_coarse_registration && python test.py
cd /home/sh/MURF/RGB-IR/fine_registration_and_fusion && python test.py
```

### 评估融合结果
```bash
cd /home/sh/MURF && python evaluate_results.py
```

---

**日志完成时间**: 2025-12-27 19:30
