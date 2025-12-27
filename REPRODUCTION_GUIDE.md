# MURF 论文复现完整指南

<p align="center">
  <b>MURF: Mutually Reinforcing Multi-Modal Image Registration and Fusion</b><br>
  <i>IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) 2023</i>
</p>

---

## 📋 目录

1. [论文信息](#1-论文信息)
2. [项目结构](#2-项目结构)
3. [环境配置](#3-环境配置)
4. [快速开始](#4-快速开始)
5. [完整测试命令](#5-完整测试命令)
6. [训练流程](#6-训练流程)
7. [数据集资源](#7-数据集资源)
8. [评估方法](#8-评估方法)
9. [常见问题与解决方案](#9-常见问题与解决方案)
10. [复现结果](#10-复现结果)
11. [引用](#11-引用)

---

## 1. 论文信息

| 项目 | 内容 |
|------|------|
| **论文标题** | MURF: Mutually Reinforcing Multi-Modal Image Registration and Fusion |
| **发表期刊** | IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) |
| **发表年份** | 2023 |
| **作者** | Han Xu, Jiayi Ma, Jiteng Yuan, Zhuliang Le, Wei Liu |
| **官方仓库** | https://github.com/hanna-xu/MURF |
| **本地路径** | `/home/sh/MURF` |
| **论文PDF** | `TPAMI_MURF.pdf` |

### 核心贡献

1. 提出了一个统一的多模态图像配准与融合框架 (MURF)
2. 设计了共享信息提取网络，用于跨模态特征学习
3. 实现了多尺度粗配准和精细配准的级联策略
4. 在多个数据集上取得了 SOTA 性能

---

## 2. 项目结构

### 2.1 总体结构

```
MURF/
├── README.md                    # 官方说明
├── TPAMI_MURF.pdf              # 论文 PDF
├── REPRODUCTION_GUIDE.md        # 本复现指南
├── EXPERIMENT_LOG.md            # 实验日志
├── PROJECT_REPORT.md            # 项目报告
│
├── setup_env.sh                 # 环境配置脚本
├── activate_gpu.sh              # GPU 环境激活脚本
├── run_all_tests.sh             # 一键测试脚本
├── evaluate_results.py          # 融合结果评估脚本
├── fix_tf2_compat.py            # TF1→TF2 兼容性修复脚本
│
├── RGB-IR/                      # RGB-红外融合 (4个测试图像)
├── RGB-NIR/                     # RGB-近红外融合 (1个测试图像)
├── PET-MRI/                     # PET-MRI 医学图像融合 (1个测试图像)
└── CT-MRI/                      # CT-MRI 医学图像融合 (1个测试图像)
```

### 2.2 模态目录结构 (以 RGB-IR 为例)

```
RGB-IR/
├── shared_information_extraction/      # Task 1: 共享信息提取
│   ├── test.py                         # 测试脚本
│   ├── test.py.tf1_original           # TF1 原始版本备份
│   ├── main.py                         # 训练入口
│   ├── train.py                        # 训练逻辑
│   ├── des_extract_model.py            # 描述符提取模型
│   ├── Encoder.py                      # 编码器网络
│   ├── utils.py                        # 工具函数
│   ├── models/                         # 预训练模型
│   │   ├── checkpoint
│   │   ├── 4200.ckpt.data-00000-of-00001
│   │   ├── 4200.ckpt.index
│   │   └── 4200.ckpt.meta
│   ├── test_imgs/                      # 测试输入
│   │   ├── RGB/
│   │   └── IR/
│   └── des_results/                    # 测试输出
│       ├── RGB/
│       └── IR/
│
├── multi-scale_coarse_registration/    # Task 2: 多尺度粗配准
│   ├── test.py
│   ├── main.py
│   ├── train.py
│   ├── affine_model.py                 # 仿射变换模型
│   ├── utils.py
│   ├── models/                         # 预训练模型 (6400.ckpt)
│   ├── test_data/                      # 测试输入
│   │   ├── images/
│   │   └── LM/
│   └── results/                        # 测试输出
│       ├── warped_RGB/
│       └── compare/
│
└── fine_registration_and_fusion/       # Task 3: 精细配准与融合
    ├── test.py
    ├── main.py
    ├── train.py
    ├── f2m_model.py                    # 融合模型
    ├── utils.py
    ├── models/                         # 预训练模型 (0000.ckpt)
    ├── test_imgs/                      # 测试输入
    │   ├── RGB/
    │   └── IR/
    └── results/                        # 测试输出
        └── fused_img/
```

### 2.3 任务说明

| 任务 | 目录名 | 功能 | 输入 | 输出 |
|------|--------|------|------|------|
| Task 1 | `shared_information_extraction/` | 共享信息提取 | 多模态图像对 | 描述符图像 |
| Task 2 | `multi-scale_coarse_registration/` | 多尺度粗配准 | 未对齐图像 + 关键点 | 仿射变换矩阵 + 粗配准图像 |
| Task 3 | `fine_registration_and_fusion/` | 精细配准与融合 | 粗配准后图像对 | 融合结果图像 |

### 2.4 预训练模型汇总

| 模态 | Task 1 | Task 2 | Task 3 |
|------|--------|--------|--------|
| RGB-IR | ✅ 4200.ckpt | ✅ 6400.ckpt | ✅ 0000.ckpt |
| RGB-NIR | ✅ 3600.ckpt | ✅ 9300.ckpt | ❌ 无模型 |
| PET-MRI | ✅ 0000.ckpt | ✅ 0000.ckpt | ✅ 0000.ckpt |
| CT-MRI | ✅ 0000.ckpt | ✅ 0000.ckpt | ✅ 0000.ckpt |

---

## 3. 环境配置

### 3.1 一键配置 (推荐)

```bash
cd /home/sh/MURF
bash setup_env.sh
```

脚本会自动检测 GPU 并配置最优环境。

### 3.2 手动配置

#### GPU 环境 (推荐，支持所有任务)

```bash
# 创建环境
conda create -n murf_gpu python=3.8 -y
conda activate murf_gpu

# 配置镜像加速
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 TensorFlow 2.10 (GPU)
pip install tensorflow==2.10.0

# 安装 CUDA 库 (pip 方式，无需系统安装)
pip install nvidia-cudnn-cu11==8.6.0.163
pip install nvidia-cublas-cu11==11.11.3.6

# 安装其他依赖
pip install scikit-image==0.19.3 opencv-python-headless imageio matplotlib h5py scipy pillow
```

#### CPU 环境 (仅支持 Task 1 和 Task 2)

```bash
# 创建环境
conda create -n murf_cpu python=3.6 -y
conda activate murf_cpu

# 安装依赖
pip install tensorflow==1.14.0
pip install scikit-image==0.17.2 opencv-python-headless imageio matplotlib h5py scipy pillow
```

### 3.3 已验证 GPU 环境

```
✅ Python: 3.8.20
✅ TensorFlow: 2.10.0
✅ CUDA: 11.x (通过 pip nvidia-cudnn-cu11)
✅ cuDNN: 8.6.0
✅ NumPy: 1.24.3
✅ scikit-image: 0.19.3
✅ scipy: 1.10.1
✅ Pillow: 10.4.0
✅ imageio: 2.36.1
✅ GPU: NVIDIA RTX 4090 D × 8
```

### 3.4 激活 GPU 环境

```bash
source /home/sh/MURF/activate_gpu.sh
```

此脚本会：
1. 激活 conda 环境
2. 配置 cuDNN/cuBLAS 库路径
3. 验证 GPU 可用性

---

## 4. 快速开始

### 4.1 一键测试所有任务

```bash
cd /home/sh/MURF
bash run_all_tests.sh
```

### 4.2 单独测试某个任务

```bash
# 激活环境
source /home/sh/MURF/activate_gpu.sh

# 测试 RGB-IR Task 1
cd /home/sh/MURF/RGB-IR/shared_information_extraction && python test.py

# 测试 RGB-IR Task 2  
cd /home/sh/MURF/RGB-IR/multi-scale_coarse_registration && python test.py

# 测试 RGB-IR Task 3
cd /home/sh/MURF/RGB-IR/fine_registration_and_fusion && python test.py
```

### 4.3 评估融合结果

```bash
cd /home/sh/MURF
python evaluate_results.py
```

---

## 5. 完整测试命令

### 5.1 RGB-IR (4个测试图像)

```bash
# Task 1: 共享信息提取
cd /home/sh/MURF/RGB-IR/shared_information_extraction
python test.py
# 输入: test_imgs/RGB/, test_imgs/IR/
# 输出: des_results/RGB/, des_results/IR/

# Task 2: 多尺度粗配准
cd /home/sh/MURF/RGB-IR/multi-scale_coarse_registration
python test.py
# 输入: test_data/images/, test_data/LM/
# 输出: results/warped_RGB/, results/compare/

# Task 3: 精细配准与融合 (需要 GPU)
cd /home/sh/MURF/RGB-IR/fine_registration_and_fusion
python test.py
# 输入: test_imgs/RGB/, test_imgs/IR/
# 输出: results/fused_img/
```

### 5.2 RGB-NIR (1个测试图像)

```bash
# Task 1: 共享信息提取
cd /home/sh/MURF/RGB-NIR/shared_information_extraction
python test.py
# 输出: des_results/RGB/, des_results/NIR/

# Task 2: 多尺度粗配准
cd /home/sh/MURF/RGB-NIR/multi-scale_coarse_registration
python test.py
# 输出: results/warped_RGB/, results/compare/

# Task 3: ❌ 无预训练模型，跳过
```

### 5.3 PET-MRI (1个测试图像)

```bash
# Task 1: 共享信息提取
cd /home/sh/MURF/PET-MRI/shared_information_extraction
python test.py
# 输出: des_results/PET/, des_results/MRI/

# Task 2: 多尺度粗配准
cd /home/sh/MURF/PET-MRI/multi-scale_coarse_registration
python test.py
# 输出: results/warped_PET/, results/compare/

# Task 3: 精细配准与融合
cd /home/sh/MURF/PET-MRI/fine_registration_and_fusion
python test.py
# 输出: results/Fusion/
```

### 5.4 CT-MRI (1个测试图像)

```bash
# Task 1: 共享信息提取
cd /home/sh/MURF/CT-MRI/shared_information_extraction
python test.py
# 输出: des_results/CT/, des_results/MRI/

# Task 2: 多尺度粗配准
cd /home/sh/MURF/CT-MRI/multi-scale_coarse_registration
python test.py
# 输出: results/warped_CT/, results/compare/

# Task 3: 精细配准与融合
cd /home/sh/MURF/CT-MRI/fine_registration_and_fusion
python test.py
# 输出: results/Fusion/
```

---

## 6. 训练流程

> ⚠️ **重要**: 三个任务需要按顺序执行，后续任务依赖前置模型。

### 6.1 Task 1: 共享信息提取

```bash
cd /home/sh/MURF/RGB-IR/shared_information_extraction

# 1. 下载训练数据到当前目录 (见数据集资源章节)
# 2. 运行训练
python main.py
```

**训练参数**:
- `patch_size`: 128
- `EPOCHES`: 50
- `BATCH_SIZE`: 32

### 6.2 Task 2: 多尺度粗配准

```bash
cd /home/sh/MURF/RGB-IR/multi-scale_coarse_registration

# 1. 确保 Task 1 模型已训练完成
# 2. 修改 main.py 中的 task1_model_savepath 指向 Task 1 模型
# 3. 下载训练数据
# 4. 运行训练
python main.py
```

**训练参数**:
- `patch_size`: 256
- `EPOCHES`: 200
- `BATCH_SIZE`: 32

### 6.3 Task 3: 精细配准与融合

```bash
cd /home/sh/MURF/RGB-IR/fine_registration_and_fusion

# 使用与 Task 1 相同的训练数据
python main.py
```

**训练参数**:
- `patch_size`: 128
- `EPOCHES`: 20
- `BATCH_SIZE`: 8

---

## 7. 数据集资源

### 7.1 训练数据 (百度网盘)

#### Task 1 训练数据

| 模态 | 链接 | 提取码 |
|------|------|--------|
| RGB-IR | https://pan.baidu.com/s/1MPSmWuOhKr2KQxD8aj5gHA | e9gf |
| RGB-NIR | https://pan.baidu.com/s/1oakDnUKCtT0MaxjP-6Q0jA | epov |
| PET-MRI | https://pan.baidu.com/s/1BgX7lFbtZ4cunR7P160cnA | hu06 |
| CT-MRI | https://pan.baidu.com/s/1WtVS8qO83tB8coy5TvJE8Q | rphq |

#### Task 2 训练数据

| 模态 | 链接 | 提取码 |
|------|------|--------|
| RGB-IR | https://pan.baidu.com/s/11-vMvbzLyR1FxnIi0jxGWg | 8sih |
| RGB-NIR | https://pan.baidu.com/s/1P24HU1vDbDxcDZmM8b_ruA | ry6r |
| PET-MRI | https://pan.baidu.com/s/1ZlQCiDfnL36qqgq2p7XxoA | th6o |
| CT-MRI | https://pan.baidu.com/s/1pYrf_GzGujFF-xW4QVA6xg | ik0k |

### 7.2 原始数据集来源

| 数据集 | 模态 | 链接 |
|--------|------|------|
| RoadScene | RGB-IR | https://github.com/hanna-xu/RoadScene |
| VIS-NIR Scene | RGB-NIR | http://matthewalunbrown.com/nirscene/nirscene.html |
| Harvard Medical | PET-MRI, CT-MRI | http://www.med.harvard.edu/AANLIB/home.html |

---

## 8. 评估方法

### 8.1 评价指标

本项目使用以下指标评估融合结果质量：

| 指标 | 全称 | 说明 | 期望 |
|------|------|------|------|
| MI | Mutual Information | 互信息，衡量融合图与源图的信息保留 | 越高越好 |
| SSIM | Structural Similarity Index | 结构相似性 | 越高越好 |
| CC | Correlation Coefficient | 相关系数 | 越高越好 |
| EN | Entropy | 信息熵，衡量图像信息量 | 越高越好 |
| SF | Spatial Frequency | 空间频率，衡量图像清晰度 | 越高越好 |
| AG | Average Gradient | 平均梯度，衡量边缘强度 | 越高越好 |
| SD | Standard Deviation | 标准差，衡量对比度 | 越高越好 |

### 8.2 运行评估

```bash
cd /home/sh/MURF
python evaluate_results.py
```

### 8.3 评估结果示例

```
================================================================================
  综合评估结果汇总表
================================================================================
模态                 MI     SSIM       CC       EN       SF       SD       AG
--------------------------------------------------------------------------------
RGB-IR         1.3452   0.6929   0.4176   6.8392  10.2869  32.1009  40.0253
PET-MRI        1.2601   0.3122   0.8322   5.0029  43.4577  76.9609 119.1321
CT-MRI         1.3694   0.6769   0.7901   5.2487  44.1857  75.8238 118.6856
================================================================================
```

---

## 9. 常见问题与解决方案

### Q1: `scipy.misc.imread` 报错

**错误信息**: `AttributeError: module 'scipy.misc' has no attribute 'imread'`

**原因**: scipy 新版本移除了 `scipy.misc.imread` 和 `scipy.misc.imresize`

**解决方案**: 代码已修复，使用 `imageio.imread` 和自定义 `imresize` 函数替代

### Q2: TensorFlow 1.x API 在 TensorFlow 2.x 中不可用

**错误信息**: `AttributeError: module 'tensorflow' has no attribute 'Session'`

**解决方案**: 
```python
# 在代码开头添加
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
# 将 tf.xxx 改为 tf.compat.v1.xxx
```

### Q3: GPU 设备分配错误

**错误信息**: `Could not satisfy explicit device specification '/device:GPU:1'`

**原因**: 代码中硬编码了 `/gpu:1`，但系统只有 GPU:0 可见

**解决方案**: 将 `affine_model.py` 中的 `/gpu:1` 改为 `/gpu:0`

### Q4: 变量名不匹配导致模型加载失败

**错误信息**: `Key Conv/biases not found in checkpoint`

**原因**: TensorFlow 2.x 中 `tf.layers.conv2d` 的变量命名与 TF1.x 不同

**解决方案**: 使用 `tf.nn.conv2d` 手动创建卷积层，并使用 `tf.compat.v1.get_variable` 创建变量

### Q5: 图像保存时数据类型错误

**错误信息**: `TypeError: Cannot handle this data type`

**原因**: `imsave` 需要 uint8 类型，但传入了 float 类型

**解决方案**: 
```python
imsave(path, (np.clip(img, 0, 1) * 255).astype(np.uint8))
```

### Q6: cuDNN/cuBLAS 库未找到

**错误信息**: `Could not load dynamic library 'libcudnn.so.8'`

**解决方案**: 
```bash
# 使用 activate_gpu.sh 脚本
source /home/sh/MURF/activate_gpu.sh
```

### Q7: Task 3 在 CPU 模式下报错

**错误信息**: `Generic conv implementation does not support grouped convolutions`

**原因**: Task 3 使用了分组卷积，TensorFlow CPU 版本不支持

**解决方案**: Task 3 必须使用 GPU 运行

---

## 10. 复现结果

### 10.1 任务完成状态

| 模块 | Task 1 | Task 2 | Task 3 | 备注 |
|------|:------:|:------:|:------:|------|
| **RGB-IR** | ✅ | ✅ | ✅ | 完整支持 |
| **RGB-NIR** | ✅ | ✅ | ❌ | Task 3 无预训练模型 |
| **PET-MRI** | ✅ | ✅ | ✅ | 完整支持 |
| **CT-MRI** | ✅ | ✅ | ✅ | 完整支持 |

**总计**: 11/12 个任务成功复现

### 10.2 融合结果评估

| 模态 | MI | SSIM | CC | EN | SF | AG | SD |
|------|-----|------|-----|-----|-----|-----|-----|
| RGB-IR | 1.3452 | 0.6929 | 0.4176 | 6.8392 | 10.2869 | 40.0253 | 32.1009 |
| PET-MRI | 1.2601 | 0.3122 | 0.8322 | 5.0029 | 43.4577 | 119.1321 | 76.9609 |
| CT-MRI | 1.3694 | 0.6769 | 0.7901 | 5.2487 | 44.1857 | 118.6856 | 75.8238 |

### 10.3 输出文件位置

| 模态 | Task 1 输出 | Task 2 输出 | Task 3 输出 |
|------|-------------|-------------|-------------|
| RGB-IR | `RGB-IR/shared_information_extraction/des_results/` | `RGB-IR/multi-scale_coarse_registration/results/` | `RGB-IR/fine_registration_and_fusion/results/fused_img/` |
| RGB-NIR | `RGB-NIR/shared_information_extraction/des_results/` | `RGB-NIR/multi-scale_coarse_registration/results/` | N/A |
| PET-MRI | `PET-MRI/shared_information_extraction/des_results/` | `PET-MRI/multi-scale_coarse_registration/results/` | `PET-MRI/fine_registration_and_fusion/results/Fusion/` |
| CT-MRI | `CT-MRI/shared_information_extraction/des_results/` | `CT-MRI/multi-scale_coarse_registration/results/` | `CT-MRI/fine_registration_and_fusion/results/Fusion/` |

---

## 11. 引用

如果本项目对您的研究有帮助，请引用以下论文：

```bibtex
@article{xu2023murf,
  title={MURF: Mutually Reinforcing Multi-modal Image Registration and Fusion},
  author={Xu, Han and Ma, Jiayi and Yuan, Jiteng and Le, Zhuliang and Liu, Wei},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={45},
  number={10},
  pages={12148--12166},
  year={2023},
  publisher={IEEE}
}

@inproceedings{xu2022rfnet,
  title={RFNet: Unsupervised Network for Mutually Reinforcing Multi-modal Image Registration and Fusion},
  author={Xu, Han and Ma, Jiayi and Yuan, Jiteng and Le, Zhuliang and Liu, Wei},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={19679--19688},
  year={2022}
}
```

---

## 附录

### A. 文件修改记录

所有 TensorFlow 1.x 原始代码已备份为 `.tf1_original` 后缀文件，共 47 个文件。

主要修改内容：
1. `tf.xxx` → `tf.compat.v1.xxx`
2. `scipy.misc.imread/imresize` → `imageio.imread` + 自定义 `imresize`
3. `tf.ceil/floor` → `tf.math.ceil/floor`
4. GPU 设备分配修复
5. 图像保存数据类型修复

### B. 相关文件

| 文件 | 说明 |
|------|------|
| `setup_env.sh` | 环境配置脚本，支持 --gpu/--cpu/--auto 三种模式 |
| `activate_gpu.sh` | GPU 环境激活脚本，配置 CUDA 库路径 |
| `run_all_tests.sh` | 一键测试脚本，支持 --quick/--full 模式，测试所有11个任务 |
| `evaluate_results.py` | 融合结果评估脚本 |
| `fix_tf2_compat.py` | TF1→TF2 兼容性自动修复脚本 |
| `EXPERIMENT_LOG.md` | 详细实验日志 |
| `PROJECT_REPORT.md` | 项目报告 |

---

**复现环境**: Ubuntu 22.04, Python 3.8, TensorFlow 2.10.0, NVIDIA RTX 4090 D  
**更新时间**: 2025-12-27
