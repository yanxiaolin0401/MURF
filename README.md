# MURF - TensorFlow 2.x 兼容版本

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange.svg)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.8-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **基于 [hanna-xu/MURF](https://github.com/hanna-xu/MURF) 复现并升级至 TensorFlow 2.x**

本项目是 IEEE TPAMI 2023 论文 **"MURF: Mutually Reinforcing Multi-modal Image Registration and Fusion"** 的代码复现版本，已从 TensorFlow 1.14 升级至 **TensorFlow 2.10**，支持现代 GPU (如 RTX 4090) 环境运行。

## 📋 主要改进

相比原始代码，本版本进行了以下改进：

| 改进项     | 原版本                  | 本版本                |
| ---------- | ----------------------- | --------------------- |
| TensorFlow | 1.14 (仅支持 CUDA 10.x) | 2.10 (支持 CUDA 11.x) |
| Python     | 3.6                     | 3.8                   |
| 图像处理   | scipy.misc (已废弃)     | imageio + PIL         |
| GPU 支持   | 旧版 GPU                | RTX 30/40 系列        |
| 环境配置   | 手动配置                | 一键脚本              |

### 代码兼容性修改

- ✅ `tf.Session` → `tf.compat.v1.Session`
- ✅ `tf.placeholder` → `tf.compat.v1.placeholder`
- ✅ `tf.contrib.layers` → `tf.compat.v1.layers`
- ✅ `scipy.misc.imread/imresize` → `imageio.imread` + `PIL.Image.resize`
- ✅ 修复变量命名以兼容预训练模型
- ✅ 修复 GPU 设备分配问题

## 🚀 快速开始

### 请查看[论文复现指南](PROJECT_REPORT.md)

## 📁 项目结构

```
MURF/
├── README.md                    # 本文件
├── setup_env.sh                 # 环境配置脚本
├── activate_gpu.sh              # GPU 环境激活脚本
├── run_all_tests.sh             # 一键测试脚本
├── fix_tf2_compat.py            # TF1→TF2 自动转换脚本
├── evaluate_results.py          # 结果评估脚本
├── PROJECT_REPORT.md            # 详细复现报告
├── EXPERIMENT_LOG.md            # 实验日志
├── REPRODUCTION_GUIDE.md        # 完整复现指南
│
├── RGB-IR/                      # 可见光-红外融合
├── RGB-NIR/                     # 可见光-近红外融合
├── PET-MRI/                     # PET-MRI 医学图像融合
└── CT-MRI/                      # CT-MRI 医学图像融合
    ├── shared_information_extraction/      # Task 1: 共享信息提取
    ├── multi-scale_coarse_registration/    # Task 2: 多尺度粗配准
    └── fine_registration_and_fusion/       # Task 3: 精细配准与融合
```

## 🎯 支持的模态与任务

| 模态    | Task 1 共享信息提取 | Task 2 多尺度粗配准 | Task 3 精细配准融合 |
| ------- | :-----------------: | :-----------------: | :-----------------: |
| RGB-IR  |          ✅          |          ✅          |          ✅          |
| RGB-NIR |          ✅          |          ✅          |  ❌ (无预训练模型)   |
| PET-MRI |          ✅          |          ✅          |          ✅          |
| CT-MRI  |          ✅          |          ✅          |          ✅          |

**成功复现: 11/12 个任务**


## 💻 测试环境

- **OS**: Ubuntu 22.04
- **GPU**: NVIDIA RTX 4090 D
- **Python**: 3.8.20
- **TensorFlow**: 2.10.0
- **CUDA**: 11.x (通过 pip nvidia-cudnn-cu11)
- **cuDNN**: 8.6.0.163


## 📚 参考文献

```bibtex
@article{xu2023murf,
  title={MURF: Mutually Reinforcing Multi-modal Image Registration and Fusion},
  author={Xu, Han and Ma, Jiayi and Yuan, Jiteng and Le, Zhuliang and Liu, Wei},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023}
}

@inproceedings{xu2022rfnet,
  title={Rfnet: Unsupervised network for mutually reinforcing multi-modal image registration and fusion},
  author={Xu, Han and Ma, Jiayi and Yuan, Jiteng and Le, Zhuliang and Liu, Wei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19679--19688},
  year={2022}
}
```

## 🙏 致谢

- 原始代码: [hanna-xu/MURF](https://github.com/hanna-xu/MURF)
- 论文作者: Han Xu, Jiayi Ma, Jiteng Yuan, Zhuliang Le, Wei Liu (武汉大学)

## 📄 License

本项目采用 MIT License 开源协议。
