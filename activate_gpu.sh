#!/bin/bash
# MURF GPU 环境激活脚本

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate murf_gpu

# 设置 cuDNN 库路径
CUDNN_PATH=$(python -c "import nvidia.cudnn; print(nvidia.cudnn.__file__)" 2>/dev/null | xargs dirname)
if [ -n "$CUDNN_PATH" ]; then
    export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$LD_LIBRARY_PATH
    echo "✓ cuDNN 路径已配置: $CUDNN_PATH/lib"
fi

# 设置 cuBLAS 库路径
CUBLAS_PATH=$(python -c "import nvidia.cublas; print(nvidia.cublas.__file__)" 2>/dev/null | xargs dirname)
if [ -n "$CUBLAS_PATH" ]; then
    export LD_LIBRARY_PATH=$CUBLAS_PATH/lib:$LD_LIBRARY_PATH
    echo "✓ cuBLAS 路径已配置: $CUBLAS_PATH/lib"
fi

# 验证 GPU
echo ""
echo "验证 TensorFlow GPU..."
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'✓ 检测到 {len(gpus)} 个 GPU')
for gpu in gpus:
    print(f'  - {gpu.name}')
" 2>/dev/null

echo ""
echo "环境已就绪！可以运行:"
echo "  cd /home/sh/MURF/RGB-IR/fine_registration_and_fusion"
echo "  python test_tf2.py"
