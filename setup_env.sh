#!/bin/bash
# ============================================================
# MURF 环境配置脚本 (GPU/CPU 自动检测版)
# 论文: MURF: Mutually Reinforcing Multi-Modal Image Registration and Fusion
# 使用方法: bash setup_env.sh [--gpu | --cpu | --auto]
# ============================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}  MURF 环境配置脚本${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}→ $1${NC}"
}

# 检测 GPU
detect_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi -L &> /dev/null; then
            GPU_COUNT=$(nvidia-smi -L | wc -l)
            if [ "$GPU_COUNT" -gt 0 ]; then
                return 0  # 有 GPU
            fi
        fi
    fi
    return 1  # 无 GPU
}

# 检查 conda
check_conda() {
    if ! command -v conda &> /dev/null; then
        print_error "未找到 conda，请先安装 Miniconda/Anaconda"
        echo "   下载地址: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
}

# 配置 GPU 环境
setup_gpu_env() {
    local ENV_NAME="murf_gpu"
    
    print_info "配置 GPU 环境: $ENV_NAME (Python 3.8 + TensorFlow 2.10)"
    
    # 检查环境是否存在
    if conda env list | grep -q "^${ENV_NAME} "; then
        print_warning "环境 '$ENV_NAME' 已存在"
        read -p "是否删除并重新创建? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            conda env remove -n $ENV_NAME -y
        else
            print_info "使用已有环境..."
            return 0
        fi
    fi
    
    # 创建环境
    print_info "创建 conda 环境..."
    conda create -n $ENV_NAME python=3.8 -y
    
    # 激活环境
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate $ENV_NAME
    
    # 配置镜像
    print_info "配置清华镜像源..."
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    pip config set global.timeout 300
    
    # 安装 TensorFlow GPU
    print_info "安装 TensorFlow 2.10.0 (GPU)..."
    pip install tensorflow==2.10.0
    
    # 安装 CUDA 库
    print_info "安装 CUDA 库 (nvidia-cudnn-cu11, nvidia-cublas-cu11)..."
    pip install nvidia-cudnn-cu11==8.6.0.163
    pip install nvidia-cublas-cu11==11.11.3.6
    
    # 安装其他依赖
    print_info "安装图像处理库..."
    pip install scikit-image==0.19.3
    pip install opencv-python-headless
    pip install imageio
    pip install matplotlib
    pip install h5py
    pip install scipy
    pip install pillow
    
    print_success "GPU 环境配置完成"
}

# 配置 CPU 环境
setup_cpu_env() {
    local ENV_NAME="murf_cpu"
    
    print_info "配置 CPU 环境: $ENV_NAME (Python 3.6 + TensorFlow 1.14)"
    print_warning "注意: CPU 环境仅支持 Task 1 和 Task 2，Task 3 需要 GPU"
    
    # 检查环境是否存在
    if conda env list | grep -q "^${ENV_NAME} "; then
        print_warning "环境 '$ENV_NAME' 已存在"
        read -p "是否删除并重新创建? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            conda env remove -n $ENV_NAME -y
        else
            print_info "使用已有环境..."
            return 0
        fi
    fi
    
    # 创建环境
    print_info "创建 conda 环境..."
    conda create -n $ENV_NAME python=3.6 -y
    
    # 激活环境
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate $ENV_NAME
    
    # 配置镜像
    print_info "配置清华镜像源..."
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    pip config set global.timeout 300
    
    # 使用 conda 安装核心依赖
    print_info "安装核心依赖..."
    conda install -c conda-forge matplotlib=3.3.4 h5py=2.10.0 -y
    
    # 安装 TensorFlow CPU
    print_info "安装 TensorFlow 1.14.0 (CPU)..."
    pip install tensorflow==1.14.0
    
    # 安装其他依赖
    print_info "安装图像处理库..."
    pip install scikit-image==0.17.2
    pip install opencv-python-headless==4.5.5.64
    pip install imageio
    
    print_success "CPU 环境配置完成"
}

# 验证安装
verify_installation() {
    local ENV_NAME=$1
    
    echo ""
    print_info "验证安装..."
    echo "--------------------------------------"
    
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate $ENV_NAME
    
    python -c "import tensorflow as tf; print('✓ TensorFlow:', tf.__version__)" 2>/dev/null || echo "✗ TensorFlow 安装失败"
    python -c "import numpy as np; print('✓ NumPy:', np.__version__)" 2>/dev/null || echo "✗ NumPy 安装失败"
    python -c "import skimage; print('✓ scikit-image:', skimage.__version__)" 2>/dev/null || echo "✗ scikit-image 安装失败"
    python -c "import h5py; print('✓ h5py:', h5py.__version__)" 2>/dev/null || echo "✗ h5py 安装失败"
    python -c "import cv2; print('✓ OpenCV:', cv2.__version__)" 2>/dev/null || echo "✗ OpenCV 安装失败"
    python -c "import matplotlib; print('✓ matplotlib:', matplotlib.__version__)" 2>/dev/null || echo "✗ matplotlib 安装失败"
    python -c "import scipy; print('✓ scipy:', scipy.__version__)" 2>/dev/null || echo "✗ scipy 安装失败"
    python -c "import PIL; print('✓ Pillow:', PIL.__version__)" 2>/dev/null || echo "✗ Pillow 安装失败"
    python -c "import imageio; print('✓ imageio:', imageio.__version__)" 2>/dev/null || echo "✗ imageio 安装失败"
    
    # GPU 环境额外验证
    if [[ "$ENV_NAME" == "murf_gpu" ]]; then
        echo ""
        print_info "验证 GPU..."
        python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print('✓ 检测到', len(gpus), '个 GPU')
    for gpu in gpus:
        print('  -', gpu.name)
else:
    print('⚠ 未检测到 GPU (将使用 CPU)')
" 2>/dev/null
    fi
    
    echo "--------------------------------------"
}

# 创建激活脚本
create_activate_script() {
    local ENV_NAME=$1
    local SCRIPT_PATH="/home/sh/MURF/activate_${ENV_NAME#murf_}.sh"
    
    if [[ "$ENV_NAME" == "murf_gpu" ]]; then
        cat > "$SCRIPT_PATH" << 'EOF'
#!/bin/bash
# MURF GPU 环境激活脚本
# 使用方法: source activate_gpu.sh

# 初始化 conda
source $(conda info --base)/etc/profile.d/conda.sh

# 激活环境
conda activate murf_gpu

# 配置 CUDA 库路径
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="${SITE_PACKAGES}/nvidia/cudnn/lib:${SITE_PACKAGES}/nvidia/cublas/lib:${LD_LIBRARY_PATH}"

# 验证
echo "✓ cuDNN 路径已配置: ${SITE_PACKAGES}/nvidia/cudnn/lib"
echo "✓ cuBLAS 路径已配置: ${SITE_PACKAGES}/nvidia/cublas/lib"
echo ""
echo "验证 TensorFlow GPU..."
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print('✓ 检测到', len(gpus), '个 GPU')
    for gpu in gpus:
        print('  -', gpu.name)
else:
    print('⚠ 未检测到 GPU')
"
echo ""
echo "环境已就绪！可以运行:"
echo "  cd /home/sh/MURF/RGB-IR/fine_registration_and_fusion"
echo "  python test.py"
EOF
        chmod +x "$SCRIPT_PATH"
        print_success "已创建 GPU 激活脚本: $SCRIPT_PATH"
    else
        cat > "$SCRIPT_PATH" << 'EOF'
#!/bin/bash
# MURF CPU 环境激活脚本
# 使用方法: source activate_cpu.sh

# 初始化 conda
source $(conda info --base)/etc/profile.d/conda.sh

# 激活环境
conda activate murf_cpu

echo "✓ CPU 环境已激活"
echo ""
echo "注意: CPU 环境仅支持 Task 1 和 Task 2"
echo ""
echo "测试命令:"
echo "  cd /home/sh/MURF/RGB-IR/shared_information_extraction"
echo "  python test.py"
EOF
        chmod +x "$SCRIPT_PATH"
        print_success "已创建 CPU 激活脚本: $SCRIPT_PATH"
    fi
}

# 打印使用说明
print_usage() {
    local ENV_NAME=$1
    
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}  ✅ 环境配置完成！${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo ""
    echo "使用以下命令激活环境:"
    if [[ "$ENV_NAME" == "murf_gpu" ]]; then
        echo "  source /home/sh/MURF/activate_gpu.sh"
    else
        echo "  source /home/sh/MURF/activate_cpu.sh"
    fi
    echo ""
    echo "或手动激活:"
    echo "  conda activate $ENV_NAME"
    echo ""
    echo "测试命令:"
    echo "  cd /home/sh/MURF"
    echo "  bash run_all_tests.sh"
    echo ""
    echo "============================================"
}

# 主函数
main() {
    print_header
    check_conda
    
    # 初始化 conda
    source $(conda info --base)/etc/profile.d/conda.sh
    
    # 解析参数
    MODE="auto"
    if [ $# -gt 0 ]; then
        case "$1" in
            --gpu)
                MODE="gpu"
                ;;
            --cpu)
                MODE="cpu"
                ;;
            --auto)
                MODE="auto"
                ;;
            *)
                echo "用法: bash setup_env.sh [--gpu | --cpu | --auto]"
                echo "  --gpu   强制配置 GPU 环境"
                echo "  --cpu   强制配置 CPU 环境"
                echo "  --auto  自动检测 (默认)"
                exit 1
                ;;
        esac
    fi
    
    # 自动检测模式
    if [ "$MODE" == "auto" ]; then
        print_info "自动检测 GPU..."
        if detect_gpu; then
            GPU_INFO=$(nvidia-smi -L | head -1)
            print_success "检测到 GPU: $GPU_INFO"
            MODE="gpu"
        else
            print_warning "未检测到 GPU，将配置 CPU 环境"
            MODE="cpu"
        fi
    fi
    
    # 配置环境
    if [ "$MODE" == "gpu" ]; then
        setup_gpu_env
        verify_installation "murf_gpu"
        create_activate_script "murf_gpu"
        print_usage "murf_gpu"
    else
        setup_cpu_env
        verify_installation "murf_cpu"
        create_activate_script "murf_cpu"
        print_usage "murf_cpu"
    fi
}

# 运行主函数
main "$@"
