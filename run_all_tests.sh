#!/bin/bash
# ============================================================
# MURF 一键测试脚本
# 测试所有 11 个可用任务 (RGB-NIR Task 3 无预训练模型)
# 使用方法: bash run_all_tests.sh [--quick | --full]
# ============================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 项目根目录
PROJECT_ROOT="/home/sh/MURF"

# 测试结果数组
declare -A TEST_RESULTS
declare -A TEST_TIMES

# 计时函数
get_timestamp() {
    date +%s
}

format_duration() {
    local seconds=$1
    if [ $seconds -lt 60 ]; then
        echo "${seconds}s"
    else
        echo "$((seconds/60))m $((seconds%60))s"
    fi
}

print_header() {
    echo ""
    echo -e "${CYAN}================================================================${NC}"
    echo -e "${CYAN}  MURF 论文复现 - 一键测试脚本${NC}"
    echo -e "${CYAN}  测试任务: 11/12 (RGB-NIR Task 3 无预训练模型)${NC}"
    echo -e "${CYAN}================================================================${NC}"
    echo ""
}

print_task_header() {
    local module=$1
    local task=$2
    local desc=$3
    echo ""
    echo -e "${BLUE}────────────────────────────────────────────────────────────────${NC}"
    echo -e "${BLUE}  $module - $task: $desc${NC}"
    echo -e "${BLUE}────────────────────────────────────────────────────────────────${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}→ $1${NC}"
}

# 检查并激活环境
setup_environment() {
    print_info "检查环境..."
    
    # 初始化 conda
    source $(conda info --base)/etc/profile.d/conda.sh
    
    # 优先使用 GPU 环境
    if conda env list | grep -q "murf_gpu"; then
        conda activate murf_gpu
        # 配置 CUDA 路径
        SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
        export LD_LIBRARY_PATH="${SITE_PACKAGES}/nvidia/cudnn/lib:${SITE_PACKAGES}/nvidia/cublas/lib:${LD_LIBRARY_PATH}"
        print_success "已激活 GPU 环境: murf_gpu"
    elif conda env list | grep -q "murf_cpu"; then
        conda activate murf_cpu
        print_warning "已激活 CPU 环境: murf_cpu (Task 3 可能无法运行)"
    else
        print_error "未找到 MURF 环境，请先运行 setup_env.sh"
        exit 1
    fi
}

# 运行单个测试
run_test() {
    local module=$1
    local task_dir=$2
    local task_name=$3
    local task_desc=$4
    local key="${module}_${task_name}"
    
    print_task_header "$module" "$task_name" "$task_desc"
    
    local task_path="${PROJECT_ROOT}/${module}/${task_dir}"
    
    if [ ! -d "$task_path" ]; then
        print_error "目录不存在: $task_path"
        TEST_RESULTS[$key]="SKIP"
        return
    fi
    
    cd "$task_path"
    
    if [ ! -f "test.py" ]; then
        print_error "测试脚本不存在: test.py"
        TEST_RESULTS[$key]="SKIP"
        return
    fi
    
    print_info "工作目录: $task_path"
    print_info "运行测试..."
    
    local start_time=$(get_timestamp)
    
    # 运行测试
    if python test.py 2>&1 | tee /tmp/murf_test_output.log; then
        local end_time=$(get_timestamp)
        local duration=$((end_time - start_time))
        TEST_RESULTS[$key]="PASS"
        TEST_TIMES[$key]=$duration
        print_success "测试通过 (耗时: $(format_duration $duration))"
    else
        local end_time=$(get_timestamp)
        local duration=$((end_time - start_time))
        TEST_RESULTS[$key]="FAIL"
        TEST_TIMES[$key]=$duration
        print_error "测试失败 (耗时: $(format_duration $duration))"
        # 显示错误信息
        tail -20 /tmp/murf_test_output.log
    fi
}

# 打印测试汇总
print_summary() {
    echo ""
    echo -e "${CYAN}================================================================${NC}"
    echo -e "${CYAN}  测试结果汇总${NC}"
    echo -e "${CYAN}================================================================${NC}"
    echo ""
    
    local pass_count=0
    local fail_count=0
    local skip_count=0
    
    printf "%-12s %-8s %-8s %-8s %-10s\n" "模块" "Task1" "Task2" "Task3" "状态"
    echo "────────────────────────────────────────────────────────"
    
    # RGB-IR
    local t1="${TEST_RESULTS[RGB-IR_Task1]:-N/A}"
    local t2="${TEST_RESULTS[RGB-IR_Task2]:-N/A}"
    local t3="${TEST_RESULTS[RGB-IR_Task3]:-N/A}"
    printf "%-12s %-8s %-8s %-8s\n" "RGB-IR" "$t1" "$t2" "$t3"
    
    # RGB-NIR
    t1="${TEST_RESULTS[RGB-NIR_Task1]:-N/A}"
    t2="${TEST_RESULTS[RGB-NIR_Task2]:-N/A}"
    t3="SKIP"  # 无模型
    printf "%-12s %-8s %-8s %-8s\n" "RGB-NIR" "$t1" "$t2" "$t3"
    
    # PET-MRI
    t1="${TEST_RESULTS[PET-MRI_Task1]:-N/A}"
    t2="${TEST_RESULTS[PET-MRI_Task2]:-N/A}"
    t3="${TEST_RESULTS[PET-MRI_Task3]:-N/A}"
    printf "%-12s %-8s %-8s %-8s\n" "PET-MRI" "$t1" "$t2" "$t3"
    
    # CT-MRI
    t1="${TEST_RESULTS[CT-MRI_Task1]:-N/A}"
    t2="${TEST_RESULTS[CT-MRI_Task2]:-N/A}"
    t3="${TEST_RESULTS[CT-MRI_Task3]:-N/A}"
    printf "%-12s %-8s %-8s %-8s\n" "CT-MRI" "$t1" "$t2" "$t3"
    
    echo "────────────────────────────────────────────────────────"
    
    # 统计
    for key in "${!TEST_RESULTS[@]}"; do
        case "${TEST_RESULTS[$key]}" in
            PASS) ((pass_count++)) ;;
            FAIL) ((fail_count++)) ;;
            SKIP) ((skip_count++)) ;;
        esac
    done
    
    echo ""
    echo -e "通过: ${GREEN}${pass_count}${NC}  失败: ${RED}${fail_count}${NC}  跳过: ${YELLOW}${skip_count}${NC}"
    echo ""
    
    if [ $fail_count -eq 0 ]; then
        echo -e "${GREEN}✓ 所有测试通过！${NC}"
    else
        echo -e "${RED}✗ 有 ${fail_count} 个测试失败${NC}"
    fi
    
    echo ""
}

# 快速测试 (每个模态只测试 Task 1)
quick_test() {
    print_info "快速测试模式: 每个模态仅测试 Task 1"
    echo ""
    
    run_test "RGB-IR" "shared_information_extraction" "Task1" "共享信息提取"
    run_test "RGB-NIR" "shared_information_extraction" "Task1" "共享信息提取"
    run_test "PET-MRI" "shared_information_extraction" "Task1" "共享信息提取"
    run_test "CT-MRI" "shared_information_extraction" "Task1" "共享信息提取"
}

# 完整测试 (所有 11 个任务)
full_test() {
    print_info "完整测试模式: 测试所有 11 个任务"
    echo ""
    
    # RGB-IR (3 个任务)
    run_test "RGB-IR" "shared_information_extraction" "Task1" "共享信息提取"
    run_test "RGB-IR" "multi-scale_coarse_registration" "Task2" "多尺度粗配准"
    run_test "RGB-IR" "fine_registration_and_fusion" "Task3" "精细配准与融合"
    
    # RGB-NIR (2 个任务, Task 3 无模型)
    run_test "RGB-NIR" "shared_information_extraction" "Task1" "共享信息提取"
    run_test "RGB-NIR" "multi-scale_coarse_registration" "Task2" "多尺度粗配准"
    print_warning "跳过 RGB-NIR Task3: 无预训练模型"
    TEST_RESULTS["RGB-NIR_Task3"]="SKIP"
    
    # PET-MRI (3 个任务)
    run_test "PET-MRI" "shared_information_extraction" "Task1" "共享信息提取"
    run_test "PET-MRI" "multi-scale_coarse_registration" "Task2" "多尺度粗配准"
    run_test "PET-MRI" "fine_registration_and_fusion" "Task3" "精细配准与融合"
    
    # CT-MRI (3 个任务)
    run_test "CT-MRI" "shared_information_extraction" "Task1" "共享信息提取"
    run_test "CT-MRI" "multi-scale_coarse_registration" "Task2" "多尺度粗配准"
    run_test "CT-MRI" "fine_registration_and_fusion" "Task3" "精细配准与融合"
}

# 主函数
main() {
    print_header
    setup_environment
    
    # 解析参数
    MODE="full"
    if [ $# -gt 0 ]; then
        case "$1" in
            --quick|-q)
                MODE="quick"
                ;;
            --full|-f)
                MODE="full"
                ;;
            --help|-h)
                echo "用法: bash run_all_tests.sh [选项]"
                echo ""
                echo "选项:"
                echo "  --quick, -q    快速测试 (仅 Task 1)"
                echo "  --full, -f     完整测试 (所有 11 个任务, 默认)"
                echo "  --help, -h     显示帮助信息"
                exit 0
                ;;
            *)
                echo "未知选项: $1"
                echo "使用 --help 查看帮助"
                exit 1
                ;;
        esac
    fi
    
    # 运行测试
    local total_start=$(get_timestamp)
    
    if [ "$MODE" == "quick" ]; then
        quick_test
    else
        full_test
    fi
    
    local total_end=$(get_timestamp)
    local total_duration=$((total_end - total_start))
    
    # 打印汇总
    print_summary
    
    echo "总耗时: $(format_duration $total_duration)"
    echo ""
    
    # 运行评估
    echo -e "${BLUE}是否运行融合结果评估? (y/n)${NC}"
    read -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd "$PROJECT_ROOT"
        python evaluate_results.py
    fi
}

# 运行主函数
main "$@"
