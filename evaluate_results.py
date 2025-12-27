#!/usr/bin/env python3
"""
MURF 论文复现结果评估脚本
根据论文中的评价方法评估所有融合结果

评价指标:
1. MI (Mutual Information) - 互信息
2. SSIM (Structural Similarity Index) - 结构相似性
3. CC (Correlation Coefficient) - 相关系数
4. PSNR (Peak Signal-to-Noise Ratio) - 峰值信噪比
5. EN (Entropy) - 信息熵
6. SF (Spatial Frequency) - 空间频率
7. SD (Standard Deviation) - 标准差
"""

import numpy as np
from PIL import Image
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')


def load_image(path, mode='L'):
    """加载图像并转换为指定模式"""
    if not os.path.exists(path):
        return None
    img = Image.open(path)
    if mode:
        img = img.convert(mode)
    return np.array(img).astype(np.float64)


def entropy(img):
    """计算图像信息熵"""
    hist, _ = np.histogram(img.flatten(), bins=256, range=[0, 256], density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))


def mutual_information(img1, img2, bins=256):
    """计算互信息 MI"""
    hist_2d, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=bins)
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    return np.sum(pxy[nzs] * np.log2(pxy[nzs] / px_py[nzs]))


def correlation_coefficient(img1, img2):
    """计算相关系数 CC"""
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()
    return np.corrcoef(img1_flat, img2_flat)[0, 1]


def spatial_frequency(img):
    """计算空间频率 SF"""
    rows, cols = img.shape
    rf = np.sqrt(np.mean(np.diff(img, axis=1) ** 2))  # 行频率
    cf = np.sqrt(np.mean(np.diff(img, axis=0) ** 2))  # 列频率
    return np.sqrt(rf ** 2 + cf ** 2)


def standard_deviation(img):
    """计算标准差 SD"""
    return np.std(img)


def average_gradient(img):
    """计算平均梯度 AG"""
    gx = ndimage.sobel(img, axis=1)
    gy = ndimage.sobel(img, axis=0)
    return np.mean(np.sqrt(gx ** 2 + gy ** 2))


def evaluate_fusion(source1_path, source2_path, fused_path, label=""):
    """评估单个融合结果"""
    source1 = load_image(source1_path, mode='L')
    source2 = load_image(source2_path, mode='L')
    fused = load_image(fused_path, mode='L')
    
    if source1 is None or source2 is None or fused is None:
        print(f"  无法加载图像: {label}")
        return None
    
    # 确保尺寸一致
    min_h = min(source1.shape[0], source2.shape[0], fused.shape[0])
    min_w = min(source1.shape[1], source2.shape[1], fused.shape[1])
    source1 = source1[:min_h, :min_w]
    source2 = source2[:min_h, :min_w]
    fused = fused[:min_h, :min_w]
    
    results = {}
    
    # 互信息 (与两个源图像)
    mi1 = mutual_information(source1, fused)
    mi2 = mutual_information(source2, fused)
    results['MI_avg'] = (mi1 + mi2) / 2
    results['MI_src1'] = mi1
    results['MI_src2'] = mi2
    
    # SSIM (与两个源图像)
    ssim1 = ssim(source1, fused, data_range=255)
    ssim2 = ssim(source2, fused, data_range=255)
    results['SSIM_avg'] = (ssim1 + ssim2) / 2
    results['SSIM_src1'] = ssim1
    results['SSIM_src2'] = ssim2
    
    # CC (与两个源图像)
    cc1 = correlation_coefficient(source1, fused)
    cc2 = correlation_coefficient(source2, fused)
    results['CC_avg'] = (cc1 + cc2) / 2
    results['CC_src1'] = cc1
    results['CC_src2'] = cc2
    
    # 融合图像质量指标
    results['EN'] = entropy(fused)
    results['SF'] = spatial_frequency(fused)
    results['SD'] = standard_deviation(fused)
    results['AG'] = average_gradient(fused)
    
    return results


def print_results(name, results):
    """打印评估结果"""
    if results is None:
        return
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  互信息 MI (avg):     {results['MI_avg']:.4f}")
    print(f"  结构相似性 SSIM (avg): {results['SSIM_avg']:.4f}")
    print(f"  相关系数 CC (avg):   {results['CC_avg']:.4f}")
    print(f"  信息熵 EN:           {results['EN']:.4f}")
    print(f"  空间频率 SF:         {results['SF']:.4f}")
    print(f"  标准差 SD:           {results['SD']:.4f}")
    print(f"  平均梯度 AG:         {results['AG']:.4f}")


def evaluate_rgb_ir():
    """评估 RGB-IR 融合结果"""
    print("\n" + "="*70)
    print("  RGB-IR 融合评估")
    print("="*70)
    
    base_path = '/home/sh/MURF/RGB-IR/fine_registration_and_fusion'
    rgb_path = os.path.join(base_path, 'test_imgs/RGB')
    ir_path = os.path.join(base_path, 'test_imgs/IR')
    fused_path = os.path.join(base_path, 'results/fused_img')
    
    all_results = []
    
    if os.path.exists(fused_path):
        for f in os.listdir(fused_path):
            if f.endswith(('.png', '.jpg')):
                src1 = os.path.join(rgb_path, f)
                src2 = os.path.join(ir_path, f)
                fused = os.path.join(fused_path, f)
                
                results = evaluate_fusion(src1, src2, fused, f"RGB-IR: {f}")
                if results:
                    all_results.append(results)
                    print_results(f"RGB-IR: {f}", results)
    
    return all_results


def evaluate_pet_mri():
    """评估 PET-MRI 融合结果"""
    print("\n" + "="*70)
    print("  PET-MRI 融合评估")
    print("="*70)
    
    base_path = '/home/sh/MURF/PET-MRI/fine_registration_and_fusion'
    pet_path = os.path.join(base_path, 'test_imgs/warped_PET')
    mri_path = os.path.join(base_path, 'test_imgs/MRI')
    fused_path = os.path.join(base_path, 'results/Fusion')
    
    all_results = []
    
    if os.path.exists(fused_path):
        for f in os.listdir(fused_path):
            if f.endswith(('.png', '.jpg')):
                src1 = os.path.join(pet_path, f.replace('.png', '.png'))
                src2 = os.path.join(mri_path, f.replace('.png', '.png'))
                fused = os.path.join(fused_path, f)
                
                results = evaluate_fusion(src1, src2, fused, f"PET-MRI: {f}")
                if results:
                    all_results.append(results)
                    print_results(f"PET-MRI: {f}", results)
    
    return all_results


def evaluate_ct_mri():
    """评估 CT-MRI 融合结果"""
    print("\n" + "="*70)
    print("  CT-MRI 融合评估")
    print("="*70)
    
    base_path = '/home/sh/MURF/CT-MRI/fine_registration_and_fusion'
    ct_path = os.path.join(base_path, 'test_imgs/warped_CT')
    mri_path = os.path.join(base_path, 'test_imgs/MRI')
    fused_path = os.path.join(base_path, 'results/Fusion')
    
    all_results = []
    
    if os.path.exists(fused_path):
        for f in os.listdir(fused_path):
            if f.endswith(('.png', '.jpg')):
                src1 = os.path.join(ct_path, f.replace('.png', '.png'))
                src2 = os.path.join(mri_path, f.replace('.png', '.png'))
                fused = os.path.join(fused_path, f)
                
                results = evaluate_fusion(src1, src2, fused, f"CT-MRI: {f}")
                if results:
                    all_results.append(results)
                    print_results(f"CT-MRI: {f}", results)
    
    return all_results


def calculate_average(all_results):
    """计算平均指标"""
    if not all_results:
        return None
    
    avg = {}
    keys = all_results[0].keys()
    for key in keys:
        values = [r[key] for r in all_results if key in r]
        if values:
            avg[key] = np.mean(values)
    return avg


def main():
    print("\n" + "#"*70)
    print("#  MURF 论文复现 - 融合结果评估")
    print("#  评价指标: MI, SSIM, CC, EN, SF, SD, AG")
    print("#"*70)
    
    # RGB-IR
    rgb_ir_results = evaluate_rgb_ir()
    if rgb_ir_results:
        avg = calculate_average(rgb_ir_results)
        print_results("RGB-IR 平均结果", avg)
    
    # PET-MRI
    pet_mri_results = evaluate_pet_mri()
    if pet_mri_results:
        avg = calculate_average(pet_mri_results)
        print_results("PET-MRI 平均结果", avg)
    
    # CT-MRI
    ct_mri_results = evaluate_ct_mri()
    if ct_mri_results:
        avg = calculate_average(ct_mri_results)
        print_results("CT-MRI 平均结果", avg)
    
    print("\n" + "="*70)
    print("  评估完成!")
    print("="*70)
    
    # 汇总表格
    print("\n" + "="*80)
    print("  综合评估结果汇总表")
    print("="*80)
    print(f"{'模态':<12} {'MI':>8} {'SSIM':>8} {'CC':>8} {'EN':>8} {'SF':>8} {'SD':>8} {'AG':>8}")
    print("-"*80)
    
    if rgb_ir_results:
        avg = calculate_average(rgb_ir_results)
        print(f"{'RGB-IR':<12} {avg['MI_avg']:>8.4f} {avg['SSIM_avg']:>8.4f} {avg['CC_avg']:>8.4f} "
              f"{avg['EN']:>8.4f} {avg['SF']:>8.4f} {avg['SD']:>8.4f} {avg['AG']:>8.4f}")
    
    if pet_mri_results:
        avg = calculate_average(pet_mri_results)
        print(f"{'PET-MRI':<12} {avg['MI_avg']:>8.4f} {avg['SSIM_avg']:>8.4f} {avg['CC_avg']:>8.4f} "
              f"{avg['EN']:>8.4f} {avg['SF']:>8.4f} {avg['SD']:>8.4f} {avg['AG']:>8.4f}")
    
    if ct_mri_results:
        avg = calculate_average(ct_mri_results)
        print(f"{'CT-MRI':<12} {avg['MI_avg']:>8.4f} {avg['SSIM_avg']:>8.4f} {avg['CC_avg']:>8.4f} "
              f"{avg['EN']:>8.4f} {avg['SF']:>8.4f} {avg['SD']:>8.4f} {avg['AG']:>8.4f}")
    
    print("="*80)


if __name__ == '__main__':
    main()
