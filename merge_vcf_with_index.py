#!/usr/bin/env python3
"""
修复版的VCF合并脚本，包含索引创建步骤
"""

import subprocess
import cyvcf2
import numpy as np

def create_index_for_vcf(vcf_file):
    """为VCF文件创建索引"""
    cmd = f"bcftools index {vcf_file}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"索引创建失败 {vcf_file}: {result.stderr}")
        return False
    else:
        print(f"索引创建成功: {vcf_file}")
        return True

def merge_vcf_files_with_index():
    """合并VCF文件并处理索引"""
    print("=== 开始VCF合并流程 ===")
    
    # 1. 确保所有VCF文件都有索引
    vcf_files = [
        "AADR_common.chr22.vcf.gz",
        "Shimao_common.chr22.vcf.gz", 
        "Tibet_common.chr22.vcf.gz"
    ]
    
    print("\n1. 检查并创建索引...")
    for vcf_file in vcf_files:
        index_file = vcf_file + ".csi"
        if not os.path.exists(index_file):
            print(f"创建索引: {vcf_file}")
            if not create_index_for_vcf(vcf_file):
                return False
        else:
            print(f"索引已存在: {vcf_file}")
    
    # 2. 创建文件列表
    print("\n2. 创建文件列表...")
    filelist_path = "merge_filelist.txt"
    with open(filelist_path, 'w') as f:
        for vcf_file in vcf_files:
            f.write(f"{vcf_file}\n")
    
    # 3. 合并VCF文件
    print("\n3. 合并VCF文件...")
    output_vcf = "merged_15326_variants_2364_samples.chr22.vcf.gz"
    merge_cmd = f"bcftools merge -l {filelist_path} -O z -o {output_vcf}"
    
    print(f"执行命令: {merge_cmd}")
    result = subprocess.run(merge_cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"合并失败:")
        print(f"错误信息: {result.stderr}")
        return False
    
    print(f"合并成功: {output_vcf}")
    
    # 4. 创建最终索引
    print("\n4. 创建最终索引...")
    if not create_index_for_vcf(output_vcf):
        return False
    
    # 5. 验证结果
    print("\n5. 验证合并结果...")
    stats = get_final_vcf_stats(output_vcf)
    
    # 清理临时文件
    if os.path.exists(filelist_path):
        os.remove(filelist_path)
    
    return True, output_vcf, stats

def get_final_vcf_stats(vcf_path):
    """获取最终合并VCF的统计信息"""
    print(f"\n=== 最终VCF统计信息 ===")
    
    vcf = cyvcf2.VCF(vcf_path)
    
    # 样本信息
    n_samples = len(vcf.samples)
    print(f"总样本数: {n_samples}")
    
    # 显示各数据集样本的分布
    aadr_samples = [s for s in vcf.samples if s.startswith('I')]
    shimao_samples = [s for s in vcf.samples if s.startswith('HRR194')]
    tibet_samples = [s for s in vcf.samples if s.startswith('HRR58')]
    
    print(f"AADR样本数: {len(aadr_samples)}")
    print(f"石峁样本数: {len(shimao_samples)}")
    print(f"西藏样本数: {len(tibet_samples)}")
    
    # 位点信息
    variant_count = 0
    missing_rates = []
    
    for variant in vcf:
        variant_count += 1
        gt_types = variant.gt_types
        missing_count = np.sum(gt_types == 3)
        missing_rate = missing_count / n_samples
        missing_rates.append(missing_rate)
        
        if variant_count % 5000 == 0:
            print(f"已处理 {variant_count} 个位点...")
    
    vcf.close()
    
    avg_missing = np.mean(missing_rates)
    median_missing = np.median(missing_rates)
    
    print(f"总位点数: {variant_count}")
    print(f"平均缺失率: {avg_missing:.4f} ({avg_missing*100:.2f}%)")
    print(f"中位缺失率: {median_missing:.4f} ({median_missing*100:.2f}%)")
    
    return {
        'n_samples': n_samples,
        'n_variants': variant_count,
        'avg_missing_rate': avg_missing,
        'median_missing_rate': median_missing,
        'aadr_samples': len(aadr_samples),
        'shimao_samples': len(shimao_samples),
        'tibet_samples': len(tibet_samples)
    }

if __name__ == "__main__":
    import os
    work_dir = "/mnt/qmtang/EvoFill_data/20251230_chr22/augment"
    os.chdir(work_dir)
    
    success, output_file, stats = merge_vcf_files_with_index()
    
    if success:
        print(f"\n=== 合并成功完成 ===")
        print(f"输出文件: {output_file}")
        print(f"总样本数: {stats['n_samples']}")
        print(f"  - AADR东亚样本: {stats['aadr_samples']}")
        print(f"  - 石峁样本: {stats['shimao_samples']}")
        print(f"  - 西藏样本: {stats['tibet_samples']}")
        print(f"位点数: {stats['n_variants']}")
        print(f"平均缺失率: {stats['avg_missing_rate']:.4f}")
        
        # 验证数量
        expected_total = 1917 + 147 + 300
        if stats['n_samples'] == expected_total:
            print(f"✓ 样本数量完全正确")
        else:
            print(f"✗ 样本数量不匹配: 期望{expected_total}, 实际{stats['n_samples']}")
            
    else:
        print("合并失败")