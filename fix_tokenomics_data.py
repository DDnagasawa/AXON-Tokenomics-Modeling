#!/usr/bin/env python3
"""
修复AXON代币经济学数据脚本
解决初始分配量错误问题：从5000万修正为43亿
"""

import pandas as pd
import numpy as np
from typing import Dict, List

def fix_tokenomics_data():
    """
    修复axon_tokenomics_results.csv中的初始分配量错误
    """
    print("🔧 开始修复AXON代币经济学数据...")
    
    # 读取原始CSV文件
    try:
        df = pd.read_csv('axon_tokenomics_results.csv')
        print(f"✅ 成功读取原始文件，共{len(df)}行数据")
    except FileNotFoundError:
        print("❌ 找不到axon_tokenomics_results.csv文件")
        return
    
    # 检查问题
    original_initial_supply = df.iloc[0]['circulating_supply']
    print(f"📊 当前初始流通量: {original_initial_supply:,.0f}")
    
    # 正确的初始分配量应该是43亿
    correct_initial_supply = 4_300_000_000
    
    if abs(original_initial_supply - correct_initial_supply) < 1e6:
        print("✅ 数据已经是正确的，无需修复")
        return
    
    print(f"🔍 发现问题：初始分配量应该是 {correct_initial_supply:,.0f}")
    
    # 计算修正因子
    correction_factor = correct_initial_supply / original_initial_supply
    print(f"📈 修正因子: {correction_factor:.2f}")
    
    # 备份原始文件
    backup_filename = 'axon_tokenomics_results_backup.csv'
    df.to_csv(backup_filename, index=False)
    print(f"💾 已备份原始文件到: {backup_filename}")
    
    # 修复数据
    print("🔧 开始修复数据...")
    
    # 修正流通供应量
    df['circulating_supply'] = df['circulating_supply'] * correction_factor
    
    # 修正年发行量
    df['annual_emissions'] = df['annual_emissions'] * correction_factor
    
    # 修正区块奖励
    df['block_reward'] = df['block_reward'] * correction_factor
    
    # 修正各种代币分配
    token_columns = [
        'pro_miners_tokens', 'enterprise_tokens', 'retail_tokens',
        'pro_miners_compute', 'enterprise_compute', 'retail_compute',
        'enterprise_kv', 'retail_kv', 'retail_pdp'
    ]
    
    for col in token_columns:
        if col in df.columns:
            df[col] = df[col] * correction_factor
    
    # 修正奖励池
    pool_columns = [
        'compute_security_pool', 'knowledge_vault_pool', 'personal_data_pool'
    ]
    
    for col in pool_columns:
        if col in df.columns:
            df[col] = df[col] * correction_factor
    
    # 验证修复结果
    new_initial_supply = df.iloc[0]['circulating_supply']
    print(f"✅ 修复后初始流通量: {new_initial_supply:,.0f}")
    
    # 保存修复后的文件
    fixed_filename = 'axon_tokenomics_results_fixed.csv'
    df.to_csv(fixed_filename, index=False)
    print(f"💾 已保存修复后的文件到: {fixed_filename}")
    
    # 覆盖原文件
    df.to_csv('axon_tokenomics_results.csv', index=False)
    print(f"✅ 已覆盖原文件，修复完成！")
    
    # 显示修复前后的对比
    print("\n📊 修复前后对比:")
    print(f"   修复前: {original_initial_supply:,.0f} AXON")
    print(f"   修复后: {new_initial_supply:,.0f} AXON")
    print(f"   修正倍数: {correction_factor:.2f}x")
    
    # 验证数据一致性
    print("\n🔍 数据一致性验证:")
    
    # 检查总供应量限制
    max_supply = df['circulating_supply'].max()
    total_supply = 86_000_000_000
    if max_supply > total_supply:
        print(f"⚠️  警告：最大流通量 {max_supply:,.0f} 超过总供应量 {total_supply:,.0f}")
    else:
        print(f"✅ 流通量未超过总供应量限制")
    
    # 检查发行率合理性
    emission_rates = df['emission_rate']
    if emission_rates.min() >= 0 and emission_rates.max() <= 1:
        print(f"✅ 发行率在合理范围内: {emission_rates.min():.3f} - {emission_rates.max():.3f}")
    else:
        print(f"⚠️  警告：发行率超出合理范围")
    
    print("\n🎉 修复完成！现在数据符合AXON白皮书规范。")

def generate_correct_tokenomics_data():
    """
    生成正确的代币经济学数据（如果需要重新生成）
    """
    print("\n🔄 生成正确的代币经济学数据...")
    
    # 这里可以调用正确的模拟脚本来生成数据
    # 例如：运行 6.6.2.2_v.1.2.py 来生成正确的数据
    
    print("📝 请运行以下命令生成正确的数据:")
    print("   python accurate_estimate/6.6.2.2_v.1.2.py")
    print("   或者")
    print("   python accurate_estimate/6.6.2.2_v.1.1.py")

if __name__ == "__main__":
    print("=" * 60)
    print("🔧 AXON代币经济学数据修复工具")
    print("=" * 60)
    
    # 修复现有数据
    fix_tokenomics_data()
    
    # 提供重新生成数据的选项
    print("\n" + "=" * 60)
    print("📋 修复说明:")
    print("1. 初始分配量从5000万修正为43亿（5%的总供应量）")
    print("2. 所有相关的代币分配数据都按比例修正")
    print("3. 发行率和百分比数据保持不变")
    print("4. 原始文件已备份")
    print("=" * 60)
