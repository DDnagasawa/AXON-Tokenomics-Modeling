#!/usr/bin/env python3
"""
测试图表布局修复的简化脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import importlib.util
spec = importlib.util.spec_from_file_location("axon_model", "accurate_estimate/6.6.2.2.py")
axon_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(axon_model)
AXONTokenomicsModel = axon_model.AXONTokenomicsModel
AXONAnalyzer = axon_model.AXONAnalyzer

def test_layout():
    """测试图表布局"""
    print("🚀 测试AXON代币经济学模型...")
    
    # 创建模型
    model = AXONTokenomicsModel()
    
    # 运行短期模拟（减少数据点以加快测试）
    print("📊 运行5年模拟...")
    results_df = model.run_simulation(years=5, steps_per_year=6)  # 30个数据点
    
    # 创建分析器
    analyzer = AXONAnalyzer(results_df)
    
    # 创建仪表板
    print("🎨 创建仪表板...")
    fig = analyzer.create_comprehensive_dashboard(figsize=(50, 35))
    
    # 保存图表
    print("💾 保存图表...")
    fig.savefig('test_layout_dashboard.png', dpi=150, bbox_inches='tight')
    
    print("✅ 测试完成！图表已保存为 test_layout_dashboard.png")
    print("📏 图表尺寸: 50x35 英寸")
    print("📐 子图间距: 水平 0.3, 垂直 0.4")
    
    return fig

if __name__ == "__main__":
    test_layout() 