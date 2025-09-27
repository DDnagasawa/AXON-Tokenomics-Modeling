#!/usr/bin/env python3
"""
Revenue Analysis 测试运行器
提供多种测试运行选项
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

def run_python_tests():
    """运行Python单元测试"""
    print("🚀 运行 Revenue Analysis Python 单元测试...")
    
    test_file = Path(__file__).parent / "test_revenue_analysis.py"
    
    try:
        # 直接运行测试文件
        result = subprocess.run([sys.executable, str(test_file)], 
                              capture_output=True, text=True, cwd=Path(__file__).parent)
        
        print(result.stdout)
        if result.stderr:
            print("错误输出:", result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ 运行测试时出错: {e}")
        return False

def run_unittest_framework():
    """使用unittest框架运行测试"""
    print("🚀 使用 unittest 框架运行测试...")
    
    try:
        # 添加测试目录到Python路径
        test_dir = Path(__file__).parent
        sys.path.insert(0, str(test_dir))
        
        # 导入并运行测试
        from test_revenue_analysis import TestRevenueAnalysis
        
        import unittest
        suite = unittest.TestLoader().loadTestsFromTestCase(TestRevenueAnalysis)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()
        
    except Exception as e:
        print(f"❌ 运行unittest时出错: {e}")
        return False

def run_individual_test(test_name):
    """运行单个测试"""
    print(f"🚀 运行单个测试: {test_name}")
    
    try:
        test_dir = Path(__file__).parent
        sys.path.insert(0, str(test_dir))
        
        from test_revenue_analysis import TestRevenueAnalysis
        
        test_instance = TestRevenueAnalysis()
        test_instance.setUp()
        
        if hasattr(test_instance, test_name):
            test_method = getattr(test_instance, test_name)
            test_method()
            print(f"✅ {test_name} - 通过")
            return True
        else:
            print(f"❌ 测试方法 {test_name} 不存在")
            return False
            
    except Exception as e:
        print(f"❌ 运行测试 {test_name} 时出错: {e}")
        return False

def list_available_tests():
    """列出所有可用的测试"""
    print("📋 可用的测试方法:")
    
    test_methods = [
        "test_emission_rate_calculation",
        "test_block_reward_calculation", 
        "test_miner_growth_model",
        "test_network_participant_modeling",
        "test_dynamic_market_cap",
        "test_token_price_calculation",
        "test_payback_period_calculation",
        "test_edge_cases",
        "test_data_validation"
    ]
    
    for i, method in enumerate(test_methods, 1):
        print(f"  {i}. {method}")
    
    print(f"\n使用方法: python run_tests.py --test {test_methods[0]}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Revenue Analysis 测试运行器")
    parser.add_argument("--method", choices=["python", "unittest"], default="python",
                       help="选择测试运行方法")
    parser.add_argument("--test", type=str, help="运行单个测试方法")
    parser.add_argument("--list", action="store_true", help="列出所有可用测试")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Revenue Analysis 测试运行器")
    print("=" * 60)
    
    if args.list:
        list_available_tests()
        return
    
    if args.test:
        success = run_individual_test(args.test)
    elif args.method == "unittest":
        success = run_unittest_framework()
    else:
        success = run_python_tests()
    
    if success:
        print("\n🎉 测试运行成功！")
        sys.exit(0)
    else:
        print("\n⚠️ 测试运行失败！")
        sys.exit(1)

if __name__ == "__main__":
    main()

