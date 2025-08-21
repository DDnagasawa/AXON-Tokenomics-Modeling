# AXON Tokenomics Mathematical Modeling

基于AXON Tokenomics白皮书的数学建模项目

## 项目简介

本项目根据AXON Tokenomics白皮书进行数学建模，实现了代币经济学核心概念的数值化分析和可视化。

## 主要功能

- **代币经济学架构分析** - 基于白皮书第3章核心概念
- **挖矿经济模拟器** - 计算代币价格、盈利能力和回本周期
- **风险敏感性分析** - 动态权重和风险评估
- **可视化报告** - 生成完整的分析图表

## 文件结构

```
modeling/
├── accurate_estimate/
│   ├── v.1.2_sen&risk.py          # 风险敏感性分析
│   ├── Mining_TkPrice.py          # 挖矿经济模拟器
│   └── 6.6.2.2.py                 # 其他建模文件
├── PNG/                           # 生成的图表
└── README.md
```

## 使用方法

1. 安装依赖：
   ```bash
   pip install matplotlib seaborn pandas networkx
   ```

2. 运行分析：
   ```bash
   python accurate_estimate/v.1.2_sen&risk.py
   python accurate_estimate/Mining_TkPrice.py
   ```

## 技术栈

- Python
- Matplotlib/Seaborn (数据可视化)
- Pandas (数据处理)
- NetworkX (图论分析)

---

*基于AXON Tokenomics白皮书开发的数学建模工具*
