# AXON Tokenomics Mathematical Modeling

基于AXON Tokenomics白皮书的数学建模项目

## 项目简介

本项目根据AXON Tokenomics白皮书进行数学建模，实现了代币经济学核心概念的数值化分析和可视化。项目包含交互式仪表板、矿场投资分析、参数敏感性分析等完整功能。

## 主要功能

- **交互式仪表板** - 基于Chart.js的实时数据分析界面
- **Revenue Analysis模块** - 基于白皮书公式的收益分析
- **矿场投资分析** - 50,000台→300,000台矿机规模增长模拟
- **参数敏感性分析** - 多维度参数影响评估
- **Python分析工具** - 独立的Python分析脚本
- **可视化报告** - 生成完整的分析图表和HTML报告

## 文件结构

```
modeling/
├── dashboard/                          # 交互式仪表板
│   ├── dashboard_v.1.html            # 主仪表板文件
│   ├── Parameter_Analysis.html        # 参数分析报告
│   ├── requirements.txt               # Python依赖
│   └── README.md                     # 仪表板说明
├── accurate_estimate/                 # 精确估算模块
│   ├── v.1.2_sen&risk.py            # 风险敏感性分析
│   ├── Mining_TkPrice.py             # 挖矿经济模拟器
│   ├── 6.6.2.2_v.1.2.py             # 白皮书验证模块
│   └── *.csv, *.png                  # 分析结果文件
├── tests/                             # 测试框架
│   ├── test_revenue_analysis.py      # Revenue Analysis测试
│   ├── test_config.json              # 测试配置
│   └── run_tests.py                  # 测试运行器
├── PNG/                              # 生成的图表
├── *.csv, *.png                      # 分析结果文件
└── README.md                         # 项目说明
```

## 快速开始

### 1. 交互式仪表板
```bash
cd dashboard
python3 -m http.server 8000
# 访问 http://localhost:8000/dashboard_v.1.html
```

### 2. Python分析工具
```bash
# 安装依赖
pip install matplotlib seaborn pandas networkx scipy openpyxl

# 运行风险敏感性分析
python accurate_estimate/v.1.2_sen&risk.py

# 运行挖矿经济模拟器
python accurate_estimate/Mining_TkPrice.py

# 运行测试
python tests/run_tests.py
```

### 3. 参数分析报告
```bash
cd dashboard
python3 -m http.server 8000
# 访问 http://localhost:8000/Parameter_Analysis.html
```

## 核心特性

### 📊 交互式仪表板
- **Payback Period Analysis** - 回本周期分析
- **Nonlinear Revenue Analysis** - 非线性收益分析  
- **Revenue Analysis** - 基于白皮书的收益分析
- **实时参数调整** - 动态更新分析结果

### 🏭 矿场投资分析
- **规模增长模型** - 50,000台→300,000台矿机
- **S曲线增长** - 24个月增长周期
- **成本收益分析** - 硬件成本+电费计算
- **回本周期预测** - 基于Revenue Analysis模块

### 📈 参数敏感性分析
- **发行模型参数** - λ值、a值影响分析
- **矿机规模参数** - 不同规模的投资建议
- **市场情景分析** - 多种市场条件下的表现
- **网络份额分配** - 参与者收益分配分析

## 技术栈

- **前端**: HTML5, CSS3, JavaScript, Chart.js
- **后端**: Python 3.x
- **数据处理**: Pandas, NumPy
- **可视化**: Matplotlib, Seaborn, Chart.js
- **科学计算**: SciPy
- **测试框架**: unittest

## 分析结果

项目生成多种格式的分析结果：
- **CSV文件** - 结构化数据输出
- **PNG图表** - 高质量可视化图表
- **HTML报告** - 交互式分析报告
- **PDF文档** - 白皮书验证结果

## 贡献指南

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目基于MIT许可证开源 - 查看 [LICENSE](LICENSE) 文件了解详情

## 联系方式

- GitHub: [@DDnagasawa](https://github.com/DDnagasawa)
- 项目链接: [AXON-Tokenomics-Modeling](https://github.com/DDnagasawa/AXON-Tokenomics-Modeling)

---

*基于AXON Tokenomics白皮书开发的数学建模工具*