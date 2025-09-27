# AXON Tokenomics Dashboard - Dash & Plotly Version


### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 启动Dashboard
```bash
# 方法1: 直接运行
python axon_dashboard_dash.py

# 方法2: 使用启动脚本 (推荐)
python run_dashboard.py
```

### 3. 访问Dashboard
打开浏览器访问: http://127.0.0.1:8050

## 📁 文件结构

```
dashboard/
├── axon_dashboard_dash.py    # 主Dashboard文件
├── run_dashboard.py          # 启动脚本
├── requirements.txt          # 依赖文件
└── README.md                # 说明文档
```


## 🎯 数据来源

Dashboard中的数据基于AXON Tokenomics v1.2.0白皮书中的数学模型：

- **公式(31)**: 发行率计算 v(x) = λ + (1-λ) · a^(-x/(N-x))
- **公式(34)**: Domain-Library动态权重 WDL(t) = 0.15 + 0.3 · e^(-t/tk)
- **公式(35)**: Data-Feed动态权重 WDF(t) = 0.6 - WDL(t)
- **表格2**: 参与者分配策略


