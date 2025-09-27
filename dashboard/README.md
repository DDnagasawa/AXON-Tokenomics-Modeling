# AXON Tokenomics Dashboard - Dash & Plotly Version

基于Dash和Plotly创建的AXON Tokenomics可视化Dashboard，使用横向排版、白色背景和蓝紫渐变色调。

## 🎨 设计特色

- **横向布局**: 充分利用屏幕宽度，6个图表并排显示
- **白色背景**: 简洁清爽的视觉体验
- **蓝紫渐变**: 现代化的配色方案
- **响应式设计**: 适配不同屏幕尺寸
- **实时数据**: 基于数学模型的动态数据

## 📊 图表内容

### 第一行 - 6个核心图表
1. **Emission Rate Verification** - 发行率验证 (对数尺度)
2. **Supply Evolution** - 供应量演化
3. **Participant Distribution** - 参与者分布
4. **Dynamic Weights** - 动态权重
5. **Token Price Trajectory** - 代币价格轨迹
6. **Block Reward Evolution** - 区块奖励演化 (对数尺度)

### 第二行 - 3个分析图表
1. **Reward Pools Structure** - 奖励池结构
2. **Network Performance** - 网络性能 (雷达图)
3. **Mining Economics** - 挖矿经济 (饼图)

### 顶部指标卡片
- **Total Supply**: 86B AXON
- **Circulating Supply**: 69B AXON
- **Emission Rate**: 3.4%
- **Block Reward**: 55 AXON

## 🚀 快速开始

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

## 🔧 技术栈

- **Dash**: Web应用框架
- **Plotly**: 交互式图表库
- **Pandas**: 数据处理
- **NumPy**: 数值计算

## 🎯 数据来源

Dashboard中的数据基于AXON Tokenomics v1.2.0白皮书中的数学模型：

- **公式(31)**: 发行率计算 v(x) = λ + (1-λ) · a^(-x/(N-x))
- **公式(34)**: Domain-Library动态权重 WDL(t) = 0.15 + 0.3 · e^(-t/tk)
- **公式(35)**: Data-Feed动态权重 WDF(t) = 0.6 - WDL(t)
- **表格2**: 参与者分配策略

## 🎨 自定义配置

### 颜色主题
```python
COLORS = {
    'primary': ['#667eea', '#764ba2', '#f093fb'],  # 主色调
    'secondary': ['#4facfe', '#00f2fe', '#43e97b'], # 辅助色
    'gradient': ['rgba(102, 126, 234, 0.2)', ...], # 渐变色
    'background': '#ffffff',                        # 背景色
    'text': '#334155',                              # 文字色
    'text_secondary': '#64748b'                     # 次要文字色
}
```

### 布局调整
- 修改 `height` 参数调整图表高度
- 修改 `margin` 参数调整图表间距
- 修改 `gap` 参数调整卡片间距

## 🔄 实时更新

Dashboard支持实时数据更新，可以通过以下方式扩展：

1. **添加回调函数**: 使用 `@app.callback` 装饰器
2. **连接数据源**: 集成实时API或数据库
3. **定时刷新**: 使用 `dcc.Interval` 组件

## 📱 响应式设计

Dashboard采用Flexbox布局，自动适配不同屏幕尺寸：

- **桌面端**: 6个图表并排显示
- **平板端**: 3x2网格布局
- **手机端**: 单列垂直布局

## 🎯 使用场景

- **客户演示**: 展示AXON Tokenomics模型
- **数据分析**: 实时监控网络指标
- **决策支持**: 基于数据的投资决策
- **教育展示**: 区块链经济学教学

## 🔧 故障排除

### 常见问题

1. **端口被占用**: 修改 `port=8050` 为其他端口
2. **依赖冲突**: 使用虚拟环境 `python -m venv venv`
3. **浏览器不兼容**: 使用Chrome或Firefox最新版本

### 调试模式
```python
app.run_server(debug=True, host='127.0.0.1', port=8050)
```

## 📈 扩展功能

可以添加的功能：

- **数据导出**: CSV/Excel下载
- **时间选择器**: 自定义时间范围
- **对比模式**: 多版本数据对比
- **预测模型**: 未来趋势预测
- **交互式筛选**: 动态数据过滤

## 📞 技术支持

如有问题，请检查：

1. Python版本 >= 3.8
2. 所有依赖已正确安装
3. 端口8050未被占用
4. 浏览器支持JavaScript

---

© 2024 AXON Tokenomics Dashboard | Powered by Mathematical Models v1.2.0
