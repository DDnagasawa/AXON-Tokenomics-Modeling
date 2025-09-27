"""
AXON Tokenomics Dashboard - Dash & Plotly Version v3.0
====================================================
创新版本：解决字体重叠问题，采用有趣的非线性布局，增强设计感
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 初始化Dash应用
app = dash.Dash(__name__, 
                title="AXON Tokenomics Dashboard",
                suppress_callback_exceptions=True)

# 创新的配色方案
COLORS = {
    'primary': ['#667eea', '#764ba2', '#f093fb'],
    'secondary': ['#4facfe', '#00f2fe', '#43e97b'],
    'accent': ['#ff6b6b', '#4ecdc4', '#45b7d1'],
    'gradient': ['rgba(102, 126, 234, 0.12)', 'rgba(118, 75, 162, 0.12)', 'rgba(240, 147, 251, 0.12)'],
    'background': '#ffffff',
    'text': '#1e293b',
    'text_secondary': '#64748b',
    'border': '#e2e8f0',
    'card_bg': '#fafbfc',
    'highlight': '#f1f5f9'
}

# 生成模拟数据
def generate_tokenomics_data():
    """生成AXON Tokenomics模拟数据"""
    years = np.linspace(0, 10, 121)  # 10年，每月一个数据点
    
    # 发行率数据 (公式31)
    emission_rates = []
    for year in years:
        supply_ratio = min(0.05 + year * 0.08, 0.8)  # 从5%到80%
        emission_rate = 0.01 + 0.99 * np.power(2.5, -supply_ratio / (1 - supply_ratio))
        emission_rates.append(emission_rate * 100)
    
    # 流通供应量
    circulating_supplies = 4.3 + (69 - 4.3) * (1 - np.exp(-years / 3))
    
    # 区块奖励 (对数衰减)
    block_rewards = 7409 * np.exp(-years * 0.3)
    
    # 动态权重
    dl_weights = 0.15 + 0.3 * np.exp(-years / 2)
    df_weights = 0.6 - dl_weights
    
    # 参与者分配
    pro_miners_pct = 28 + 12 * np.exp(-years / 2)
    enterprise_pct = 42 + 18 * (1 - np.exp(-years / 3))
    retail_pct = 30 + 6 * np.exp(-years / 4)
    
    # 奖励池
    compute_pool = block_rewards * 0.4
    dl_pool = block_rewards * 0.6 * dl_weights
    df_pool = block_rewards * 0.6 * df_weights
    
    # 代币价格 (模拟)
    base_price = 1.5
    price_volatility = 0.3
    prices = base_price + price_volatility * np.sin(years * 0.5) + 0.1 * years
    
    return pd.DataFrame({
        'year': years,
        'emission_rate': emission_rates,
        'circulating_supply': circulating_supplies,
        'block_reward': block_rewards,
        'dl_weight': dl_weights,
        'df_weight': df_weights,
        'pro_miners_pct': pro_miners_pct,
        'enterprise_pct': enterprise_pct,
        'retail_pct': retail_pct,
        'compute_pool': compute_pool,
        'dl_pool': dl_pool,
        'df_pool': df_pool,
        'token_price': prices
    })

# 生成数据
df = generate_tokenomics_data()

# 创新的图表创建函数 - 解决字体重叠问题
def create_emission_chart():
    """创建发行率图表"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['year'],
        y=df['emission_rate'],
        mode='lines',
        name='Emission Rate',
        line=dict(color=COLORS['primary'][0], width=3),
        fill='tonexty',
        fillcolor=COLORS['gradient'][0]
    ))
    
    fig.update_layout(
        title=dict(
            text="Emission Rate",
            font=dict(size=16, color=COLORS['text'], weight='bold'),
            x=0.5,
            y=0.98
        ),
        xaxis=dict(
            title="",
            showgrid=True,
            gridcolor=COLORS['border'],
            zeroline=False,
            showticklabels=True,
            tickfont=dict(size=11, color=COLORS['text_secondary']),
            tickmode='auto',
            nticks=6
        ),
        yaxis=dict(
            title="",
            type="log",
            showgrid=True,
            gridcolor=COLORS['border'],
            zeroline=False,
            showticklabels=True,
            tickfont=dict(size=11, color=COLORS['text_secondary']),
            tickmode='auto',
            nticks=5
        ),
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        margin=dict(l=50, r=30, t=60, b=50),
        height=300,
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig

def create_supply_chart():
    """创建供应量图表"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['year'],
        y=df['circulating_supply'],
        mode='lines',
        name='Circulating Supply',
        line=dict(color=COLORS['primary'][1], width=3),
        fill='tonexty',
        fillcolor=COLORS['gradient'][1]
    ))
    
    fig.update_layout(
        title=dict(
            text="Supply Evolution",
            font=dict(size=16, color=COLORS['text'], weight='bold'),
            x=0.5,
            y=0.98
        ),
        xaxis=dict(
            title="",
            showgrid=True,
            gridcolor=COLORS['border'],
            zeroline=False,
            showticklabels=True,
            tickfont=dict(size=11, color=COLORS['text_secondary']),
            tickmode='auto',
            nticks=6
        ),
        yaxis=dict(
            title="",
            showgrid=True,
            gridcolor=COLORS['border'],
            zeroline=False,
            showticklabels=True,
            tickfont=dict(size=11, color=COLORS['text_secondary']),
            tickmode='auto',
            nticks=5
        ),
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        margin=dict(l=50, r=30, t=60, b=50),
        height=300,
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig

def create_participants_chart():
    """创建参与者分布图表 - 折线图分割三个区域"""
    fig = go.Figure()
    
    # 添加三个区域的填充
    # 区域1: Professional Miners (底部区域)
    fig.add_trace(go.Scatter(
        x=df['year'],
        y=df['pro_miners_pct'],
        mode='lines',
        name='Professional Miners',
        line=dict(color=COLORS['primary'][0], width=3),
        fill='tonexty',
        fillcolor=COLORS['gradient'][0],
        stackgroup='one'
    ))
    
    # 区域2: Enterprise Users (中间区域)
    fig.add_trace(go.Scatter(
        x=df['year'],
        y=df['enterprise_pct'],
        mode='lines',
        name='Enterprise Users',
        line=dict(color=COLORS['primary'][1], width=3),
        fill='tonexty',
        fillcolor=COLORS['gradient'][1],
        stackgroup='one'
    ))
    
    # 区域3: Retail Users (顶部区域)
    fig.add_trace(go.Scatter(
        x=df['year'],
        y=df['retail_pct'],
        mode='lines',
        name='Retail Users',
        line=dict(color=COLORS['primary'][2], width=3),
        fill='tonexty',
        fillcolor=COLORS['gradient'][2],
        stackgroup='one'
    ))
    
    # 添加分割线
    # 第一条分割线：Professional Miners 和 Enterprise Users 之间
    fig.add_trace(go.Scatter(
        x=df['year'],
        y=df['pro_miners_pct'],
        mode='lines',
        name='',
        line=dict(color=COLORS['text'], width=2, dash='dash'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # 第二条分割线：Enterprise Users 和 Retail Users 之间
    fig.add_trace(go.Scatter(
        x=df['year'],
        y=[df['pro_miners_pct'][i] + df['enterprise_pct'][i] for i in range(len(df))],
        mode='lines',
        name='',
        line=dict(color=COLORS['text'], width=2, dash='dash'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=dict(
            text="Participant Distribution Over Time",
            font=dict(size=16, color=COLORS['text'], weight='bold'),
            x=0.5,
            y=0.98
        ),
        xaxis=dict(
            title="",
            showgrid=True,
            gridcolor=COLORS['border'],
            zeroline=False,
            showticklabels=True,
            tickfont=dict(size=11, color=COLORS['text_secondary']),
            tickmode='auto',
            nticks=6
        ),
        yaxis=dict(
            title="Market Share (%)",
            showgrid=True,
            gridcolor=COLORS['border'],
            zeroline=False,
            showticklabels=True,
            tickfont=dict(size=11, color=COLORS['text_secondary']),
            tickmode='auto',
            nticks=5,
            range=[0, 100]
        ),
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        margin=dict(l=50, r=30, t=60, b=50),
        height=300,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor=COLORS['border'],
            borderwidth=1
        ),
        hovermode='x unified'
    )
    
    return fig

def create_weights_chart():
    """创建动态权重图表"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['year'],
        y=df['dl_weight'],
        mode='lines',
        name='Domain-Library',
        line=dict(color=COLORS['primary'][0], width=3),
        fill='tonexty',
        fillcolor=COLORS['gradient'][0]
    ))
    
    fig.add_trace(go.Scatter(
        x=df['year'],
        y=df['df_weight'],
        mode='lines',
        name='Data-Feed',
        line=dict(color=COLORS['primary'][1], width=3),
        fill='tonexty',
        fillcolor=COLORS['gradient'][1]
    ))
    
    fig.update_layout(
        title=dict(
            text="Dynamic Weights",
            font=dict(size=16, color=COLORS['text'], weight='bold'),
            x=0.5,
            y=0.98
        ),
        xaxis=dict(
            title="",
            showgrid=True,
            gridcolor=COLORS['border'],
            zeroline=False,
            showticklabels=True,
            tickfont=dict(size=11, color=COLORS['text_secondary']),
            tickmode='auto',
            nticks=6
        ),
        yaxis=dict(
            title="",
            showgrid=True,
            gridcolor=COLORS['border'],
            zeroline=False,
            showticklabels=True,
            tickfont=dict(size=11, color=COLORS['text_secondary']),
            tickmode='auto',
            nticks=5,
            range=[0, 0.7]
        ),
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        margin=dict(l=50, r=30, t=60, b=50),
        height=300,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor=COLORS['border'],
            borderwidth=1
        ),
        hovermode='x unified'
    )
    
    return fig

def create_price_chart():
    """创建代币价格图表"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['year'],
        y=df['token_price'],
        mode='lines',
        name='Token Price',
        line=dict(color=COLORS['accent'][0], width=3),
        fill='tonexty',
        fillcolor=COLORS['gradient'][2]
    ))
    
    fig.update_layout(
        title=dict(
            text="Token Price",
            font=dict(size=16, color=COLORS['text'], weight='bold'),
            x=0.5,
            y=0.98
        ),
        xaxis=dict(
            title="",
            showgrid=True,
            gridcolor=COLORS['border'],
            zeroline=False,
            showticklabels=True,
            tickfont=dict(size=11, color=COLORS['text_secondary']),
            tickmode='auto',
            nticks=6
        ),
        yaxis=dict(
            title="",
            showgrid=True,
            gridcolor=COLORS['border'],
            zeroline=False,
            showticklabels=True,
            tickfont=dict(size=11, color=COLORS['text_secondary']),
            tickmode='auto',
            nticks=5
        ),
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        margin=dict(l=50, r=30, t=60, b=50),
        height=300,
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig

def create_block_reward_chart():
    """创建区块奖励图表"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['year'],
        y=df['block_reward'],
        mode='lines',
        name='Block Reward',
        line=dict(color=COLORS['accent'][1], width=3),
        fill='tonexty',
        fillcolor=COLORS['gradient'][2]
    ))
    
    fig.update_layout(
        title=dict(
            text="Block Reward",
            font=dict(size=16, color=COLORS['text'], weight='bold'),
            x=0.5,
            y=0.98
        ),
        xaxis=dict(
            title="",
            showgrid=True,
            gridcolor=COLORS['border'],
            zeroline=False,
            showticklabels=True,
            tickfont=dict(size=11, color=COLORS['text_secondary']),
            tickmode='auto',
            nticks=6
        ),
        yaxis=dict(
            title="",
            type="log",
            showgrid=True,
            gridcolor=COLORS['border'],
            zeroline=False,
            showticklabels=True,
            tickfont=dict(size=11, color=COLORS['text_secondary']),
            tickmode='auto',
            nticks=5
        ),
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        margin=dict(l=50, r=30, t=60, b=50),
        height=300,
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig

def create_reward_pools_chart():
    """创建奖励池图表"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['year'],
        y=df['compute_pool'],
        mode='lines',
        name='Compute & Security',
        line=dict(color=COLORS['primary'][0], width=2.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['year'],
        y=df['dl_pool'],
        mode='lines',
        name='Domain-Library',
        line=dict(color=COLORS['primary'][1], width=2.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['year'],
        y=df['df_pool'],
        mode='lines',
        name='Data-Feed',
        line=dict(color=COLORS['primary'][2], width=2.5)
    ))
    
    fig.update_layout(
        title=dict(
            text="Reward Pools",
            font=dict(size=18, color=COLORS['text'], weight='bold'),
            x=0.5,
            y=0.98
        ),
        xaxis=dict(
            title="",
            showgrid=True,
            gridcolor=COLORS['border'],
            zeroline=False,
            showticklabels=True,
            tickfont=dict(size=12, color=COLORS['text_secondary']),
            tickmode='auto',
            nticks=6
        ),
        yaxis=dict(
            title="",
            type="log",
            showgrid=True,
            gridcolor=COLORS['border'],
            zeroline=False,
            showticklabels=True,
            tickfont=dict(size=12, color=COLORS['text_secondary']),
            tickmode='auto',
            nticks=5
        ),
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        margin=dict(l=60, r=40, t=70, b=60),
        height=350,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor=COLORS['border'],
            borderwidth=1
        ),
        hovermode='x unified'
    )
    
    return fig

def create_network_metrics_chart():
    """创建网络指标雷达图"""
    categories = ['Efficiency', 'Security', 'Scalability', 'Decentralization', 'Sustainability']
    values = [85, 92, 78, 88, 95]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Network Performance',
        line_color=COLORS['primary'][0],
        fillcolor=COLORS['gradient'][0]
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor=COLORS['border'],
                tickfont=dict(size=11, color=COLORS['text_secondary'])
            ),
            angularaxis=dict(
                gridcolor=COLORS['border'],
                tickfont=dict(size=11, color=COLORS['text_secondary'])
            )
        ),
        title=dict(
            text="Network Performance",
            font=dict(size=18, color=COLORS['text'], weight='bold'),
            x=0.5,
            y=0.98
        ),
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        margin=dict(l=60, r=40, t=70, b=60),
        height=350,
        showlegend=False
    )
    
    return fig

def create_mining_economics_chart():
    """创建挖矿经济饼图"""
    labels = ['Hardware Cost', 'Electricity', 'Maintenance', 'Profit']
    values = [35, 25, 15, 25]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=COLORS['primary'],
        textinfo='label+percent',
        textfont=dict(color=COLORS['text'], size=12),
        textposition='outside'
    )])
    
    fig.update_layout(
        title=dict(
            text="Mining Economics",
            font=dict(size=18, color=COLORS['text'], weight='bold'),
            x=0.5,
            y=0.98
        ),
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        margin=dict(l=60, r=40, t=70, b=60),
        height=350,
        showlegend=False
    )
    
    return fig

# 创新的应用布局 - 非线性有趣组合
app.layout = html.Div([
    # 标题栏 - 创新设计
    html.Div([
        html.Div([
            html.H1("AXON Tokenomics Dashboard", 
                    style={
                        'textAlign': 'center',
                        'fontSize': '36px',
                        'fontWeight': '900',
                        'marginBottom': '12px',
                        'background': f'linear-gradient(135deg, {COLORS["primary"][0]}, {COLORS["primary"][1]}, {COLORS["primary"][2]})',
                        'backgroundClip': 'text',
                        'WebkitBackgroundClip': 'text',
                        'WebkitTextFillColor': 'transparent',
                        'letterSpacing': '2px',
                        'textShadow': '0 2px 4px rgba(0,0,0,0.1)'
                    }),
            html.P("Real-time blockchain network data visualization and analysis",
                   style={
                       'textAlign': 'center',
                       'color': COLORS['text_secondary'],
                       'fontSize': '18px',
                       'fontWeight': '400',
                       'marginBottom': '0',
                       'letterSpacing': '1px'
                   })
        ], style={
            'padding': '40px 0',
            'borderBottom': f'2px solid {COLORS["border"]}',
            'marginBottom': '40px',
            'background': f'linear-gradient(180deg, {COLORS["highlight"]}, {COLORS["background"]})'
        })
    ]),
    
    # 实时指标卡片 - 创新布局
    html.Div([
        html.Div([
            html.Div([
                html.H3("86B", style={
                    'margin': '0 0 10px 0', 
                    'fontSize': '32px', 
                    'fontWeight': '800', 
                    'color': COLORS['primary'][0],
                    'lineHeight': '1'
                }),
                html.P("Total Supply", style={
                    'margin': '0 0 15px 0', 
                    'color': COLORS['text_secondary'],
                    'fontSize': '15px',
                    'fontWeight': '600'
                }),
                html.Span("+0.0%", style={
                    'color': '#059669', 
                    'fontWeight': '700',
                    'fontSize': '13px',
                    'padding': '6px 12px',
                    'backgroundColor': '#d1fae5',
                    'borderRadius': '8px'
                })
            ], style={
                'textAlign': 'center',
                'padding': '30px 20px',
                'backgroundColor': COLORS['card_bg'],
                'borderRadius': '16px',
                'border': f'2px solid {COLORS["border"]}',
                'boxShadow': '0 4px 12px rgba(0, 0, 0, 0.1)',
                'transition': 'all 0.3s ease',
                'transform': 'translateY(0)'
            })
        ], style={'flex': '1', 'margin': '0 10px'}),
        
        html.Div([
            html.Div([
                html.H3("69B", style={
                    'margin': '0 0 10px 0', 
                    'fontSize': '32px', 
                    'fontWeight': '800', 
                    'color': COLORS['primary'][1],
                    'lineHeight': '1'
                }),
                html.P("Circulating Supply", style={
                    'margin': '0 0 15px 0', 
                    'color': COLORS['text_secondary'],
                    'fontSize': '15px',
                    'fontWeight': '600'
                }),
                html.Span("+3.4%", style={
                    'color': '#059669', 
                    'fontWeight': '700',
                    'fontSize': '13px',
                    'padding': '6px 12px',
                    'backgroundColor': '#d1fae5',
                    'borderRadius': '8px'
                })
            ], style={
                'textAlign': 'center',
                'padding': '30px 20px',
                'backgroundColor': COLORS['card_bg'],
                'borderRadius': '16px',
                'border': f'2px solid {COLORS["border"]}',
                'boxShadow': '0 4px 12px rgba(0, 0, 0, 0.1)',
                'transition': 'all 0.3s ease',
                'transform': 'translateY(0)'
            })
        ], style={'flex': '1', 'margin': '0 10px'}),
        
        html.Div([
            html.Div([
                html.H3("3.4%", style={
                    'margin': '0 0 10px 0', 
                    'fontSize': '32px', 
                    'fontWeight': '800', 
                    'color': COLORS['primary'][2],
                    'lineHeight': '1'
                }),
                html.P("Emission Rate", style={
                    'margin': '0 0 15px 0', 
                    'color': COLORS['text_secondary'],
                    'fontSize': '15px',
                    'fontWeight': '600'
                }),
                html.Span("-0.8%", style={
                    'color': '#dc2626', 
                    'fontWeight': '700',
                    'fontSize': '13px',
                    'padding': '6px 12px',
                    'backgroundColor': '#fee2e2',
                    'borderRadius': '8px'
                })
            ], style={
                'textAlign': 'center',
                'padding': '30px 20px',
                'backgroundColor': COLORS['card_bg'],
                'borderRadius': '16px',
                'border': f'2px solid {COLORS["border"]}',
                'boxShadow': '0 4px 12px rgba(0, 0, 0, 0.1)',
                'transition': 'all 0.3s ease',
                'transform': 'translateY(0)'
            })
        ], style={'flex': '1', 'margin': '0 10px'}),
        
        html.Div([
            html.Div([
                html.H3("55", style={
                    'margin': '0 0 10px 0', 
                    'fontSize': '32px', 
                    'fontWeight': '800', 
                    'color': COLORS['accent'][0],
                    'lineHeight': '1'
                }),
                html.P("Block Reward", style={
                    'margin': '0 0 15px 0', 
                    'color': COLORS['text_secondary'],
                    'fontSize': '15px',
                    'fontWeight': '600'
                }),
                html.Span("-2.1%", style={
                    'color': '#dc2626', 
                    'fontWeight': '700',
                    'fontSize': '13px',
                    'padding': '6px 12px',
                    'backgroundColor': '#fee2e2',
                    'borderRadius': '8px'
                })
            ], style={
                'textAlign': 'center',
                'padding': '30px 20px',
                'backgroundColor': COLORS['card_bg'],
                'borderRadius': '16px',
                'border': f'2px solid {COLORS["border"]}',
                'boxShadow': '0 4px 12px rgba(0, 0, 0, 0.1)',
                'transition': 'all 0.3s ease',
                'transform': 'translateY(0)'
            })
        ], style={'flex': '1', 'margin': '0 10px'})
    ], style={
        'display': 'flex',
        'justifyContent': 'space-between',
        'marginBottom': '50px',
        'gap': '20px'
    }),
    
    # 创新的图表布局 - 非线性组合
    # 第一行：2个大图表 + 2个小图表
    html.Div([
        # 左侧大图表
        html.Div([
            dcc.Graph(
                id='emission-chart',
                figure=create_emission_chart(),
                config={'displayModeBar': False}
            )
        ], style={
            'flex': '2', 
            'margin': '0 10px',
            'backgroundColor': COLORS['card_bg'],
            'borderRadius': '16px',
            'border': f'2px solid {COLORS["border"]}',
            'padding': '20px',
            'boxShadow': '0 4px 12px rgba(0, 0, 0, 0.1)'
        }),
        
        # 右侧两个小图表垂直排列
        html.Div([
            html.Div([
                dcc.Graph(
                    id='price-chart',
                    figure=create_price_chart(),
                    config={'displayModeBar': False}
                )
            ], style={
                'marginBottom': '20px',
                'backgroundColor': COLORS['card_bg'],
                'borderRadius': '12px',
                'border': f'1px solid {COLORS["border"]}',
                'padding': '15px',
                'boxShadow': '0 2px 8px rgba(0, 0, 0, 0.08)'
            }),
            
            html.Div([
                dcc.Graph(
                    id='block-reward-chart',
                    figure=create_block_reward_chart(),
                    config={'displayModeBar': False}
                )
            ], style={
                'backgroundColor': COLORS['card_bg'],
                'borderRadius': '12px',
                'border': f'1px solid {COLORS["border"]}',
                'padding': '15px',
                'boxShadow': '0 2px 8px rgba(0, 0, 0, 0.08)'
            })
        ], style={'flex': '1', 'margin': '0 10px'})
    ], style={
        'display': 'flex',
        'justifyContent': 'space-between',
        'marginBottom': '40px',
        'gap': '20px'
    }),
    
    # 第二行：3个中等图表
    html.Div([
        html.Div([
            dcc.Graph(
                id='supply-chart',
                figure=create_supply_chart(),
                config={'displayModeBar': False}
            )
        ], style={
            'flex': '1', 
            'margin': '0 8px',
            'backgroundColor': COLORS['card_bg'],
            'borderRadius': '16px',
            'border': f'2px solid {COLORS["border"]}',
            'padding': '20px',
            'boxShadow': '0 4px 12px rgba(0, 0, 0, 0.1)'
        }),
        
        html.Div([
            dcc.Graph(
                id='participants-chart',
                figure=create_participants_chart(),
                config={'displayModeBar': False}
            )
        ], style={
            'flex': '1', 
            'margin': '0 8px',
            'backgroundColor': COLORS['card_bg'],
            'borderRadius': '16px',
            'border': f'2px solid {COLORS["border"]}',
            'padding': '20px',
            'boxShadow': '0 4px 12px rgba(0, 0, 0, 0.1)'
        }),
        
        html.Div([
            dcc.Graph(
                id='weights-chart',
                figure=create_weights_chart(),
                config={'displayModeBar': False}
            )
        ], style={
            'flex': '1', 
            'margin': '0 8px',
            'backgroundColor': COLORS['card_bg'],
            'borderRadius': '16px',
            'border': f'2px solid {COLORS["border"]}',
            'padding': '20px',
            'boxShadow': '0 4px 12px rgba(0, 0, 0, 0.1)'
        })
    ], style={
        'display': 'flex',
        'justifyContent': 'space-between',
        'marginBottom': '40px',
        'gap': '16px'
    }),
    
    # 第三行：3个大图表
    html.Div([
        html.Div([
            dcc.Graph(
                id='reward-pools-chart',
                figure=create_reward_pools_chart(),
                config={'displayModeBar': False}
            )
        ], style={
            'flex': '1', 
            'margin': '0 12px',
            'backgroundColor': COLORS['card_bg'],
            'borderRadius': '16px',
            'border': f'2px solid {COLORS["border"]}',
            'padding': '25px',
            'boxShadow': '0 4px 12px rgba(0, 0, 0, 0.1)'
        }),
        
        html.Div([
            dcc.Graph(
                id='network-metrics-chart',
                figure=create_network_metrics_chart(),
                config={'displayModeBar': False}
            )
        ], style={
            'flex': '1', 
            'margin': '0 12px',
            'backgroundColor': COLORS['card_bg'],
            'borderRadius': '16px',
            'border': f'2px solid {COLORS["border"]}',
            'padding': '25px',
            'boxShadow': '0 4px 12px rgba(0, 0, 0, 0.1)'
        }),
        
        html.Div([
            dcc.Graph(
                id='mining-economics-chart',
                figure=create_mining_economics_chart(),
                config={'displayModeBar': False}
            )
        ], style={
            'flex': '1', 
            'margin': '0 12px',
            'backgroundColor': COLORS['card_bg'],
            'borderRadius': '16px',
            'border': f'2px solid {COLORS["border"]}',
            'padding': '25px',
            'boxShadow': '0 4px 12px rgba(0, 0, 0, 0.1)'
        })
    ], style={
        'display': 'flex',
        'justifyContent': 'space-between',
        'marginBottom': '50px',
        'gap': '24px'
    }),
    
    # 页脚 - 创新设计
    html.Div([
        html.P("© 2024 AXON Tokenomics Dashboard | Powered by Mathematical Models v1.2.0",
               style={
                   'textAlign': 'center',
                   'color': COLORS['text_secondary'],
                   'fontSize': '14px',
                   'fontWeight': '500',
                   'margin': '0',
                   'padding': '30px 0',
                   'borderTop': f'2px solid {COLORS["border"]}',
                   'background': f'linear-gradient(180deg, {COLORS["background"]}, {COLORS["highlight"]})'
               })
    ])
    
], style={
    'backgroundColor': COLORS['background'],
    'minHeight': '100vh',
    'padding': '0 40px',
    'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    'lineHeight': '1.6'
})

# 回调函数用于实时更新
@app.callback(
    Output('emission-chart', 'figure'),
    Input('emission-chart', 'id')
)
def update_emission_chart(_):
    return create_emission_chart()

if __name__ == '__main__':
    print("🚀 Starting AXON Tokenomics Dashboard v3.0...")
    print("📊 Dashboard will be available at: http://127.0.0.1:8050")
    print("🎨 Theme: Innovative layout with creative combinations")
    print("📱 Layout: Non-linear interesting arrangement")
    print("✨ Features: Fixed text overlap, creative design")
    
    app.run(
        debug=True,
        host='127.0.0.1',
        port=8050
    )
