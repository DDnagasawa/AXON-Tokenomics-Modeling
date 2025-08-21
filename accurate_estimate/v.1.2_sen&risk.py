"""
AXON Tokenomics Comprehensive Visualization Framework
==================================================
包含白皮书核心框架图 + 扩展分析可视化
Based on AXON Network Tokenomic Framework v1.2.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Circle
from matplotlib.patches import Polygon, Arrow
import networkx as nx
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 综合可视化引擎
# =============================================================================

class ComprehensiveVisualizationEngine:
    """综合可视化引擎：白皮书框架图 + 扩展分析"""
    
    def __init__(self):
        self._setup_academic_style()
        self.colors = self._define_color_palette()
    
    def _setup_academic_style(self):
        """设置学术风格"""
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.family': ['DejaVu Sans', 'Arial', 'sans-serif'],  # 避免字体问题
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            'axes.linewidth': 1.2,
            'grid.alpha': 0.3,
            'lines.linewidth': 2.0
        })
    
    def _define_color_palette(self):
        """定义学术色彩方案"""
        return {
            'primary': '#2E8B57',      # 海绿色
            'secondary': '#4169E1',    # 皇家蓝
            'accent': '#DC143C',       # 深红色
            'warning': '#FF8C00',      # 橙色
            'success': '#228B22',      # 森林绿
            'info': '#4682B4',         # 钢蓝色
            'light_gray': '#F5F5F5',   # 浅灰
            'dark_gray': '#696969'     # 深灰
        }

# =============================================================================
# 1. 白皮书核心框架可视化
# =============================================================================

class WhitepaperFrameworkVisualizer:
    """白皮书核心框架可视化器"""
    
    def __init__(self, viz_engine):
        self.viz = viz_engine
        self.colors = viz_engine.colors
    
    def create_tokenomics_architecture_diagram(self) -> plt.Figure:
        """
        创建代币经济学架构图
        对应白皮书第3章核心概念定义
        """
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # 标题
        fig.suptitle('AXON Network Tokenomics Architecture\nBased on Whitepaper Chapter 3 Core Concepts', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # 1. 三层架构核心组件
        self._draw_three_layer_architecture(ax)
        
        # 2. FAP算法核心
        self._draw_fap_core(ax)
        
        # 3. 代币流动
        self._draw_token_flows(ax)
        
        # 4. 治理机制
        self._draw_governance_mechanism(ax)
        
        return fig
    
    def _draw_three_layer_architecture(self, ax):
        """绘制三层架构"""
        # Layer 1: Compute-Grid
        compute_box = FancyBboxPatch(
            (0.5, 7.5), 2.5, 1.5, boxstyle="round,pad=0.1",
            facecolor=self.colors['primary'], alpha=0.7, edgecolor='black'
        )
        ax.add_patch(compute_box)
        ax.text(1.75, 8.25, 'Compute-Grid\n(Layer 1)', ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')
        ax.text(1.75, 7.8, 'Formula(6): CGk = {Ik, Pk, Hk, Rk, Lk, Vk, Ek}', 
                ha='center', va='center', fontsize=8, color='white')
        
        # Layer 2: Domain-Library
        domain_box = FancyBboxPatch(
            (4, 7.5), 2.5, 1.5, boxstyle="round,pad=0.1",
            facecolor=self.colors['secondary'], alpha=0.7, edgecolor='black'
        )
        ax.add_patch(domain_box)
        ax.text(5.25, 8.25, 'Domain-Library\n(Layer 2)', ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')
        ax.text(5.25, 7.8, 'Formula(1): DLi = {Ii, Di, Mi, Ei, Ci, Ti, Si, Fi}', 
                ha='center', va='center', fontsize=8, color='white')
        
        # Layer 3: Data-Feed
        data_box = FancyBboxPatch(
            (7.5, 7.5), 2, 1.5, boxstyle="round,pad=0.1",
            facecolor=self.colors['accent'], alpha=0.7, edgecolor='black'
        )
        ax.add_patch(data_box)
        ax.text(8.5, 8.25, 'Data-Feed\n(Layer 3)', ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')
        ax.text(8.5, 7.8, 'Formula(5): DFj = {Ij, Dj,enc, Hj, Cj,auth, Sj,usage, Uj}', 
                ha='center', va='center', fontsize=8, color='white')
    
    def _draw_fap_core(self, ax):
        """绘制FAP算法核心"""
        # FAP核心圆形
        fap_circle = Circle((5, 5.5), 1.2, facecolor=self.colors['warning'], 
                           alpha=0.8, edgecolor='black', linewidth=2)
        ax.add_patch(fap_circle)
        ax.text(5, 5.8, 'FAP Algorithm', ha='center', va='center',
                fontsize=12, fontweight='bold')
        ax.text(5, 5.2, 'Formula(8): F = {L, G, A, R}', ha='center', va='center',
                fontsize=9)
        
        # FAP四个组件
        components = [
            ('L\n(Local Training)', (2.5, 5.5)),
            ('G\n(Global Aggregation)', (5, 3.8)),
            ('A\n(Attribution)', (7.5, 5.5)),
            ('R\n(Reward Distribution)', (5, 7.2))
        ]
        
        for comp_text, (x, y) in components:
            comp_box = FancyBboxPatch(
                (x-0.4, y-0.3), 0.8, 0.6, boxstyle="round,pad=0.05",
                facecolor='lightblue', alpha=0.6, edgecolor='navy'
            )
            ax.add_patch(comp_box)
            ax.text(x, y, comp_text, ha='center', va='center', fontsize=8, fontweight='bold')
            
            # 连接线到FAP核心
            ax.arrow(x, y, 5-x, 5.5-y, head_width=0.05, head_length=0.05, 
                    fc='gray', ec='gray', alpha=0.6, length_includes_head=True)
    
    def _draw_token_flows(self, ax):
        """绘制代币流动"""
        # $AXON代币池
        token_box = FancyBboxPatch(
            (1, 2.5), 3, 1, boxstyle="round,pad=0.1",
            facecolor=self.colors['success'], alpha=0.7, edgecolor='black'
        )
        ax.add_patch(token_box)
        ax.text(2.5, 3, '$AXON Token Pool\nFormula(31): v(x) = λ + (1-λ)·a^(-x/(N-x))', 
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        # veAXON治理
        veaxon_box = FancyBboxPatch(
            (6, 2.5), 3, 1, boxstyle="round,pad=0.1",
            facecolor=self.colors['info'], alpha=0.7, edgecolor='black'
        )
        ax.add_patch(veaxon_box)
        ax.text(7.5, 3, 'veAXON Governance\nFormula(43): Vve,i = k·log(1+Vst,i)·(1+Tlock,i/Tmax)', 
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        # 流动箭头
        ax.arrow(4, 3, 1.8, 0, head_width=0.1, head_length=0.1, 
                fc=self.colors['dark_gray'], ec=self.colors['dark_gray'])
        ax.text(5, 3.3, 'Staking', ha='center', va='center', fontsize=8)
    
    def _draw_governance_mechanism(self, ax):
        """绘制治理机制"""
        # DAO治理框
        dao_box = FancyBboxPatch(
            (3.5, 0.5), 3, 1, boxstyle="round,pad=0.1",
            facecolor=self.colors['dark_gray'], alpha=0.7, edgecolor='black'
        )
        ax.add_patch(dao_box)
        ax.text(5, 1, 'DAO Governance\nFormula(44): Pgov,i = (Vve,i)^0.75 / Σj(Vve,j)^0.75', 
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        # 治理连接线
        ax.arrow(7.5, 2.5, -2, -1.3, head_width=0.08, head_length=0.08, 
                fc='purple', ec='purple', linestyle='--')
        ax.text(6.5, 1.8, 'Governance\nControl', ha='center', va='center', 
                fontsize=8, color='purple')

    def create_dynamic_weights_evolution_diagram(self) -> plt.Figure:
        """
        创建动态权重演化图
        对应白皮书公式(34)和(35)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Dynamic Weight Evolution Framework\nBased on Whitepaper Formulas (34) and (35)', 
                    fontsize=14, fontweight='bold')
        
        # 左图：权重演化曲线
        time_years = np.linspace(0, 8, 100)
        tk = 2.0  # 基线值
        
        # 公式(34): WDL(t) = 0.15 + 0.3 · e^(-t/tk)
        wdl = 0.15 + 0.3 * np.exp(-time_years / tk)
        # 公式(35): WDF(t) = 0.6 - WDL(t)
        wdf = 0.6 - wdl
        
        ax1.fill_between(time_years, 0, wdl, alpha=0.6, color=self.colors['primary'], 
                        label='Domain-Library Weight')
        ax1.fill_between(time_years, wdl, 0.6, alpha=0.6, color=self.colors['secondary'], 
                        label='Data-Feed Weight')
        
        # 标记交叉点
        crossover_time = tk * np.log(2)
        ax1.axvline(x=crossover_time, color='red', linestyle='--', linewidth=2, 
                   label=f'Crossover Point: {crossover_time:.2f}y')
        ax1.plot(crossover_time, 0.3, 'ro', markersize=8, markeredgecolor='white', 
                markeredgewidth=2)
        
        ax1.set_xlabel('Time (Years)')
        ax1.set_ylabel('Weight')
        ax1.set_title('Panel A: Weight Evolution\nImplementation of Formulas (34)&(35)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 0.6)
        
        # 右图：不同tk值的影响
        tk_values = [1.0, 1.5, 2.0, 2.5, 3.0]
        colors_tk = plt.cm.viridis(np.linspace(0, 1, len(tk_values)))
        
        for i, tk_val in enumerate(tk_values):
            wdl_tk = 0.15 + 0.3 * np.exp(-time_years / tk_val)
            crossover_tk = tk_val * np.log(2)
            
            ax2.plot(time_years, wdl_tk, color=colors_tk[i], linewidth=2.5,
                    label=f'tk={tk_val}y (crossover: {crossover_tk:.2f}y)')
            
            # 标记交叉点
            if crossover_tk <= 8:
                ax2.plot(crossover_tk, 0.3, 'o', color=colors_tk[i], markersize=6,
                        markeredgecolor='white', markeredgewidth=1)
        
        ax2.axhline(y=0.3, color='black', linestyle='-', alpha=0.5, 
                   label='Crossover Level (0.3)')
        ax2.set_xlabel('Time (Years)')
        ax2.set_ylabel('Domain-Library Weight')
        ax2.set_title('Panel B: tk Parameter Sensitivity\nCrossover Time Sensitivity Analysis')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0.1, 0.5)
        
        plt.tight_layout()
        return fig

    def create_emission_mechanism_diagram(self) -> plt.Figure:
        """
        创建发行机制图
        对应白皮书公式(31)和6.6.2.1节
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Emission Rate Mechanism Analysis\nBased on Whitepaper Formula (31) and Section 6.6.2.1', 
                    fontsize=14, fontweight='bold')
        
        # 左图：发行率曲线
        supply_ratios = np.linspace(0.05, 0.95, 100)
        total_supply = 86_000_000_000
        lambda_min = 0.01
        a = 2.5
        
        emission_rates = []
        for ratio in supply_ratios:
            supply = ratio * total_supply
            x_over_n_minus_x = supply / (total_supply - supply)
            rate = lambda_min + (1 - lambda_min) * np.power(a, -x_over_n_minus_x)
            emission_rates.append(rate * 100)
        
        ax1.semilogy(supply_ratios * 100, emission_rates, color=self.colors['primary'], 
                    linewidth=3, label='Emission Rate (%)')
        
        # 标记白皮书检查点
        checkpoints = [
            (10, 90.4, '10% Supply\n90.4% Emission'),
            (50, 40.6, '50% Supply\n40.6% Emission'),
            (90, 1.025, '90% Supply\n1.025% Emission')
        ]
        
        for supply_pct, emission_pct, label in checkpoints:
            ax1.plot(supply_pct, emission_pct, 'ro', markersize=8, 
                    markeredgecolor='white', markeredgewidth=2)
            ax1.annotate(label, (supply_pct, emission_pct), 
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=8, bbox=dict(boxstyle="round,pad=0.3", 
                        facecolor='yellow', alpha=0.7))
        
        ax1.set_xlabel('Circulating Supply (%)')
        ax1.set_ylabel('Emission Rate (%)')
        ax1.set_title('Panel A: Emission Rate Decay\nFormula (31): v(x) = λ + (1-λ)·a^(-x/(N-x))')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 右图：区块奖励演化
        years = np.linspace(0, 10, 100)
        supply_progression = 0.05 + 0.90 * (years / 10)  # 简化的供应量增长
        
        block_rewards = []
        for supply_pct in supply_progression:
            supply = supply_pct * total_supply
            ratio = supply / (total_supply - supply)
            emission_rate = lambda_min + (1 - lambda_min) * np.power(a, -ratio)
            remaining_supply = total_supply - supply
            
            # 基于3秒区块时间
            block_reward = emission_rate * remaining_supply * (3 / (365 * 24 * 3600))
            block_rewards.append(block_reward)
        
        ax2.semilogy(years, block_rewards, color=self.colors['accent'], 
                    linewidth=3, label='Block Reward (AXON)')
        
        # 标记关键时点
        key_points = [
            (0.5, 6658, 'Early Stage\n~6,658 AXON'),
            (5, 1446, 'Mid Stage\n~1,446 AXON'),
            (9, 8.12, 'Late Stage\n~8.12 AXON')
        ]
        
        for year, reward, label in key_points:
            if year <= 10:
                closest_idx = np.argmin(np.abs(years - year))
                actual_reward = block_rewards[closest_idx]
                ax2.plot(year, actual_reward, 'bs', markersize=8, 
                        markeredgecolor='white', markeredgewidth=2)
                ax2.annotate(label, (year, actual_reward), 
                           xytext=(10, -10), textcoords='offset points',
                           fontsize=8, bbox=dict(boxstyle="round,pad=0.3", 
                           facecolor='lightblue', alpha=0.7))
        
        ax2.set_xlabel('Time (Years)')
        ax2.set_ylabel('Block Reward (AXON)')
        ax2.set_title('Panel B: Block Reward Evolution\nBased on 3-Second Block Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

# =============================================================================
# 2. 扩展分析可视化
# =============================================================================

class ExtendedAnalysisVisualizer:
    """扩展分析可视化器"""
    
    def __init__(self, viz_engine):
        self.viz = viz_engine
        self.colors = viz_engine.colors
    
    def create_risk_assessment_matrix(self) -> plt.Figure:
        """
        创建风险评估矩阵
        基于白皮书框架的扩展分析
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Risk Assessment Matrix (Extended Analysis)\nExtended Risk Analysis Based on Whitepaper Framework', 
                    fontsize=14, fontweight='bold')
        
        # 1. Death Spiral风险热力图
        self._plot_death_spiral_heatmap(ax1)
        
        # 2. 流动性风险演化
        self._plot_liquidity_evolution(ax2)
        
        # 3. 中心化风险动态
        self._plot_centralization_dynamics(ax3)
        
        # 4. 综合风险雷达图
        self._plot_comprehensive_risk_radar(ax4)
        
        plt.tight_layout()
        return fig
    
    def _plot_death_spiral_heatmap(self, ax):
        """Death Spiral风险热力图"""
        # 价格冲击 vs veAXON锁定期
        price_shocks = np.linspace(-0.1, -0.9, 10)
        lock_periods = np.linspace(0.5, 4.0, 10)
        
        risk_matrix = np.zeros((len(lock_periods), len(price_shocks)))
        
        for i, lock_period in enumerate(lock_periods):
            for j, shock in enumerate(price_shocks):
                # 基于白皮书6.5.1节的缓解机制
                unstaking_pressure = abs(shock) * 0.8  # 弹性系数
                time_buffer = 1 - np.exp(-lock_period * 0.6)  # 时间缓冲
                effective_risk = unstaking_pressure * (1 - time_buffer)
                risk_matrix[i, j] = min(effective_risk * 1.5, 1.0)
        
        im = ax.imshow(risk_matrix, cmap='Reds', aspect='auto',
                      extent=[abs(price_shocks[0])*100, abs(price_shocks[-1])*100,
                             lock_periods[0], lock_periods[-1]])
        
        ax.set_xlabel('Price Shock (%)')
        ax.set_ylabel('veAXON Lock Period (Years)')
        ax.set_title('Death Spiral Risk Matrix\nBased on Whitepaper Section 6.5.1 veAXON Mechanism')
        
        # 添加等高线
        contours = ax.contour(np.linspace(10, 90, 10), np.linspace(0.5, 4.0, 10), 
                             risk_matrix, levels=[0.3, 0.7], colors=['white'], 
                             linewidths=2, alpha=0.8)
        ax.clabel(contours, inline=True, fontsize=8)
        
        plt.colorbar(im, ax=ax, label='Risk Score')
    
    def _plot_liquidity_evolution(self, ax):
        """流动性风险演化"""
        years = np.linspace(0, 10, 100)
        
        # 基于白皮书公式(31)的发行率衰减
        supply_progression = 0.05 + 0.90 * (years / 10)
        emission_rates = []
        
        for supply_pct in supply_progression:
            ratio = supply_pct / (1 - supply_pct)
            rate = 0.01 + 0.99 * np.power(2.5, -ratio)
            emission_rates.append(rate)
        
        emission_rates = np.array(emission_rates)
        
        # 流动性风险建模
        base_liquidity_risk = 1 - emission_rates / emission_rates[0]
        
        # DAO国库缓解（基于白皮书6.4.3节）
        dao_buffer = np.minimum(years / 5, 0.5)  # 5年达到50%缓解
        mitigated_risk = base_liquidity_risk * (1 - dao_buffer)
        
        # 外部市场条件影响（扩展分析）
        market_volatility = 0.1 * np.sin(years * 2 * np.pi / 3) + 0.1  # 3年周期
        total_risk = mitigated_risk + market_volatility
        
        ax.fill_between(years, 0, base_liquidity_risk, alpha=0.3, 
                       color='red', label='Base Liquidity Risk')
        ax.fill_between(years, 0, mitigated_risk, alpha=0.6, 
                       color='blue', label='Post-DAO Mitigation')
        ax.plot(years, total_risk, 'purple', linewidth=2, linestyle='--',
               label='Including Market Conditions')
        
        ax.set_xlabel('Time (Years)')
        ax.set_ylabel('Liquidity Risk Score')
        ax.set_title('Liquidity Risk Evolution\nWhitepaper Section 6.4.3 + Market Conditions Extension')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    def _plot_centralization_dynamics(self, ax):
        """中心化风险动态"""
        years = np.linspace(0, 10, 100)
        
        # 基于白皮书Table 2的参与者演化
        enterprise_share = np.zeros(len(years))
        retail_share = np.zeros(len(years))
        
        for i, year in enumerate(years):
            if year <= 2:
                enterprise_share[i] = 0.525
                retail_share[i] = 0.195
            elif year <= 5:
                progress = (year - 2) / 3
                enterprise_share[i] = 0.525 - progress * (0.525 - 0.315)
                retail_share[i] = 0.195 + progress * (0.565 - 0.195)
            else:
                enterprise_share[i] = 0.315
                retail_share[i] = 0.565
        
        # 原始中心化风险
        governance_concentration = enterprise_share * 0.8
        base_centralization_risk = np.maximum(
            (governance_concentration - 0.33) / (1 - 0.33), 0
        )
        
        # 白皮书公式(44)的缓解效应
        concave_mitigation = 1 - np.power(governance_concentration, 0.75)
        mitigated_risk = base_centralization_risk * (1 - concave_mitigation * 0.8)
        
        # 网络效应增强（扩展分析）
        network_effect = 1 - retail_share  # 零售参与度越高，网络效应越强
        final_risk = mitigated_risk * network_effect
        
        ax.plot(years, base_centralization_risk, 'r-', linewidth=2, 
               label='Base Centralization Risk')
        ax.plot(years, mitigated_risk, 'b-', linewidth=2, 
               label='Post-Equation(44) Mitigation')
        ax.plot(years, final_risk, 'g--', linewidth=2, 
               label='Including Network Effects')
        
        # 双轴显示参与者份额
        ax2 = ax.twinx()
        ax2.plot(years, enterprise_share * 100, 'orange', alpha=0.6, 
                label='Enterprise Share (%)')
        ax2.plot(years, retail_share * 100, 'purple', alpha=0.6, 
                label='Retail Share (%)')
        
        ax.set_xlabel('Time (Years)')
        ax.set_ylabel('Centralization Risk', color='blue')
        ax2.set_ylabel('Participant Share (%)', color='purple')
        ax.set_title('Centralization Risk Dynamics\nTable 2 + Network Effects Analysis')
        
        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=8)
        
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        ax2.set_ylim(0, 80)
    
    def _plot_comprehensive_risk_radar(self, ax):
        """综合风险雷达图"""
        # 风险维度
        risk_categories = [
            'Death Spiral\nRisk',
            'Liquidity\nRisk', 
            'Centralization\nRisk',
            'Market\nVolatility',
            'Regulatory\nRisk',
            'Technical\nRisk'
        ]
        
        # 三个时期的风险评分
        early_stage_risks = [0.6, 0.8, 0.7, 0.9, 0.5, 0.4]    # 0-2年
        mid_stage_risks = [0.4, 0.5, 0.4, 0.6, 0.6, 0.3]      # 2-5年
        late_stage_risks = [0.2, 0.3, 0.2, 0.4, 0.7, 0.2]     # 5+年
        
        # 角度计算
        angles = np.linspace(0, 2 * np.pi, len(risk_categories), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        # 数据闭合
        early_stage_risks += early_stage_risks[:1]
        mid_stage_risks += mid_stage_risks[:1]
        late_stage_risks += late_stage_risks[:1]
        
        # 绘制雷达图
        ax.plot(angles, early_stage_risks, 'o-', linewidth=2, 
               label='Early Stage (0-2y)', color=self.colors['accent'])
        ax.fill(angles, early_stage_risks, alpha=0.25, color=self.colors['accent'])
        
        ax.plot(angles, mid_stage_risks, 's-', linewidth=2, 
               label='Mid Stage (2-5y)', color=self.colors['secondary'])
        ax.fill(angles, mid_stage_risks, alpha=0.25, color=self.colors['secondary'])
        
        ax.plot(angles, late_stage_risks, '^-', linewidth=2, 
               label='Late Stage (5+y)', color=self.colors['success'])
        ax.fill(angles, late_stage_risks, alpha=0.25, color=self.colors['success'])
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(risk_categories, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title('Comprehensive Risk Profile\nMulti-dimensional Risk Evolution Radar')
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax.grid(True)

    def create_monte_carlo_simulation_dashboard(self) -> plt.Figure:
        """
        创建蒙特卡罗模拟仪表板
        基于白皮书6.7节博弈论分析的扩展
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Monte Carlo Simulation Dashboard\nExtended Analysis Based on Whitepaper Section 6.7 Game Theory Framework', 
                    fontsize=14, fontweight='bold')
        
        # 设置随机种子以确保可重复性
        np.random.seed(42)
        
        # 1. 参数不确定性分析
        self._plot_parameter_uncertainty(ax1)
        
        # 2. 收益分布模拟
        self._plot_reward_distribution_simulation(ax2)
        
        # 3. 市场冲击响应
        self._plot_market_shock_response(ax3)
        
        # 4. 长期稳定性测试
        self._plot_long_term_stability(ax4)
        
        plt.tight_layout()
        return fig
    
    def _plot_parameter_uncertainty(self, ax):
        """参数不确定性分析"""
        # 基于白皮书参数的不确定性建模
        n_simulations = 1000
        
        # 参数分布（基于白皮书基线值±20%）
        decay_coeffs = np.random.normal(2.5, 0.3, n_simulations)  # a参数
        tk_values = np.random.normal(2.0, 0.4, n_simulations)     # tk参数
        
        # 边界限制
        decay_coeffs = np.clip(decay_coeffs, 1.0, 5.0)
        tk_values = np.clip(tk_values, 0.5, 4.0)
        
        # 计算关键指标
        crossover_times = tk_values * np.log(2)  # DL-DF交叉点
        early_emission_rates = 0.01 + 0.99 * np.power(decay_coeffs, -0.1/0.9)  # 10%供应量时
        
        # 创建散点图
        scatter = ax.scatter(crossover_times, early_emission_rates * 100, 
                           c=decay_coeffs, cmap='viridis', alpha=0.6, s=20)
        
        # 标记基线值
        baseline_crossover = 2.0 * np.log(2)
        baseline_emission = 0.01 + 0.99 * np.power(2.5, -0.1/0.9)
        ax.plot(baseline_crossover, baseline_emission * 100, 'r*', 
               markersize=15, markeredgecolor='white', markeredgewidth=2,
               label='Baseline (a=2.5, tk=2.0)')
        
        ax.set_xlabel('Crossover Time (Years)')
        ax.set_ylabel('Early Emission Rate (%)')
        ax.set_title('Parameter Uncertainty Analysis\n1000 Monte Carlo Simulations')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Decay Coefficient (a)')
    
    def _plot_reward_distribution_simulation(self, ax):
        """收益分布模拟"""
        # 基于白皮书6.3节奖励结构的模拟
        n_participants = 1000
        n_periods = 50
        
        # 三类参与者的收益模拟
        pro_miners_rewards = []
        enterprise_rewards = []
        retail_rewards = []
        
        for period in range(n_periods):
            # 基于Table 2的时间演化
            progress = period / n_periods
            
            if progress <= 0.4:  # 早期阶段
                pro_share, ent_share, ret_share = 0.28, 0.525, 0.195
            elif progress <= 0.8:  # 中期阶段
                pro_share, ent_share, ret_share = 0.20, 0.3375, 0.4625
            else:  # 后期阶段
                pro_share, ent_share, ret_share = 0.12, 0.315, 0.565
            
            # 模拟区块奖励衰减
            supply_ratio = 0.05 + 0.90 * progress
            emission_rate = 0.01 + 0.99 * np.power(2.5, -supply_ratio/(1-supply_ratio))
            block_reward = emission_rate * 1000  # 标准化
            
            # 分配奖励
            pro_miners_rewards.append(block_reward * pro_share)
            enterprise_rewards.append(block_reward * ent_share)
            retail_rewards.append(block_reward * ret_share)
        
        # 绘制累积收益
        periods = np.arange(n_periods)
        ax.plot(periods, np.cumsum(pro_miners_rewards), 'b-', linewidth=2,
               label='Professional Miners')
        ax.plot(periods, np.cumsum(enterprise_rewards), 'r-', linewidth=2,
               label='Enterprise Users')
        ax.plot(periods, np.cumsum(retail_rewards), 'g-', linewidth=2,
               label='Retail Participants')
        
        # 添加阶段分割线
        ax.axvline(x=20, color='gray', linestyle='--', alpha=0.5, label='Stage Transitions')
        ax.axvline(x=40, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Time Periods')
        ax.set_ylabel('Cumulative Rewards')
        ax.set_title('Reward Distribution Simulation\nBased on Table 2 Evolution Path')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_market_shock_response(self, ax):
        """市场冲击响应"""
        # 模拟不同程度的市场冲击
        shock_intensities = np.array([-0.1, -0.3, -0.5, -0.7, -0.9])
        recovery_times = []
        final_impacts = []
        
        for shock in shock_intensities:
            # 基于veAXON机制的缓解模型
            time_steps = np.linspace(0, 5, 100)  # 5年恢复期
            
            # 初始冲击
            initial_impact = abs(shock)
            
            # veAXON锁定缓解（基于白皮书6.5.1节）
            avg_lock_period = 2.5  # 平均锁定期
            time_buffer = 1 - np.exp(-avg_lock_period * 0.6)
            
            # 恢复曲线（指数恢复）
            recovery_curve = initial_impact * np.exp(-time_steps / 2) * (1 - time_buffer)
            
            # 计算恢复时间（冲击减少到5%）
            recovery_threshold = initial_impact * 0.05
            recovery_idx = np.where(recovery_curve <= recovery_threshold)[0]
            recovery_time = time_steps[recovery_idx[0]] if len(recovery_idx) > 0 else 5
            
            recovery_times.append(recovery_time)
            final_impacts.append(recovery_curve[-1])
            
            # 绘制恢复曲线
            ax.plot(time_steps, recovery_curve, linewidth=2, 
                   label=f'{abs(shock):.0%} Shock')
        
        ax.set_xlabel('Time (Years)')
        ax.set_ylabel('Residual Impact')
        ax.set_title('Market Shock Response\nveAXON Lock-up Mitigation Effect')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    def _plot_long_term_stability(self, ax):
        """长期稳定性测试"""
        # 10年期稳定性分析
        years = np.linspace(0, 10, 120)  # 月度数据
        
        # 基础经济指标
        supply_ratio = 0.05 + 0.90 * (years / 10)
        emission_rates = 0.01 + 0.99 * np.power(2.5, -supply_ratio/(1-supply_ratio))
        
        # 添加随机波动（市场噪声）
        np.random.seed(42)
        market_noise = np.random.normal(0, 0.02, len(years))  # 2%标准差
        noisy_emissions = emission_rates + market_noise
        noisy_emissions = np.clip(noisy_emissions, 0.01, 1.0)  # 边界限制
        
        # 计算波动性指标
        rolling_window = 12  # 12个月滚动窗口
        volatility = []
        for i in range(rolling_window, len(noisy_emissions)):
            window_data = noisy_emissions[i-rolling_window:i]
            vol = np.std(window_data) / np.mean(window_data)  # 变异系数
            volatility.append(vol)
        
        volatility = np.array(volatility)
        vol_years = years[rolling_window:]
        
        # 双轴图
        ax2 = ax.twinx()
        
        # 发行率趋势
        ax.plot(years, emission_rates * 100, 'b-', linewidth=2, 
               label='Baseline Emission Rate', alpha=0.8)
        ax.fill_between(years, (emission_rates - 2*market_noise) * 100,
                       (emission_rates + 2*market_noise) * 100,
                       alpha=0.3, color='blue', label='±2σ Confidence Band')
        
        # 波动性
        ax2.plot(vol_years, volatility * 100, 'r-', linewidth=2,
                label='Rolling Volatility (12m)')
        
        # 稳定性阈值
        ax2.axhline(y=5, color='orange', linestyle='--', alpha=0.7,
                   label='Stability Threshold (5%)')
        
        ax.set_xlabel('Time (Years)')
        ax.set_ylabel('Emission Rate (%)', color='blue')
        ax2.set_ylabel('Volatility (%)', color='red')
        ax.set_title('Long-term Stability Analysis\nEmission Mechanism Stability Test')
        
        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
        
        ax.grid(True, alpha=0.3)

# =============================================================================
# 3. 集成可视化生成器
# =============================================================================

class IntegratedVisualizationGenerator:
    """集成可视化生成器"""
    
    def __init__(self):
        self.viz_engine = ComprehensiveVisualizationEngine()
        self.whitepaper_viz = WhitepaperFrameworkVisualizer(self.viz_engine)
        self.extended_viz = ExtendedAnalysisVisualizer(self.viz_engine)
    
    def generate_complete_report_figures(self):
        """生成完整的报告图表包"""
        print("🎨 Generating Complete Visualization Report Package...")
        print("=" * 60)
        
        figures = {}
        
        # 1. Whitepaper Core Framework Charts
        print("📖 Generating Whitepaper Core Framework Charts...")
        figures['architecture'] = self.whitepaper_viz.create_tokenomics_architecture_diagram()
        figures['dynamic_weights'] = self.whitepaper_viz.create_dynamic_weights_evolution_diagram()
        figures['emission'] = self.whitepaper_viz.create_emission_mechanism_diagram()
        
        # 2. Extended Analysis Charts
        print("🔬 Generating Extended Analysis Charts...")
        figures['risk_matrix'] = self.extended_viz.create_risk_assessment_matrix()
        figures['monte_carlo'] = self.extended_viz.create_monte_carlo_simulation_dashboard()
        
        print("✅ Visualization Generation Complete!")
        return figures
    
    def save_all_figures(self, figures, output_dir="./"):
        """Save all charts"""
        print("💾 Saving Chart Files...")
        
        saved_files = []
        for name, fig in figures.items():
            # Save PNG format
            png_filename = f"{output_dir}axon_{name}_analysis.png"
            fig.savefig(png_filename, dpi=300, bbox_inches='tight', format='png')
            saved_files.append(png_filename)
            
            print(f"  ✅ Saved: {png_filename}")
        
        print(f"📁 Total {len(saved_files)} chart files saved")
        return saved_files
    
    def display_all_figures(self, figures):
        """Display all charts"""
        print("🖼️ Displaying All Charts...")
        plt.show()

# =============================================================================
# 主程序
# =============================================================================

def main():
    """Main Program: Generate Complete Visualization Analysis Report"""
    
    print("🚀 AXON Tokenomics Comprehensive Visualization Report")
    print("=" * 70)
    print("📊 Contents:")
    print("   📖 Whitepaper Core Framework Charts (Chapter 3 Architecture + Formulas 34&35 + Formula 31)")
    print("   🔬 Extended Risk Analysis Charts (Reasonable Extension Based on Whitepaper Framework)")
    print("   🎲 Monte Carlo Simulation Dashboard (Based on Section 6.7 Game Theory Analysis)")
    print("=" * 70)
    
    # 创建集成生成器
    generator = IntegratedVisualizationGenerator()
    
    # 生成所有图表
    figures = generator.generate_complete_report_figures()
    
    # Save charts
    saved_files = generator.save_all_figures(figures)
    
    # Display charts
    generator.display_all_figures(figures)
    
    print("\n📋 Visualization Report Summary:")
    print("=" * 40)
    print("📖 Whitepaper Core Framework Charts:")
    print("   ✅ Tokenomics Architecture Chart (Based on Chapter 3)")
    print("   ✅ Dynamic Weight Evolution Chart (Formulas 34&35)")
    print("   ✅ Emission Mechanism Analysis Chart (Formula 31)")
    
    print("\n🔬 Extended Analysis Charts:")
    print("   ✅ Risk Assessment Matrix (Multi-dimensional Risk Analysis)")
    print("   ✅ Monte Carlo Dashboard (Uncertainty Analysis)")
    
    print(f"\n💾 Saved {len(saved_files)} high-resolution chart files")
    print("📁 All charts ready for academic reports")
    print("=" * 70)
    
    return figures, saved_files

if __name__ == "__main__":
    figures, saved_files = main()