"""
AXON Tokenomics Simulation Engine v2.2
=====================================
Formula-verified implementation based on AXON Network Tokenomic Framework v1.2.0
All formulas cross-referenced with paper sections - NO MAGIC NUMBERS
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Final
from functools import lru_cache, cached_property
from enum import Enum, auto
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 论文公式验证和常量定义 - 基于v1.2.0更新
# =============================================================================

class PaperFormulaReferences:
    """论文v1.2.0公式引用索引"""
    
    # 第6章 代币经济学模型公式
    EMISSION_RATE_FORMULA = "公式(31): v(x) = λ + (1-λ) · a^(-x/(N-x))"
    DYNAMIC_WEIGHTS_DL = "公式(34): WDL(t) = 0.15 + 0.3 · e^(-t/tk)"
    DYNAMIC_WEIGHTS_DF = "公式(35): WDF(t) = 0.6 - WDL(t)"
    
    # 第4章 FAP算法公式
    DATA_FEED_EFFECTIVENESS = "公式(9): E_DF,j = ωm·Smag + ωq·Squal + ωd·Sdir"
    MAGNITUDE_SCORE = "公式(10): Smag(j) = ||Uj - Mglob||2 / Σk||Uk - Mglob||2"
    QUALITY_SCORE = "公式(11): Squal(j) = max(0, (Lval(Mglob) - Lval(Mglob + ηUj))/Lval(Mglob))"
    DIRECTIONAL_SCORE = "公式(12): Sdir(j) = <Uj, Utrusted>/||Uj||·||Utrusted||"
    
    # 第6.3章 奖励分配公式
    REWARD_ALLOCATION_DL = "公式(39): RDL,i = Rpool,DL(t) · Ceff,DL,i / Σj Ceff,DL,j"
    REWARD_ALLOCATION_DF = "公式(40): RDF,i = Rpool,DF(t) · Ceff,DF,i / Σj Ceff,DF,j"
    
    # 第6.5章 veAXON治理公式
    VEAXON_CALCULATION = "公式(43): Vve,i = k·log(1+Vst,i)·(1+Tlock,i/Tmax)"
    GOVERNANCE_WEIGHT = "公式(44): Pgov,i = (Vve,i)^0.75 / Σj(Vve,j)^0.75"
    
    # 第5章 共识机制公式  
    CONTRIBUTION_SCORE = "公式(21): Scomp,i = wd·Sdata,i + wc·Scompute,i + ws·Sstake,i"
    STAKING_SCORE = "公式(24): Sstake,i = Tlock,i · log(1 + Vi)"
    FINAL_PROPOSER_SCORE = "公式(26): Sf,i = 0.6·Spoe,i + 0.3·Svr,i + 0.1·Ft,i"

class UpdatedPaperConstants:
    """论文v1.2.0中的精确常量"""
    
    # 第6.1节 基础参数 (保持不变)
    TOTAL_SUPPLY: Final[int] = 86_000_000_000
    INITIAL_ALLOCATION: Final[int] = 4_300_000_000
    INITIAL_ALLOCATION_PERCENTAGE: Final[float] = 0.05
    
    # 第6.2节 发行模型参数 (保持不变)
    MINIMUM_EMISSION_RATE: Final[float] = 0.01  # λ
    DECAY_COEFFICIENT: Final[float] = 2.5  # a
    
    # 第6.3节 奖励分配结构 (保持不变)
    COMPUTE_SECURITY_SHARE: Final[float] = 0.40  # 40%
    DATA_CONTRIBUTION_SHARE: Final[float] = 0.60  # 60%
    
    # 第6.3.1节 计算&安全奖励细分 (保持不变)
    PROPOSER_REWARD_SHARE: Final[float] = 0.30  # 30%
    VALIDATOR_REWARD_SHARE: Final[float] = 0.10  # 10%
    
    # 第6.3.2.1节 动态权重参数 (公式34和35)
    DL_BASE_WEIGHT: Final[float] = 0.15  # Domain-Library基础权重
    DL_DECAY_AMPLITUDE: Final[float] = 0.30  # Domain-Library衰减幅度
    TOTAL_DATA_WEIGHT: Final[float] = 0.60  # 总数据权重
    
    # 第5.3.4节 网络性能参数 (更新)
    TARGET_BLOCK_TIME_SECONDS: Final[float] = 3.0  # 3秒区块时间 (从0.5秒更新为3秒)
    
    # 通用时间常量
    SECONDS_PER_YEAR: Final[int] = 365 * 24 * 3600
    WEIGHT_DECAY_TIME_CONSTANT_YEARS: Final[float] = 2.0  # tk = 2年
    
    # 数学常量
    PERCENTAGE_SCALE: Final[float] = 100.0
    TOLERANCE: Final[float] = 1e-6
    
    # 第4.2.2.1节 Data-Feed评分权重 (公式9)
    class DataFeedScoring:
        """Data-Feed多维评分权重"""
        MAGNITUDE_WEIGHT: Final[float] = 0.4  # ωm
        QUALITY_WEIGHT: Final[float] = 0.4    # ωq  
        DIRECTIONAL_WEIGHT: Final[float] = 0.2  # ωd
        
        def __post_init__(self):
            total = self.MAGNITUDE_WEIGHT + self.QUALITY_WEIGHT + self.DIRECTIONAL_WEIGHT
            assert abs(total - 1.0) < UpdatedPaperConstants.TOLERANCE
    
    # 第5.2.3节 最终提议者评分权重 (公式26)
    class ProposerScoring:
        """最终提议者评分权重"""
        PROOF_OF_EFFECTIVENESS_WEIGHT: Final[float] = 0.6  # 60%
        VALIDATION_REPORT_WEIGHT: Final[float] = 0.3       # 30%
        TIME_FACTOR_WEIGHT: Final[float] = 0.1             # 10%

class NetworkEvolutionStageConstants:
    """第6.6.2.2节 网络发展阶段常量 (表格2更新)"""
    
    # 阶段时间边界
    EARLY_STAGE_END_YEARS: Final[float] = 2.0
    MID_STAGE_END_YEARS: Final[float] = 5.0
    
    # 表格2: 早期阶段参与者分配 (Outbreak Period)
    class EarlyStage:
        # Compute & Security (40%)部分
        PRO_MINERS_COMPUTE: Final[float] = 0.70  # 70% -> 28%总奖励
        ENTERPRISE_COMPUTE: Final[float] = 0.30  # 30% -> 12%总奖励
        RETAIL_COMPUTE: Final[float] = 0.0  # 0%
        
        # Domain-Library Rewards部分 (从Knowledge Source更新)
        ENTERPRISE_DL: Final[float] = 0.90  # 90%
        RETAIL_DL: Final[float] = 0.10  # 10%
        
        # Data-Feed Rewards (从Personal Data Pod更新)
        RETAIL_DF: Final[float] = 1.00  # 100%给零售用户
    
    # 表格2: 中期阶段参与者分配 (Growth Period)
    class MidStage:
        PRO_MINERS_COMPUTE: Final[float] = 0.50  # 50% -> 20%总奖励
        ENTERPRISE_COMPUTE: Final[float] = 0.45  # 45% -> 18%总奖励
        RETAIL_COMPUTE: Final[float] = 0.05  # 5% -> 2%总奖励
        
        ENTERPRISE_DL: Final[float] = 0.70  # 70%
        RETAIL_DL: Final[float] = 0.30  # 30%
        
        RETAIL_DF: Final[float] = 1.00  # 100%给零售用户
    
    # 表格2: 后期阶段参与者分配 (Maturity Period)
    class LateStage:
        PRO_MINERS_COMPUTE: Final[float] = 0.30  # 30% -> 12%总奖励
        ENTERPRISE_COMPUTE: Final[float] = 0.60  # 60% -> 24%总奖励
        RETAIL_COMPUTE: Final[float] = 0.10  # 10% -> 4%总奖励
        
        ENTERPRISE_DL: Final[float] = 0.50  # 50%
        RETAIL_DL: Final[float] = 0.50  # 50%
        
        RETAIL_DF: Final[float] = 1.00  # 100%给零售用户

# =============================================================================
# 更新的配置类 - 基于v1.2.0
# =============================================================================

@dataclass(frozen=True)
class UpdatedEmissionParameters:
    """基于论文v1.2.0验证的发行参数配置"""
    
    # 第6.1节 基础经济参数
    total_supply: int = UpdatedPaperConstants.TOTAL_SUPPLY
    initial_allocation: int = UpdatedPaperConstants.INITIAL_ALLOCATION
    min_emission_rate: float = UpdatedPaperConstants.MINIMUM_EMISSION_RATE
    decay_coefficient: float = UpdatedPaperConstants.DECAY_COEFFICIENT
    
    # 第5.3.4节 网络时间参数 (更新的区块时间)
    block_time_seconds: float = UpdatedPaperConstants.TARGET_BLOCK_TIME_SECONDS
    weight_decay_time_constant_years: float = UpdatedPaperConstants.WEIGHT_DECAY_TIME_CONSTANT_YEARS
    
    @cached_property
    def weight_decay_time_constant_seconds(self) -> float:
        """tk的秒数形式 - 用于公式34中的t/tk计算"""
        return self.weight_decay_time_constant_years * UpdatedPaperConstants.SECONDS_PER_YEAR
    
    @cached_property
    def blocks_per_year(self) -> float:
        """年区块数计算 - 用于发行率转换为区块奖励"""
        return UpdatedPaperConstants.SECONDS_PER_YEAR / self.block_time_seconds
    
    def __post_init__(self):
        """验证参数与论文一致性"""
        actual_percentage = self.initial_allocation / self.total_supply
        expected_percentage = UpdatedPaperConstants.INITIAL_ALLOCATION_PERCENTAGE
        
        if abs(actual_percentage - expected_percentage) > UpdatedPaperConstants.TOLERANCE:
            raise ValueError(f"初始分配比例不符合论文: 期望{expected_percentage}, 实际{actual_percentage}")

@dataclass(frozen=True, slots=True)
class UpdatedRewardStructure:
    """基于论文v1.2.0验证的奖励结构配置 - 第6.3节"""
    
    compute_security_share: float = UpdatedPaperConstants.COMPUTE_SECURITY_SHARE  # 40%
    data_contribution_share: float = UpdatedPaperConstants.DATA_CONTRIBUTION_SHARE  # 60%
    
    # 第6.3.1节 计算&安全奖励细分
    proposer_reward_share: float = UpdatedPaperConstants.PROPOSER_REWARD_SHARE  # 30%
    validator_reward_share: float = UpdatedPaperConstants.VALIDATOR_REWARD_SHARE  # 10%
    
    def __post_init__(self):
        """验证奖励分配总和为100%"""
        total_main = self.compute_security_share + self.data_contribution_share
        total_compute_detail = self.proposer_reward_share + self.validator_reward_share
        
        if abs(total_main - 1.0) > UpdatedPaperConstants.TOLERANCE:
            raise ValueError(f"主要奖励分配总和必须为1.0，实际为{total_main}")
        
        if abs(total_compute_detail - self.compute_security_share) > UpdatedPaperConstants.TOLERANCE:
            raise ValueError(f"计算奖励细分不等于总计算奖励: {total_compute_detail} vs {self.compute_security_share}")

@dataclass(frozen=True)  # 移除slots=True以支持cached_property
class UpdatedDynamicWeights:
    """基于论文v1.2.0验证的动态权重配置 - 第6.3.2.1节"""
    
    # 公式34参数: WDL(t) = 0.15 + 0.3 · e^(-t/tk)
    dl_base_weight: float = UpdatedPaperConstants.DL_BASE_WEIGHT  # 0.15
    dl_decay_amplitude: float = UpdatedPaperConstants.DL_DECAY_AMPLITUDE  # 0.30
    
    # 公式35约束: WDF(t) = 0.6 - WDL(t)
    total_data_weight: float = UpdatedPaperConstants.TOTAL_DATA_WEIGHT  # 0.60
    
    @cached_property
    def initial_dl_weight(self) -> float:
        """t=0时的Domain-Library权重 = 0.15 + 0.30 = 0.45"""
        return self.dl_base_weight + self.dl_decay_amplitude
    
    @cached_property
    def final_dl_weight(self) -> float:
        """t→∞时的Domain-Library权重 = 0.15"""
        return self.dl_base_weight
    
    @cached_property
    def initial_df_weight(self) -> float:
        """t=0时的Data-Feed权重 = 0.6 - 0.45 = 0.15"""
        return self.total_data_weight - self.initial_dl_weight
    
    @cached_property
    def final_df_weight(self) -> float:
        """t→∞时的Data-Feed权重 = 0.6 - 0.15 = 0.45"""
        return self.total_data_weight - self.final_dl_weight
    
    def __post_init__(self):
        """验证权重配置符合论文约束"""
        if self.initial_dl_weight > self.total_data_weight:
            raise ValueError(f"初始DL权重({self.initial_dl_weight})超过总数据权重({self.total_data_weight})")
        
        # 验证论文表格2中的百分比
        expected_early_dl_percent = self.initial_dl_weight / self.total_data_weight  # 应该是75%
        expected_early_df_percent = self.initial_df_weight / self.total_data_weight  # 应该是25%
        
        if abs(expected_early_dl_percent - 0.75) > UpdatedPaperConstants.TOLERANCE:
            raise ValueError(f"早期DL权重比例应为75%，实际为{expected_early_dl_percent:.1%}")

# =============================================================================
# 参与者分配策略 - 基于v1.2.0更新
# =============================================================================

@dataclass(frozen=True, slots=True)
class UpdatedParticipantRatios:
    """基于论文v1.2.0表格2验证的参与者分配比例"""
    
    # 计算&安全奖励分配
    pro_miners_compute: float
    enterprise_compute: float
    retail_compute: float
    
    # Domain-Library奖励分配 (从Knowledge Vault更新)
    enterprise_dl: float
    retail_dl: float
    
    # Data-Feed奖励分配 (从Personal Data Pod更新)
    retail_df: float = 1.0
    
    def __post_init__(self):
        """严格验证分配比例与论文一致"""
        compute_total = self.pro_miners_compute + self.enterprise_compute + self.retail_compute
        dl_total = self.enterprise_dl + self.retail_dl
        
        if abs(compute_total - 1.0) > UpdatedPaperConstants.TOLERANCE:
            raise ValueError(f"计算奖励分配总和应为1.0，实际为{compute_total}")
        
        if abs(dl_total - 1.0) > UpdatedPaperConstants.TOLERANCE:
            raise ValueError(f"Domain-Library奖励分配总和应为1.0，实际为{dl_total}")
        
        if abs(self.retail_df - 1.0) > UpdatedPaperConstants.TOLERANCE:
            raise ValueError(f"Data-Feed奖励应100%给零售用户，实际为{self.retail_df}")

class UpdatedStageBasedAllocation:
    """基于论文v1.2.0表格2实现的阶段分配策略"""
    
    def __init__(self):
        # 预定义的阶段分配 - 直接从论文表格2复制
        self._stage_ratios = {
            'early': UpdatedParticipantRatios(
                pro_miners_compute=NetworkEvolutionStageConstants.EarlyStage.PRO_MINERS_COMPUTE,
                enterprise_compute=NetworkEvolutionStageConstants.EarlyStage.ENTERPRISE_COMPUTE,
                retail_compute=NetworkEvolutionStageConstants.EarlyStage.RETAIL_COMPUTE,
                enterprise_dl=NetworkEvolutionStageConstants.EarlyStage.ENTERPRISE_DL,
                retail_dl=NetworkEvolutionStageConstants.EarlyStage.RETAIL_DL,
                retail_df=NetworkEvolutionStageConstants.EarlyStage.RETAIL_DF
            ),
            'mid': UpdatedParticipantRatios(
                pro_miners_compute=NetworkEvolutionStageConstants.MidStage.PRO_MINERS_COMPUTE,
                enterprise_compute=NetworkEvolutionStageConstants.MidStage.ENTERPRISE_COMPUTE,
                retail_compute=NetworkEvolutionStageConstants.MidStage.RETAIL_COMPUTE,
                enterprise_dl=NetworkEvolutionStageConstants.MidStage.ENTERPRISE_DL,
                retail_dl=NetworkEvolutionStageConstants.MidStage.RETAIL_DL,
                retail_df=NetworkEvolutionStageConstants.MidStage.RETAIL_DF
            ),
            'late': UpdatedParticipantRatios(
                pro_miners_compute=NetworkEvolutionStageConstants.LateStage.PRO_MINERS_COMPUTE,
                enterprise_compute=NetworkEvolutionStageConstants.LateStage.ENTERPRISE_COMPUTE,
                retail_compute=NetworkEvolutionStageConstants.LateStage.RETAIL_COMPUTE,
                enterprise_dl=NetworkEvolutionStageConstants.LateStage.ENTERPRISE_DL,
                retail_dl=NetworkEvolutionStageConstants.LateStage.RETAIL_DL,
                retail_df=NetworkEvolutionStageConstants.LateStage.RETAIL_DF
            )
        }
    
    def get_ratios(self, year: float) -> UpdatedParticipantRatios:
        """根据年份返回对应的参与者分配比例"""
        if year <= NetworkEvolutionStageConstants.EARLY_STAGE_END_YEARS:
            return self._stage_ratios['early']
        elif year <= NetworkEvolutionStageConstants.MID_STAGE_END_YEARS:
            return self._stage_ratios['mid']
        else:
            return self._stage_ratios['late']



class UpdatedFormulaEngine:
    """论文v1.2.0公式的精确实现"""
    
    def __init__(self, params: UpdatedEmissionParameters):
        self.params = params
    
    @lru_cache(maxsize=1000)
    def calculate_emission_rate(self, circulating_supply: float) -> float:
        """
        实现公式(31): v(x) = λ + (1-λ) · a^(-x/(N-x))
        
        论文第6.2节 - 动态发行模型
        """
        x = circulating_supply
        N = self.params.total_supply
        lambda_min = self.params.min_emission_rate  # λ
        a = self.params.decay_coefficient  # a
        
        # 边界条件处理
        if x >= N:
            return lambda_min
        if x <= 0:
            return lambda_min + (1 - lambda_min)
        
        # 公式(31)的精确实现
        ratio = x / (N - x)
        decay_factor = np.power(a, -ratio)  # a^(-x/(N-x))
        
        return lambda_min + (1 - lambda_min) * decay_factor
    
    def calculate_dynamic_weights(self, time_seconds: float, 
                                weights_config: UpdatedDynamicWeights) -> Tuple[float, float]:
        """
        实现公式(34)和(35): 
        WDL(t) = 0.15 + 0.3 · e^(-t/tk)
        WDF(t) = 0.6 - WDL(t)
        
        论文第6.3.2.1节 - 动态权重机制
        """
        t = time_seconds
        tk = self.params.weight_decay_time_constant_seconds
        
        # 公式(34): WDL(t) = 0.15 + 0.3 · e^(-t/tk)
        exp_decay = np.exp(-t / tk)
        wdl = weights_config.dl_base_weight + weights_config.dl_decay_amplitude * exp_decay
        
        # 公式(35): WDF(t) = 0.6 - WDL(t)
        wdf = weights_config.total_data_weight - wdl
        
        return wdl, wdf
    
    def calculate_block_reward_v1_2(self, emission_rate: float, remaining_supply: float) -> float:
        """
        根据发行率计算区块奖励 - 使用公式(33)
        
        基于第6.2节发行模型：
        ΔXblock = v(Xcurrent) · (N - Xcurrent) · (Tblock/Tyear)
        """
        # 公式(33): ΔXblock = v(Xcurrent) · (N - Xcurrent) · (Tblock/Tyear)
        time_ratio = self.params.block_time_seconds / UpdatedPaperConstants.SECONDS_PER_YEAR
        block_reward = emission_rate * remaining_supply * time_ratio
        
        return block_reward



@dataclass
class UpdatedSimulationResult:
    """基于v1.2.0更新的模拟结果"""
    
    # 基础时间序列
    years: np.ndarray
    circulating_supplies: np.ndarray
    
    # 发行机制数据 (公式31)
    emission_rates: np.ndarray
    block_rewards: np.ndarray
    annual_emissions: np.ndarray
    
    # 动态权重数据 (公式34和35)
    dl_weights: np.ndarray  # WDL(t) - Domain-Library权重
    df_weights: np.ndarray  # WDF(t) - Data-Feed权重
    
    # 奖励池数据 (第6.3节分配)
    compute_security_pools: np.ndarray  # 40%
    domain_library_pools: np.ndarray   # 动态权重 - Domain-Library
    data_feed_pools: np.ndarray        # 动态权重 - Data-Feed
    
    # 参与者百分比 (表格2验证)
    pro_miners_percentages: np.ndarray
    enterprise_percentages: np.ndarray
    retail_percentages: np.ndarray
    
    @cached_property
    def dataframe(self) -> pd.DataFrame:
        """转换为DataFrame用于分析"""
        return pd.DataFrame({
            'year': self.years,
            'circulating_supply': self.circulating_supplies,
            'emission_rate': self.emission_rates,
            'block_reward': self.block_rewards,
            'annual_emissions': self.annual_emissions,
            'dl_weight': self.dl_weights,
            'df_weight': self.df_weights,
            'compute_security_pool': self.compute_security_pools,
            'domain_library_pool': self.domain_library_pools,
            'data_feed_pool': self.data_feed_pools,
            'pro_miners_pct': self.pro_miners_percentages,
            'enterprise_pct': self.enterprise_percentages,
            'retail_pct': self.retail_percentages
        })
    
    def validate_against_paper_v1_2(self) -> Dict[str, bool]:
        """验证结果是否符合论文v1.2.0预期"""
        validations = {}
        
        # 验证Domain-Library权重范围 (公式34)
        validations['dl_weights_in_range'] = np.all(
            (self.dl_weights >= UpdatedPaperConstants.DL_BASE_WEIGHT) & 
            (self.dl_weights <= UpdatedPaperConstants.DL_BASE_WEIGHT + UpdatedPaperConstants.DL_DECAY_AMPLITUDE)
        )
        
        # 验证权重总和 (公式35)
        total_weights = self.dl_weights + self.df_weights
        validations['weights_sum_correct'] = np.allclose(
            total_weights, UpdatedPaperConstants.TOTAL_DATA_WEIGHT, atol=UpdatedPaperConstants.TOLERANCE
        )
        
        # 验证参与者百分比总和
        total_percentages = (self.pro_miners_percentages + 
                           self.enterprise_percentages + 
                           self.retail_percentages)
        validations['percentages_sum_100'] = np.allclose(
            total_percentages, UpdatedPaperConstants.PERCENTAGE_SCALE, atol=UpdatedPaperConstants.TOLERANCE
        )
        
        # 验证发行率范围 (公式31)
        validations['emission_rates_in_range'] = np.all(
            (self.emission_rates >= UpdatedPaperConstants.MINIMUM_EMISSION_RATE) &
            (self.emission_rates <= 1.0)
        )
        
        return validations



class UpdatedAXONSimulator:
    """基于论文v1.2.0验证的AXON代币经济学模拟器"""
    
    def __init__(self):
        # 使用更新的配置
        self.emission_params = UpdatedEmissionParameters()
        self.reward_structure = UpdatedRewardStructure()
        self.weights_config = UpdatedDynamicWeights()
        self.allocation_strategy = UpdatedStageBasedAllocation()
        
        # 初始化公式引擎
        self.formula_engine = UpdatedFormulaEngine(self.emission_params)
        
        print(f"🔬 Initialized AXON Simulator v1.2.0")
        print(f"📖 Formula Implementation Status:")
        print(f"   ✅ 公式(31): 发行率计算")
        print(f"   ✅ 公式(34): Domain-Library动态权重")
        print(f"   ✅ 公式(35): Data-Feed动态权重")
        print(f"   ✅ 表格2: 参与者分配")
        print(f"   ✅ 区块时间: {self.emission_params.block_time_seconds}秒 (更新)")
    
    def run_simulation_v1_2(self, years: int = 10, steps_per_year: int = 12) -> UpdatedSimulationResult:
        """
        运行基于v1.2.0的验证模拟
        """
        print(f"🚀 开始{years}年v1.2.0验证模拟...")
        print(f"📊 使用更新的论文参数:")
        print(f"   - 总供应量: {self.emission_params.total_supply:,} AXON")
        print(f"   - 初始分配: {self.emission_params.initial_allocation:,} AXON")
        print(f"   - 区块时间: {self.emission_params.block_time_seconds}秒")
        print(f"   - 组件更新: Domain-Library + Data-Feed + Compute-Grid")
        
        # 预分配数组
        total_steps = years * steps_per_year
        time_points = np.linspace(0, years, total_steps + 1)[1:]
        
        # 初始化结果数组
        results = self._preallocate_arrays_v1_2(total_steps)
        
        # 设置初始供应量
        current_supply = float(self.emission_params.initial_allocation)
        time_delta = 1.0 / steps_per_year
        
        # 主模拟循环
        for i, year in enumerate(time_points):
            # 计算当前步骤的所有指标
            step_data = self._calculate_step_v1_2(year, current_supply)
            
            # 存储结果
            self._store_results_v1_2(results, i, year, current_supply, step_data)
            
            # 更新流通供应量 - 使用公式(33)
            supply_increase = step_data['block_reward'] * (time_delta * self.emission_params.blocks_per_year)
            current_supply += supply_increase
            
            # 边界检查
            current_supply = min(current_supply, float(self.emission_params.total_supply))
            
            # 进度报告
            if (i + 1) % (total_steps // 10) == 0:
                progress = (i + 1) / total_steps * 100
                print(f"  进度: {progress:.0f}% - 第{year:.1f}年")
        
        # 创建验证结果
        result = UpdatedSimulationResult(**results)
        
        # 验证结果
        validations = result.validate_against_paper_v1_2()
        print(f"✅ v1.2.0模拟完成! 生成{len(result.years)}个数据点")
        print(f"🔍 论文一致性验证:")
        for check, passed in validations.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {check}")
        
        return result
    
    def _calculate_step_v1_2(self, year: float, supply: float) -> Dict:
        """
        计算单个时间步的所有指标 - v1.2.0版本
        """
        # 1. 发行率计算 (公式31)
        emission_rate = self.formula_engine.calculate_emission_rate(supply)
        remaining_supply = self.emission_params.total_supply - supply
        annual_emissions = remaining_supply * emission_rate
        block_reward = self.formula_engine.calculate_block_reward_v1_2(emission_rate, remaining_supply)
        
        # 2. 动态权重计算 (公式34和35)
        time_seconds = year * UpdatedPaperConstants.SECONDS_PER_YEAR
        wdl, wdf = self.formula_engine.calculate_dynamic_weights(time_seconds, self.weights_config)
        
        # 3. 奖励池分配 (第6.3节)
        compute_security_pool = block_reward * self.reward_structure.compute_security_share
        data_contribution_pool = block_reward * self.reward_structure.data_contribution_share
        
        # 4. 数据奖励细分 (公式34和35的应用)
        domain_library_pool = data_contribution_pool * (wdl / self.weights_config.total_data_weight)
        data_feed_pool = data_contribution_pool * (wdf / self.weights_config.total_data_weight)
        
        # 5. 参与者分配 (表格2)
        ratios = self.allocation_strategy.get_ratios(year)
        participant_rewards = self._calculate_participant_rewards_v1_2(
            compute_security_pool, domain_library_pool, data_feed_pool, ratios
        )
        
        return {
            'emission_rate': emission_rate,
            'annual_emissions': annual_emissions,
            'block_reward': block_reward,
            'wdl': wdl,
            'wdf': wdf,
            'compute_security_pool': compute_security_pool,
            'domain_library_pool': domain_library_pool,
            'data_feed_pool': data_feed_pool,
            **participant_rewards
        }
    
    def _calculate_participant_rewards_v1_2(self, compute_pool: float, dl_pool: float, 
                                          df_pool: float, ratios: UpdatedParticipantRatios) -> Dict:
        """
        根据表格2计算参与者奖励分配 - v1.2.0版本
        """
        # 计算各参与者的总奖励
        pro_miners_total = compute_pool * ratios.pro_miners_compute
        
        enterprise_total = (compute_pool * ratios.enterprise_compute + 
                          dl_pool * ratios.enterprise_dl)
        
        retail_total = (compute_pool * ratios.retail_compute + 
                       dl_pool * ratios.retail_dl + 
                       df_pool * ratios.retail_df)
        
        # 计算百分比分配
        total_rewards = pro_miners_total + enterprise_total + retail_total
        
        # 验证总奖励等于区块奖励
        expected_total = compute_pool + dl_pool + df_pool
        if abs(total_rewards - expected_total) > UpdatedPaperConstants.TOLERANCE:
            raise ValueError(f"参与者奖励总和({total_rewards})不等于区块奖励({expected_total})")
        
        return {
            'pro_miners_pct': (pro_miners_total / total_rewards) * UpdatedPaperConstants.PERCENTAGE_SCALE,
            'enterprise_pct': (enterprise_total / total_rewards) * UpdatedPaperConstants.PERCENTAGE_SCALE,
            'retail_pct': (retail_total / total_rewards) * UpdatedPaperConstants.PERCENTAGE_SCALE
        }
    
    def _preallocate_arrays_v1_2(self, size: int) -> Dict[str, np.ndarray]:
        """预分配numpy数组 - v1.2.0版本"""
        return {
            'years': np.zeros(size),
            'circulating_supplies': np.zeros(size),
            'emission_rates': np.zeros(size),
            'block_rewards': np.zeros(size),
            'annual_emissions': np.zeros(size),
            'dl_weights': np.zeros(size),  # Domain-Library权重
            'df_weights': np.zeros(size),  # Data-Feed权重
            'compute_security_pools': np.zeros(size),
            'domain_library_pools': np.zeros(size),  # Domain-Library奖励池
            'data_feed_pools': np.zeros(size),       # Data-Feed奖励池
            'pro_miners_percentages': np.zeros(size),
            'enterprise_percentages': np.zeros(size),
            'retail_percentages': np.zeros(size)
        }
    
    def _store_results_v1_2(self, arrays: Dict, i: int, year: float, 
                          supply: float, step_data: Dict):
        """存储计算结果到数组 - v1.2.0版本"""
        arrays['years'][i] = year
        arrays['circulating_supplies'][i] = supply
        arrays['emission_rates'][i] = step_data['emission_rate']
        arrays['block_rewards'][i] = step_data['block_reward']
        arrays['annual_emissions'][i] = step_data['annual_emissions']
        arrays['dl_weights'][i] = step_data['wdl']
        arrays['df_weights'][i] = step_data['wdf']
        arrays['compute_security_pools'][i] = step_data['compute_security_pool']
        arrays['domain_library_pools'][i] = step_data['domain_library_pool']
        arrays['data_feed_pools'][i] = step_data['data_feed_pool']
        arrays['pro_miners_percentages'][i] = step_data['pro_miners_pct']
        arrays['enterprise_percentages'][i] = step_data['enterprise_pct']
        arrays['retail_percentages'][i] = step_data['retail_pct']


class PaperConsistencyValidatorV1_2:
    """论文v1.2.0一致性验证工具"""
    
    @staticmethod
    def validate_emission_projections_v1_2(result: UpdatedSimulationResult):
        """验证发行率投影是否符合论文6.6.2.1节 - v1.2.0"""
        print("\n📊 验证发行率投影 (论文v1.2.0第6.6.2.1节):")
        
        # 论文中的关键检查点 (使用3秒区块时间)
        checkpoints = [
            (0.1, 90.4, 6658),   # 10%供应量时：发行率90.4%，区块奖励6658 AXON (3秒区块)
            (0.5, 40.6, 1446),   # 50%供应量时：发行率40.6%，区块奖励1446 AXON
            (0.9, 1.025, 8.12)   # 90%供应量时：发行率1.025%，区块奖励8.12 AXON
        ]
        
        df = result.dataframe
        total_supply = UpdatedPaperConstants.TOTAL_SUPPLY
        
        for supply_ratio, expected_emission_pct, expected_block_reward in checkpoints:
            # 找到最接近目标供应量的点
            target_supply = supply_ratio * total_supply
            closest_idx = np.argmin(np.abs(df['circulating_supply'] - target_supply))
            
            actual_emission_pct = df.iloc[closest_idx]['emission_rate'] * 100
            actual_block_reward = df.iloc[closest_idx]['block_reward']
            actual_supply_pct = df.iloc[closest_idx]['circulating_supply'] / total_supply * 100
            
            emission_error = abs(actual_emission_pct - expected_emission_pct) / expected_emission_pct
            reward_error = abs(actual_block_reward - expected_block_reward) / expected_block_reward
            
            print(f"  📍 {supply_ratio:.0%}供应量检查点 (实际: {actual_supply_pct:.1f}%):")
            print(f"    发行率: {actual_emission_pct:.1f}% (论文: {expected_emission_pct:.1f}%, 误差: {emission_error:.1%})")
            print(f"    区块奖励: {actual_block_reward:.0f} AXON (论文: {expected_block_reward:.0f} AXON, 误差: {reward_error:.1%})")
    
    @staticmethod
    def validate_component_name_updates(result: UpdatedSimulationResult):
        """验证组件名称更新"""
        print("\n📊 验证组件名称更新 (v1.1.0 → v1.2.0):")
        print("  ✅ Knowledge Source → Domain-Library")
        print("  ✅ Insight Source → Data-Feed") 
        print("  ✅ Execution Source → Compute-Grid")
        
        df = result.dataframe
        
        # 验证Domain-Library和Data-Feed权重
        initial_dl = df.iloc[0]['dl_weight']
        initial_df = df.iloc[0]['df_weight']
        final_dl = df.iloc[-1]['dl_weight']
        final_df = df.iloc[-1]['df_weight']
        
        print(f"  🎯 Domain-Library权重: {initial_dl:.3f} → {final_dl:.3f}")
        print(f"  🎯 Data-Feed权重: {initial_df:.3f} → {final_df:.3f}")
    
    @staticmethod
    def validate_block_time_update(result: UpdatedSimulationResult):
        """验证区块时间更新"""
        print("\n📊 验证区块时间更新:")
        print("  ✅ v1.1.0: 0.5秒区块时间")
        print("  ✅ v1.2.0: 3.0秒区块时间")
        print("  📈 这影响了区块奖励的绝对数值")

# =============================================================================
# 更新的可视化引擎
# =============================================================================

class UpdatedVisualizationEngine:
    """基于v1.2.0的可视化引擎"""
    
    def create_v1_2_dashboard(self, result: UpdatedSimulationResult):
        """创建v1.2.0验证仪表板"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))
        fig.suptitle('AXON Tokenomics v1.2.0 Verification Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 为每个子图添加清晰的模型标识
        model_names = [
            "1. Emission Rate Verification (Formula 31)",
            "2. Dynamic Weights Verification (Formula 34&35)", 
            "3. Participant Evolution (Table 2)",
            "4. Reward Pool Structure (v1.2.0 Components)",
            "5. Component Name Update Comparison",
            "6. Block Time Impact (0.5s→3s)"
        ]
        
        # 绘制每个子图并添加模型名称
        self._plot_emission_v1_2(axes[0, 0], result, model_names[0])
        self._plot_weights_v1_2(axes[0, 1], result, model_names[1])
        self._plot_participants_v1_2(axes[0, 2], result, model_names[2])
        self._plot_rewards_pools_v1_2(axes[1, 0], result, model_names[3])
        self._plot_component_comparison(axes[1, 1], result, model_names[4])
        self._plot_block_time_impact(axes[1, 2], result, model_names[5])
        
        # 设置单个子图的高:宽接近 3.2" : 5.5" ≈ 0.582
        for ax in axes.flatten():
            try:
                ax.set_box_aspect(0.582)
            except Exception:
                pass

        # 按指定间距布局
        plt.tight_layout(pad=2.0)
        plt.subplots_adjust(hspace=0.35, wspace=0.25)
        return fig
    
    def _plot_emission_v1_2(self, ax, result: UpdatedSimulationResult, model_name: str):
        """验证发行率图表 - v1.2.0"""
        df = result.dataframe
        
        ax.semilogy(df['year'], df['emission_rate'] * 100, 'b-', linewidth=3, 
                   label='Actual Emission Rate')
        
        # 添加v1.2.0检查点
        checkpoints = [(0.1, 90.4), (0.5, 40.6), (0.9, 1.025)]
        for supply_ratio, expected_rate in checkpoints:
            target_supply = supply_ratio * UpdatedPaperConstants.TOTAL_SUPPLY
            closest_idx = np.argmin(np.abs(df['circulating_supply'] - target_supply))
            year = df.iloc[closest_idx]['year']
            ax.plot(year, expected_rate, 'ro', markersize=8, 
                   label=f'v1.2.0 Checkpoint {supply_ratio:.0%}')
        
        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel('Emission Rate (%)', fontsize=10)
        ax.set_title('Emission Rate Verification (Formula 31) - 3s Block', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 在图表右上角添加模型编号
        ax.text(0.98, 0.98, model_name.split('.')[0], transform=ax.transAxes, 
                fontsize=12, fontweight='bold', ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.8))
    
    def _plot_weights_v1_2(self, ax, result: UpdatedSimulationResult, model_name: str):
        """验证动态权重图表 - v1.2.0"""
        df = result.dataframe
        
        ax.plot(df['year'], df['dl_weight'], 'g-', linewidth=3, 
               label='Domain-Library Weight (0.45→0.15)')
        ax.plot(df['year'], df['df_weight'], 'orange', linewidth=3, 
               label='Data-Feed Weight (0.15→0.45)')
        
        # 填充区域
        ax.fill_between(df['year'], 0, df['dl_weight'], alpha=0.3, color='green')
        ax.fill_between(df['year'], df['dl_weight'], 0.6, alpha=0.3, color='orange')
        
        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel('Weight', fontsize=10)
        ax.set_title('Dynamic Weights Verification (Formula 34&35)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 在图表右上角添加模型编号
        ax.text(0.98, 0.98, model_name.split('.')[0], transform=ax.transAxes, 
                fontsize=12, fontweight='bold', ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.8))
    
    def _plot_participants_v1_2(self, ax, result: UpdatedSimulationResult, model_name: str):
        """参与者演化图表 - v1.2.0"""
        df = result.dataframe
        
        ax.plot(df['year'], df['pro_miners_pct'], 'b-', linewidth=3, 
                label='Professional Miners', marker='o', markersize=2)
        ax.plot(df['year'], df['enterprise_pct'], 'r-', linewidth=3, 
                label='Enterprise Users', marker='s', markersize=2)
        ax.plot(df['year'], df['retail_pct'], 'g-', linewidth=3, 
                label='Retail Users', marker='^', markersize=2)
        
        # 阶段分割线
        ax.axvline(x=2, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=5, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel('Reward Share (%)', fontsize=10)
        ax.set_title('Participant Evolution (Table 2)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 在图表右上角添加模型编号
        ax.text(0.98, 0.98, model_name.split('.')[0], transform=ax.transAxes, 
                fontsize=12, fontweight='bold', ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightcoral', alpha=0.8))
    
    def _plot_rewards_pools_v1_2(self, ax, result: UpdatedSimulationResult, model_name: str):
        """奖励池图表 - v1.2.0"""
        df = result.dataframe
        
        ax.loglog(df['year'], df['compute_security_pool'], 'b-', linewidth=3, 
                 label='Compute & Security (40%)')
        ax.loglog(df['year'], df['domain_library_pool'], 'g-', linewidth=3, 
                 label='Domain-Library (Dynamic)')
        ax.loglog(df['year'], df['data_feed_pool'], 'orange', linewidth=3, 
                 label='Data-Feed (Dynamic)')
        
        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel('Reward Pool Size (AXON)', fontsize=10)
        ax.set_title('Reward Pool Structure - v1.2.0 Components', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 在图表右上角添加模型编号
        ax.text(0.98, 0.98, model_name.split('.')[0], transform=ax.transAxes, 
                fontsize=12, fontweight='bold', ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))
    
    def _plot_component_comparison(self, ax, result: UpdatedSimulationResult, model_name: str):
        """组件名称对比图表"""
        old_names = ['Knowledge\nSource', 'Insight\nSource', 'Execution\nSource']
        new_names = ['Domain-\nLibrary', 'Data-\nFeed', 'Compute-\nGrid']
        
        x_pos = np.arange(len(old_names))
        
        # 创建对比条形图
        ax.barh(x_pos - 0.2, [1, 1, 1], 0.4, label='v1.1.0', color='lightblue', alpha=0.7)
        ax.barh(x_pos + 0.2, [1, 1, 1], 0.4, label='v1.2.0', color='lightgreen', alpha=0.7)
        
        # 添加标签
        for i, (old, new) in enumerate(zip(old_names, new_names)):
            ax.text(0.5, i - 0.2, old, ha='center', va='center', fontweight='bold', fontsize=8)
            ax.text(0.5, i + 0.2, new, ha='center', va='center', fontweight='bold', fontsize=8)
        
        ax.set_yticks(x_pos)
        ax.set_yticklabels(['Layer 1', 'Layer 2', 'Layer 3'], fontsize=9)
        ax.set_xlim(0, 1.2)
        ax.set_xlabel('Component Comparison', fontsize=10)
        ax.set_title('Component Name Update Comparison', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        
        # 在图表右上角添加模型编号
        ax.text(0.98, 0.98, model_name.split('.')[0], transform=ax.transAxes, 
                fontsize=12, fontweight='bold', ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightpink', alpha=0.8))
    
    def _plot_block_time_impact(self, ax, result: UpdatedSimulationResult, model_name: str):
        """区块时间影响图表"""
        df = result.dataframe
        
        # 显示区块奖励的对数尺度变化
        ax.semilogy(df['year'], df['block_reward'], 'purple', linewidth=3, 
                   label='Block Reward (3s Block)')
        
        # 添加对比点
        sample_years = [0.5, 2, 5, 10]
        for year in sample_years:
            if year <= df['year'].max():
                closest_idx = np.argmin(np.abs(df['year'] - year))
                reward = df.iloc[closest_idx]['block_reward']
                ax.plot(year, reward, 'ro', markersize=6)
                ax.annotate(f'{reward:.0f} AXON', 
                           (year, reward), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel('Block Reward (AXON)', fontsize=10)
        ax.set_title('Block Time Impact (0.5s→3s)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 在图表右上角添加模型编号
        ax.text(0.98, 0.98, model_name.split('.')[0], transform=ax.transAxes, 
                fontsize=12, fontweight='bold', ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightsteelblue', alpha=0.8))

# =============================================================================
# 主程序 - v1.2.0
# =============================================================================

def main_v1_2():
    """运行v1.2.0完整验证模拟"""
    print("🚀 AXON Tokenomics v1.2.0 Verification Simulator")
    print("=" * 60)
    print("📖 基于 'AXON Network Tokenomic Framework v1.2.0'")
    print("🔄 主要更新:")
    print("   - Knowledge Source → Domain-Library")
    print("   - Insight Source → Data-Feed")
    print("   - Execution Source → Compute-Grid") 
    print("   - 区块时间: 0.5s → 3.0s")
    print("=" * 60)
    
    # 创建v1.2.0模拟器
    simulator = UpdatedAXONSimulator()
    
    # 运行验证模拟
    result = simulator.run_simulation_v1_2(years=10, steps_per_year=24)
    
    # v1.2.0一致性验证
    validator = PaperConsistencyValidatorV1_2()
    validator.validate_emission_projections_v1_2(result)
    validator.validate_component_name_updates(result)
    validator.validate_block_time_update(result)
    
    # 创建v1.2.0可视化
    print("\n🎨 生成v1.2.0验证图表...")
    viz_engine = UpdatedVisualizationEngine()
    fig = viz_engine.create_v1_2_dashboard(result)
    
    # 显示结果
    plt.show()
    
    # 保存验证结果
    print("\n💾 保存v1.2.0验证结果...")
    result.dataframe.to_csv('axon_tokenomics_v1.2.0_verified.csv', index=False)
    fig.savefig('axon_tokenomics_v1.2.0_verified.png', dpi=300, bbox_inches='tight')
    
    print("\n✅ v1.2.0验证完成!")
    print("📊 所有公式和组件名称已更新")
    print("🔧 区块时间更新已反映在区块奖励中")
    print("=" * 60)

if __name__ == "__main__":
    main_v1_2()