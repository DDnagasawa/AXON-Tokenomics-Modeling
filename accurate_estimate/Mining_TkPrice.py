import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Set font for better display
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class MiningConfiguration:
    """Mining Configuration Parameters - According to Your Requirements"""
    # Network Configuration - Per Your Requirements
    initial_miners: int = 50_000      # Initial 50,000 mining rigs
    miners_at_6_months: int = 300_000 # 300,000 mining rigs after 6 months
    
    # Mining Hardware Configuration
    power_consumption_watts: float = 120  # 120 watts
    hours_per_day: float = 24
    days_per_month: float = 30
    
    # Electricity Cost Configuration (RMB to USD, exchange rate 7.2)
    electricity_cost_high: float = 1.2 / 7.2  # 1.2 RMB/kWh
    electricity_cost_low: float = 0.6 / 7.2   # 0.6 RMB/kWh
    
    # Economic Targets
    target_profit_6_months: float = 5_000  # 6-month target profit $5000
    miner_cost_usd: float = 5_000          # Mining rig cost $5000
    
    # Payback Requirements
    payback_scenarios: Dict[str, Dict] = None
    
    def __post_init__(self):
        if self.payback_scenarios is None:
            self.payback_scenarios = {
                "early_mining": {  # t < 3 months start mining
                    "start_time_months": 1,
                    "payback_months": 1,
                    "target_profit": self.miner_cost_usd,
                    "description": "Early Mining 1-Month Payback"
                },
                "mid_mining": {   # t < 6 months start mining  
                    "start_time_months": 4,
                    "payback_months": 3,
                    "target_profit": self.miner_cost_usd,
                    "description": "Mid-Stage Mining 3-Month Payback"
                },
                "late_mining": {  # t > 6 months start mining
                    "start_time_months": 7,
                    "payback_months": 6,
                    "target_profit": self.miner_cost_usd,
                    "description": "Late-Stage Mining 6-Month Payback"
                }
            }

class AXONMiningSimulator:
    """AXON Mining Economics Simulator - Strictly Following Whitepaper Formulas"""
    
    def __init__(self, config: MiningConfiguration):
        self.config = config
        
        # AXON Network Parameters (Strictly Following Whitepaper)
        self.total_supply = 86_000_000_000      # 86B total supply
        self.initial_allocation = 4_300_000_000  # 5% initial allocation
        self.min_emission_rate = 0.01           # λ = 0.01
        self.decay_coefficient = 2.5            # a = 2.5
        self.block_time_seconds = 3.0           # 3-second block time
        self.blocks_per_day = (24 * 3600) / self.block_time_seconds
        
        # Reward Distribution (Whitepaper Formulas)
        self.compute_security_share = 0.40  # 40% for compute and security
        self.data_contribution_share = 0.60 # 60% for data contribution
        
        # Time Constant (Whitepaper Formulas 34,35)
        self.tk = 2.0  # 2-year time constant
        
    def calculate_emission_rate(self, circulating_supply: float) -> float:
        """Calculate emission rate - Whitepaper Formula (31): v(x) = λ + (1-λ) * a^(-x/(N-x))"""
        x = circulating_supply
        N = self.total_supply
        lambda_min = self.min_emission_rate
        a = self.decay_coefficient
        
        if x >= N:
            return lambda_min
        if x <= 0:
            return lambda_min + (1 - lambda_min)
        
        ratio = x / (N - x)
        decay_factor = np.power(a, -ratio)
        return lambda_min + (1 - lambda_min) * decay_factor
    
    def calculate_block_reward(self, emission_rate: float, remaining_supply: float) -> float:
        """Calculate block reward"""
        time_ratio = self.block_time_seconds / (365 * 24 * 3600)
        return emission_rate * remaining_supply * time_ratio
    
    def calculate_dynamic_weights(self, time_years: float) -> tuple:
        """Calculate dynamic weights - Whitepaper Formulas (34,35)"""
        # WDL(t) = 0.15 + 0.3 * e^(-t/tk)
        w_dl = 0.15 + 0.3 * np.exp(-time_years / self.tk)
        # WDF(t) = 0.6 - WDL(t)  
        w_df = 0.6 - w_dl
        return w_dl, w_df
    
    def calculate_miners_count(self, month: float) -> int:
        """Calculate miner count - According to your requirements"""
        if month <= 6:
            # First 6 months: grow from 50,000 to 300,000
            growth_rate = (self.config.miners_at_6_months / self.config.initial_miners) ** (1/6)
            return int(self.config.initial_miners * (growth_rate ** month))
        else:
            # After 6 months: continue growing
            additional_months = month - 6
            monthly_growth = 1.1  # 10% monthly growth
            return int(self.config.miners_at_6_months * (monthly_growth ** additional_months))
    
    def simulate_mining_economics(self, months: int = 12) -> pd.DataFrame:
        """Simulate mining economics"""
        results = []
        current_supply = float(self.initial_allocation)
        
        for month in range(1, months + 1):
            # Time conversion
            time_years = month / 12.0
            
            # Calculate current network state
            emission_rate = self.calculate_emission_rate(current_supply)
            remaining_supply = self.total_supply - current_supply
            block_reward = self.calculate_block_reward(emission_rate, remaining_supply)
            
            # Calculate dynamic weights
            w_dl, w_df = self.calculate_dynamic_weights(time_years)
            
            # Calculate miner count
            miners_count = self.calculate_miners_count(month)
            
            # Calculate daily and monthly rewards
            daily_total_rewards = block_reward * self.blocks_per_day
            monthly_total_rewards = daily_total_rewards * self.config.days_per_month
            
            # Allocate reward pools
            monthly_compute_rewards = monthly_total_rewards * self.compute_security_share
            monthly_data_rewards = monthly_total_rewards * self.data_contribution_share
            
            # Calculate individual miner compute rewards (evenly distributed)
            avg_compute_reward_per_miner = monthly_compute_rewards / miners_count
            
            # Data reward allocation (geometric distribution: 80/20 rule)
            monthly_dl_rewards = monthly_data_rewards * w_dl / 0.6  # Domain-Library rewards
            monthly_df_rewards = monthly_data_rewards * w_df / 0.6  # Data-Feed rewards
            
            # 20% of miners get 80% of Domain-Library rewards
            top_20_percent = int(miners_count * 0.2)
            bottom_80_percent = miners_count - top_20_percent
            
            avg_dl_reward_top = (monthly_dl_rewards * 0.8) / top_20_percent if top_20_percent > 0 else 0
            avg_dl_reward_regular = (monthly_dl_rewards * 0.2) / bottom_80_percent if bottom_80_percent > 0 else 0
            
            # Data-Feed rewards also follow 80/20 distribution
            avg_df_reward_top = (monthly_df_rewards * 0.8) / top_20_percent if top_20_percent > 0 else 0
            avg_df_reward_regular = (monthly_df_rewards * 0.2) / bottom_80_percent if bottom_80_percent > 0 else 0
            
            # Calculate total rewards
            total_reward_top_miner = avg_compute_reward_per_miner + avg_dl_reward_top + avg_df_reward_top
            total_reward_regular_miner = avg_compute_reward_per_miner + avg_dl_reward_regular + avg_df_reward_regular
            
            # Calculate electricity costs
            monthly_power_kwh = (self.config.power_consumption_watts * self.config.hours_per_day * self.config.days_per_month) / 1000
            electricity_cost_high = monthly_power_kwh * self.config.electricity_cost_high
            electricity_cost_low = monthly_power_kwh * self.config.electricity_cost_low
            
            # Update circulating supply
            current_supply += monthly_total_rewards
            
            results.append({
                'month': month,
                'time_years': time_years,
                'emission_rate': emission_rate,
                'block_reward': block_reward,
                'miners_count': miners_count,
                'w_dl': w_dl,
                'w_df': w_df,
                'monthly_total_rewards': monthly_total_rewards,
                'monthly_compute_rewards': monthly_compute_rewards,
                'monthly_data_rewards': monthly_data_rewards,
                'monthly_dl_rewards': monthly_dl_rewards,
                'monthly_df_rewards': monthly_df_rewards,
                'avg_compute_reward_per_miner': avg_compute_reward_per_miner,
                'total_reward_top_miner': total_reward_top_miner,
                'total_reward_regular_miner': total_reward_regular_miner,
                'electricity_cost_high': electricity_cost_high,
                'electricity_cost_low': electricity_cost_low,
                'circulating_supply': current_supply
            })
        
        return pd.DataFrame(results)
    
    def calculate_required_token_price(self, mining_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate token price required to meet payback targets"""
        price_scenarios = []
        
        for scenario_name, scenario in self.config.payback_scenarios.items():
            start_month = scenario["start_time_months"]
            payback_months = scenario["payback_months"]
            target_profit = scenario["target_profit"]
            
            # Calculate cumulative token output and electricity costs
            end_month = min(start_month + payback_months, len(mining_data))
            period_data = mining_data[
                (mining_data['month'] >= start_month) & 
                (mining_data['month'] <= end_month)
            ].copy()
            
            if len(period_data) == 0:
                continue
            
            # Cumulative token output
            cumulative_tokens_regular = period_data['total_reward_regular_miner'].sum()
            cumulative_tokens_top = period_data['total_reward_top_miner'].sum()
            
            # Cumulative electricity costs
            cumulative_electricity_high = period_data['electricity_cost_high'].sum()
            cumulative_electricity_low = period_data['electricity_cost_low'].sum()
            
            # Calculate required token price
            # Target revenue = target profit + cumulative electricity costs
            target_revenue_high = target_profit + cumulative_electricity_high
            target_revenue_low = target_profit + cumulative_electricity_low
            
            required_price_regular_high = target_revenue_high / cumulative_tokens_regular if cumulative_tokens_regular > 0 else 0
            required_price_regular_low = target_revenue_low / cumulative_tokens_regular if cumulative_tokens_regular > 0 else 0
            
            required_price_top_high = target_revenue_high / cumulative_tokens_top if cumulative_tokens_top > 0 else 0
            required_price_top_low = target_revenue_low / cumulative_tokens_top if cumulative_tokens_top > 0 else 0
            
            price_scenarios.append({
                'scenario': scenario_name,
                'description': scenario["description"],
                'start_month': start_month,
                'payback_months': payback_months,
                'target_profit': target_profit,
                'cumulative_tokens_regular': cumulative_tokens_regular,
                'cumulative_tokens_top': cumulative_tokens_top,
                'cumulative_electricity_high': cumulative_electricity_high,
                'cumulative_electricity_low': cumulative_electricity_low,
                'required_price_regular_high_elec': required_price_regular_high,
                'required_price_regular_low_elec': required_price_regular_low,
                'required_price_top_high_elec': required_price_top_high,
                'required_price_top_low_elec': required_price_top_low
            })
        
        return pd.DataFrame(price_scenarios)
    
    def generate_token_price_trajectory(self, mining_data: pd.DataFrame, 
                                      price_scenarios: pd.DataFrame) -> pd.DataFrame:
        """Generate token price trajectory"""
        months = mining_data['month'].values
        
        # Design price curve based on payback requirements
        early_price = price_scenarios[price_scenarios['scenario'] == 'early_mining']['required_price_regular_low_elec'].iloc[0]
        mid_price = price_scenarios[price_scenarios['scenario'] == 'mid_mining']['required_price_regular_low_elec'].iloc[0]
        late_price = price_scenarios[price_scenarios['scenario'] == 'late_mining']['required_price_regular_low_elec'].iloc[0]
        
        prices = []
        for month in months:
            if month <= 3:
                # Early stage high price, supporting 1-month payback
                price = early_price * (1 - 0.05 * (month - 1))  # Slight decline
            elif month <= 6:
                # Mid-stage price, supporting 3-month payback
                price = mid_price * (1 + 0.1 * np.sin(month * np.pi / 6))  # Add volatility
            else:
                # Late-stage price, supporting 6-month payback
                decline_factor = np.exp(-0.1 * (month - 6))
                price = late_price * (0.8 + 0.2 * decline_factor)
            
            prices.append(max(price, 0.001))
        
        return pd.DataFrame({
            'month': months,
            'token_price_usd': prices
        })
    
    def calculate_mining_profitability(self, mining_data: pd.DataFrame, 
                                     price_trajectory: pd.DataFrame) -> pd.DataFrame:
        """Calculate mining profitability"""
        merged = mining_data.merge(price_trajectory, on='month')
        
        # Calculate monthly revenue and profit
        merged['revenue_regular_miner'] = merged['total_reward_regular_miner'] * merged['token_price_usd']
        merged['revenue_top_miner'] = merged['total_reward_top_miner'] * merged['token_price_usd']
        
        merged['profit_regular_high_elec'] = merged['revenue_regular_miner'] - merged['electricity_cost_high']
        merged['profit_regular_low_elec'] = merged['revenue_regular_miner'] - merged['electricity_cost_low']
        merged['profit_top_high_elec'] = merged['revenue_top_miner'] - merged['electricity_cost_high']
        merged['profit_top_low_elec'] = merged['revenue_top_miner'] - merged['electricity_cost_low']
        
        # Calculate cumulative profit
        merged['cumulative_profit_regular_high'] = merged['profit_regular_high_elec'].cumsum()
        merged['cumulative_profit_regular_low'] = merged['profit_regular_low_elec'].cumsum()
        merged['cumulative_profit_top_high'] = merged['profit_top_high_elec'].cumsum()
        merged['cumulative_profit_top_low'] = merged['profit_top_low_elec'].cumsum()
        
        return merged

def create_detailed_analysis_dashboard(mining_data, price_scenarios, price_trajectory, profitability, config):
    """Create detailed analysis dashboard"""
    
    fig, axes = plt.subplots(3, 3, figsize=(24, 18))
    fig.suptitle('AXON Mining Economics Analysis Dashboard\nBased on Whitepaper Exact Parameters and Your Specific Requirements', 
                fontsize=18, fontweight='bold')
    
    # 1. Token Production Analysis
    ax1 = axes[0, 0]
    ax1.plot(mining_data['month'], mining_data['total_reward_regular_miner'], 
             'b-', linewidth=2, label='Regular Miner Monthly Output', marker='o')
    ax1.plot(mining_data['month'], mining_data['total_reward_top_miner'], 
             'r-', linewidth=2, label='Top Miner Monthly Output', marker='s')
    ax1.set_title('Individual Miner AXON Token Monthly Output', fontweight='bold')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('AXON Token Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Miner Count Growth
    ax2 = axes[0, 1]
    ax2.semilogy(mining_data['month'], mining_data['miners_count'], 'g-', linewidth=3, marker='o')
    ax2.axvline(x=6, color='red', linestyle='--', alpha=0.7, label='6-Month Mark')
    ax2.set_title('Network Miner Count Growth\n(50K→300K)', fontweight='bold')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Miner Count (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Dynamic Weight Evolution
    ax3 = axes[0, 2]
    ax3.plot(mining_data['month'], mining_data['w_dl'], 'orange', linewidth=3, label='Domain-Library Weight')
    ax3.plot(mining_data['month'], mining_data['w_df'], 'purple', linewidth=3, label='Data-Feed Weight')
    ax3.set_title('Dynamic Weight Evolution\n(Whitepaper Formulas 34,35)', fontweight='bold')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Weight Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Token Price Trajectory
    ax4 = axes[1, 0]
    ax4.plot(price_trajectory['month'], price_trajectory['token_price_usd'], 
             'purple', linewidth=3, marker='s')
    ax4.set_title('AXON Token Price Trajectory\n(Meeting Payback Requirements)', fontweight='bold')
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Price (USD)')
    ax4.grid(True, alpha=0.3)
    
    # 5. Cumulative Profit Analysis
    ax5 = axes[1, 1]
    ax5.plot(profitability['month'], profitability['cumulative_profit_regular_low'], 
             'b-', linewidth=2, label='Regular Miner (Low Electricity)', marker='o')
    ax5.plot(profitability['month'], profitability['cumulative_profit_regular_high'], 
             'b--', linewidth=2, label='Regular Miner (High Electricity)', marker='o')
    ax5.plot(profitability['month'], profitability['cumulative_profit_top_low'], 
             'r-', linewidth=2, label='Top Miner (Low Electricity)', marker='s')
    ax5.axhline(y=5000, color='orange', linestyle=':', linewidth=2, label='Target Profit ($5000)')
    ax5.set_title('Cumulative Mining Profit', fontweight='bold')
    ax5.set_xlabel('Month')
    ax5.set_ylabel('Cumulative Profit (USD)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Payback Time Analysis
    ax6 = axes[1, 2]
    scenarios = ['Early (<3m)', 'Mid (<6m)', 'Late (>6m)']
    payback_months = price_scenarios['payback_months'].tolist()
    colors = ['green', 'orange', 'red']
    bars = ax6.bar(range(len(scenarios)), payback_months, color=colors, alpha=0.7)
    ax6.set_title('Payback Time for Different Mining Start Periods', fontweight='bold')
    ax6.set_xlabel('Mining Start Period')
    ax6.set_ylabel('Payback Time (Months)')
    ax6.set_xticks(range(len(scenarios)))
    ax6.set_xticklabels(scenarios)
    
    for i, (bar, months) in enumerate(zip(bars, payback_months)):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{months} months', ha='center', va='bottom', fontweight='bold')
    
    # 7. Electricity Cost Ratio Analysis
    ax7 = axes[2, 0]
    revenue_regular = profitability['revenue_regular_miner']
    elec_high = profitability['electricity_cost_high'] 
    elec_low = profitability['electricity_cost_low']
    
    elec_ratio_high = (elec_high / revenue_regular * 100).fillna(0)
    elec_ratio_low = (elec_low / revenue_regular * 100).fillna(0)
    
    ax7.plot(profitability['month'], elec_ratio_high, 'r-', linewidth=2, label='High Electricity (1.2 RMB/kWh)')
    ax7.plot(profitability['month'], elec_ratio_low, 'g-', linewidth=2, label='Low Electricity (0.6 RMB/kWh)')
    ax7.set_title('Electricity Cost as % of Mining Revenue', fontweight='bold')
    ax7.set_xlabel('Month')
    ax7.set_ylabel('Electricity Cost Ratio (%)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Emission Rate Evolution
    ax8 = axes[2, 1]
    ax8.plot(mining_data['month'], mining_data['emission_rate'] * 100, 'b-', linewidth=3, marker='o')
    ax8.set_title('Emission Rate Evolution\n(Whitepaper Formula 31)', fontweight='bold')
    ax8.set_xlabel('Month')
    ax8.set_ylabel('Emission Rate (%)')
    ax8.grid(True, alpha=0.3)
    
    # 9. Block Reward Evolution
    ax9 = axes[2, 2]
    ax9.semilogy(mining_data['month'], mining_data['block_reward'], 'purple', linewidth=3, marker='s')
    ax9.set_title('Block Reward Evolution\n(3-Second Block Time)', fontweight='bold')
    ax9.set_xlabel('Month')
    ax9.set_ylabel('Block Reward (AXON)')
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_comprehensive_analysis():
    """Create comprehensive analysis"""
    
    print("🚀 AXON Mining Economics Analysis")
    print("=" * 60)
    
    # Initialize configuration
    config = MiningConfiguration()
    simulator = AXONMiningSimulator(config)
    
    # Run simulation
    print("📊 Running 12-month mining simulation...")
    mining_data = simulator.simulate_mining_economics(months=12)
    
    # Calculate required token prices
    print("💰 Calculating token prices to meet payback requirements...")
    price_scenarios = simulator.calculate_required_token_price(mining_data)
    
    # Generate price trajectory
    print("📈 Generating token price trajectory...")
    price_trajectory = simulator.generate_token_price_trajectory(mining_data, price_scenarios)
    
    # Calculate profitability
    print("💵 Calculating mining profitability...")
    profitability = simulator.calculate_mining_profitability(mining_data, price_trajectory)
    
    return mining_data, price_scenarios, price_trajectory, profitability, config, simulator

def print_detailed_analysis(mining_data, price_scenarios, price_trajectory, profitability, config, simulator):
    """Print detailed analysis results"""
    
    print("\n" + "="*80)
    print("📋 AXON Mining Economics Analysis Detailed Report")
    print("="*80)
    
    # 1. Network Basic Parameters
    print("\n🔧 Network Basic Parameters:")
    print(f"   Total Supply: {simulator.total_supply:,} AXON")
    print(f"   Initial Miner Count: {config.initial_miners:,}")
    print(f"   Miner Count After 6 Months: {config.miners_at_6_months:,}")
    print(f"   Mining Rig Power: {config.power_consumption_watts}W")
    print(f"   High Electricity Cost: {config.electricity_cost_high*7.2:.2f} RMB/kWh")
    print(f"   Low Electricity Cost: {config.electricity_cost_low*7.2:.2f} RMB/kWh")
    
    # 2. Token Production Analysis
    print("\n📊 Token Production Analysis (First 6 Months):")
    first_6_months = mining_data.head(6)
    for _, row in first_6_months.iterrows():
        print(f"   Month {row['month']}:")
        print(f"     - Miner Count: {row['miners_count']:,}")
        print(f"     - Regular Miner Monthly Output: {row['total_reward_regular_miner']:,.0f} AXON")
        print(f"     - Top Miner Monthly Output: {row['total_reward_top_miner']:,.0f} AXON")
        print(f"     - Block Reward: {row['block_reward']:,.2f} AXON")
    
    # 3. Price Requirement Analysis  
    print("\n💰 Token Price Requirement Analysis:")
    for _, scenario in price_scenarios.iterrows():
        print(f"   {scenario['description']}:")
        print(f"     - Start Time: Month {scenario['start_month']}")
        print(f"     - Payback Period: {scenario['payback_months']} months")
        print(f"     - Cumulative Token Output (Regular): {scenario['cumulative_tokens_regular']:,.0f} AXON")
        print(f"     - Required Price (Low Electricity): ${scenario['required_price_regular_low_elec']:.4f}")
        print(f"     - Required Price (High Electricity): ${scenario['required_price_regular_high_elec']:.4f}")
    
    # 4. Profitability Verification
    print("\n📈 Profitability Verification (Month 6):")
    month_6 = profitability[profitability['month'] == 6].iloc[0]
    print(f"   Token Price: ${month_6['token_price_usd']:.4f}")
    print(f"   Regular Miner Cumulative Profit (Low Electricity): ${month_6['cumulative_profit_regular_low']:,.0f}")
    print(f"   Regular Miner Cumulative Profit (High Electricity): ${month_6['cumulative_profit_regular_high']:,.0f}")
    print(f"   Top Miner Cumulative Profit (Low Electricity): ${month_6['cumulative_profit_top_low']:,.0f}")
    
    # 5. Key Conclusions
    print("\n🎯 Key Conclusions:")
    
    # Check 6-month profit target
    target_met_regular_low = month_6['cumulative_profit_regular_low'] >= config.target_profit_6_months
    target_met_regular_high = month_6['cumulative_profit_regular_high'] >= config.target_profit_6_months
    
    print(f"   ✅ Regular Miner 6-Month Profit Target ($5000): {'Achieved' if target_met_regular_low else 'Not Achieved'} (Low Electricity)")
    print(f"   {'✅' if target_met_regular_high else '❌'} Regular Miner 6-Month Profit Target ($5000): {'Achieved' if target_met_regular_high else 'Not Achieved'} (High Electricity)")
    
    # Token price reasonableness
    initial_price = price_trajectory.iloc[0]['token_price_usd']
    final_price = price_trajectory.iloc[-1]['token_price_usd']
    print(f"   📊 Token Price Range: ${final_price:.4f} - ${initial_price:.4f}")
    print(f"   📉 Price Decline: {(1-final_price/initial_price)*100:.1f}%")
    
    # Electricity cost ratio
    avg_elec_ratio_low = profitability['electricity_cost_low'].sum() / profitability['revenue_regular_miner'].sum() * 100
    avg_elec_ratio_high = profitability['electricity_cost_high'].sum() / profitability['revenue_regular_miner'].sum() * 100
    print(f"   ⚡ Average Electricity Cost Ratio: {avg_elec_ratio_low:.1f}% (Low Electricity), {avg_elec_ratio_high:.1f}% (High Electricity)")

def save_results(mining_data, price_scenarios, price_trajectory, profitability):
    """Save analysis results"""
    
    print("\n💾 Saving analysis results...")
    
    # Save raw data
    mining_data.to_csv('axon_mining_data.csv', index=False)
    price_scenarios.to_csv('axon_price_scenarios.csv', index=False)
    price_trajectory.to_csv('axon_price_trajectory.csv', index=False)
    profitability.to_csv('axon_mining_profitability.csv', index=False)
    
    # Create summary report
    summary = {
        'analysis_date': pd.Timestamp.now(),
        'total_miners_month_1': mining_data.iloc[0]['miners_count'],
        'total_miners_month_6': mining_data.iloc[5]['miners_count'],
        'token_price_month_1': price_trajectory.iloc[0]['token_price_usd'],
        'token_price_month_6': price_trajectory.iloc[5]['token_price_usd'],
        'regular_miner_profit_6m_low_elec': profitability.iloc[5]['cumulative_profit_regular_low'],
        'regular_miner_profit_6m_high_elec': profitability.iloc[5]['cumulative_profit_regular_high'],
        'target_profit_met': profitability.iloc[5]['cumulative_profit_regular_low'] >= 5000
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv('axon_mining_summary.csv', index=False)
    
    print("   ✅ Files saved:")
    print("     - axon_mining_data.csv")
    print("     - axon_price_scenarios.csv") 
    print("     - axon_price_trajectory.csv")
    print("     - axon_mining_profitability.csv")
    print("     - axon_mining_summary.csv")

def main():
    """Main function"""
    # Run analysis
    mining_data, price_scenarios, price_trajectory, profitability, config, simulator = create_comprehensive_analysis()
    
    # Create visualization
    print("🎨 Generating analysis charts...")
    fig = create_detailed_analysis_dashboard(mining_data, price_scenarios, price_trajectory, profitability, config)
    
    # Print detailed analysis
    print_detailed_analysis(mining_data, price_scenarios, price_trajectory, profitability, config, simulator)
    
    # Save results
    save_results(mining_data, price_scenarios, price_trajectory, profitability)
    
    # Display charts
    plt.show()
    
    # Save chart
    fig.savefig('axon_mining_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Chart saved: axon_mining_analysis_dashboard.png")
    
    print("\n✅ AXON Mining Economics Analysis Complete!")
    print("="*80)

if __name__ == "__main__":
    main()