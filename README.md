# AXON Tokenomics Analysis Project

A comprehensive tokenomics analysis and simulation framework for AXON Network, based on the official whitepaper formulas and parameters.

## 🚀 Features

- **Formula-Verified Simulation Engine**: Implements all whitepaper formulas with strict verification
- **Mining Economics Analysis**: Comprehensive mining profitability and token price analysis
- **Risk Assessment Framework**: Multi-dimensional risk analysis with Monte Carlo simulations
- **Dynamic Weight Evolution**: Implementation of formulas (34) and (35) for time-dependent reward allocation
- **Participant Evolution Modeling**: Based on Table 2 from the whitepaper
- **English Visualizations**: All charts and reports generated in English for international use

## 📊 Generated Analysis

### Core Tokenomics Charts
- **Architecture Analysis**: Complete tokenomics framework visualization
- **Dynamic Weight Evolution**: KV/PDP weight changes over time
- **Emission Rate Analysis**: Token emission based on formula (31)

### Extended Analysis
- **Risk Assessment Matrix**: Multi-dimensional risk evaluation
- **Monte Carlo Simulations**: Uncertainty analysis with parameter sensitivity
- **Mining Economics Dashboard**: Profitability analysis for different mining scenarios

## 🛠️ Components

### `/accurate_estimate/`
- `v.1.2_sen&risk.py`: Comprehensive visualization framework with whitepaper verification
- `Mining_TkPrice.py`: Mining economics and token price analysis
- `6.6.2.2_v.1.1.py`: Formula-verified implementation based on whitepaper v1.1.0
- `6.6.2.2_v.1.2.py`: Enhanced version with additional features

### Generated Files
- Various CSV files with simulation results
- High-resolution PNG charts for academic use
- Verification reports confirming whitepaper compliance

## 🔬 Whitepaper Compliance

All implementations are strictly verified against:
- Formula (23): Emission rate calculation
- Formulas (24) & (25): Dynamic weight evolution
- Formula (31): Block reward mechanism
- Table 2: Participant allocation over time

## 📈 Usage

1. **Run Core Analysis**:
   ```bash
   python accurate_estimate/v.1.2_sen&risk.py
   ```

2. **Mining Economics Analysis**:
   ```bash
   python accurate_estimate/Mining_TkPrice.py
   ```

3. **Formula Verification**:
   ```bash
   python accurate_estimate/6.6.2.2_v.1.1.py
   ```

## 📋 Requirements

- Python 3.8+
- NumPy, Pandas, Matplotlib, Seaborn
- NetworkX (for network analysis)

## 🎯 Key Results

- ✅ All whitepaper formulas implemented and verified
- ✅ Mining profitability analysis completed
- ✅ Risk assessment framework established
- ✅ English-language visualizations generated
- ✅ Academic-quality charts for presentations

## 📧 Contact

For questions about the analysis or implementation, please refer to the AXON Network whitepaper v1.1.0.
