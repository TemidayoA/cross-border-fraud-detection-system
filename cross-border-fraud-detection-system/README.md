# Context-Aware Fraud Detection for Cross-Border Payments

A dynamic signal weighting system that reduces false positives in international payment fraud detection by adapting to corridor-specific behaviour patterns.

## The Challenge

Standard fraud detection models apply uniform thresholds across all transactions, ignoring fundamental differences in how legitimate payments behave across international corridors. A £500 transfer to Nigeria exhibits different velocity patterns, timing distributions, and sender behaviours than an equivalent transfer to Germany, yet traditional models treat them identically.

This mismatch produces two costly outcomes:
- **High false positive rates** that block legitimate remittances and damage customer trust
- **Missed fraud** when anomalous behaviour in one corridor appears normal by global standards

## The Solution

This repository implements a context-aware fraud detection framework that dynamically adjusts signal weights based on:

1. **Origin-Destination Corridor Characteristics**: Each payment corridor has distinct baseline patterns for transaction size, frequency, and timing
2. **Transaction Velocity Normalisation**: Thresholds adapt to corridor-typical activity levels rather than global averages
3. **Historical Corridor Risk Scoring**: Risk indicators are weighted by observed fraud rates within specific corridors

## Results

When deployed in production:
- **12% reduction** in fraud losses
- **90%+ recall rate** maintained (critical for regulatory compliance)
- **Significant reduction** in false positive rates, improving customer experience

## Repository Structure

```
├── docs/                    # Detailed methodology and findings
├── notebooks/               # Step-by-step analysis and experimentation
├── src/                     # Production-ready implementation
│   ├── corridor_risk/       # Corridor profiling and risk scoring
│   ├── signal_weighting/    # Dynamic weight adjustment logic
│   └── evaluation/          # Model evaluation utilities
├── tests/                   # Unit and integration tests
└── data/synthetic/          # Synthetic datasets for demonstration
```

## Key Concepts

### Corridor Risk Taxonomy

Payment corridors are categorised by:
- Historical fraud incidence rates
- Transaction volume and value distributions
- Regulatory environment (AML requirements vary by jurisdiction)
- Payment infrastructure maturity

### Dynamic Signal Weighting

Rather than static rules like "flag transactions over £1,000", the system calculates:

```
adjusted_score = base_score × corridor_weight × velocity_factor × temporal_adjustment
```

Where each component reflects corridor-specific learned parameters.

### Evaluation Framework

Performance is measured against:
- Precision-Recall curves segmented by corridor
- False positive rates at fixed recall thresholds
- Value-weighted fraud capture rates

## Getting Started

```bash
# Clone the repository
git clone https://github.com/TemidayoA/cross-border-fraud-detection.git

# Install dependencies
pip install -r requirements.txt

# Run the demonstration notebook
jupyter notebook notebooks/01_corridor_analysis.ipynb
```

## Technical Stack

- Python 3.9+
- pandas, numpy for data processing
- scikit-learn for baseline models
- Custom implementations for dynamic weighting logic

## Author

**Temidayo Akindahunsi**  Financial Data Analytics | Machine Learning for Fintech

Experience building fraud detection and credit risk systems across emerging and developed markets, including cross-border payment platforms and debt collection operations.

## License

MIT License - See LICENSE file for details.

---

*This project demonstrates techniques developed during production deployment at a cross-border payments fintech. All data used in this repository is synthetic to protect proprietary information.*
