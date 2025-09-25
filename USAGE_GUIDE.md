# 📚 Enhanced Taxi Pricing Framework Usage Guide

This guide shows how to use the enhanced features of the taxi pricing experiment framework.

## 🚀 Quick Start

### 1. Run Enhanced Experiments

```bash
# Run experiment with automatic S3 storage
./run.sh --processing-date 2019-10-06 \
         --methods MinMaxCostFlow MAPS LinUCB \
         --boroughs Manhattan \
         --num-iter 100
```

Results are automatically saved to:
- **Local**: `experiments/run_YYYYMMDD_HHMMSS_YYYYMMDD/`
- **S3**: `s3://taxi-benchmark/experiments/{experiment_id}/`

### 2. Analyze Results

```bash
# Launch Jupyter notebook for analysis
jupyter notebook analysis_notebook.ipynb
```

Or analyze programmatically:

```python
from src.data import S3ResultsUploader

# Load recent experiments
analyzer = ExperimentAnalyzer()
experiments = analyzer.list_experiments()
decisions_df = analyzer.load_decisions_data(experiments[0])

# Quick analysis
print(f"Loaded {len(decisions_df):,} decisions")
print(decisions_df.groupby('method')['profit'].mean())
```

## 📊 Enhanced Decision Data Structure

Each decision record now includes:

```python
{
    'time_window_idx': 0,              # Time window index
    'borough': 'Manhattan',            # NYC borough
    'method': 'MinMaxCostFlow',        # Pricing method
    'acceptance_function': 'PL',       # PL or Sigmoid
    'requester_id': 42,                # Trip requester ID
    'price': 12.50,                    # Offered price ($)
    'acceptance_prob': 0.75,           # Predicted acceptance probability
    'sampled_decision': 1,             # ✨ NEW: Actual sampled decision (0/1)
    'was_matched': 1,                  # ✨ NEW: Whether matched in optimization
    'profit': 12.50,                   # ✨ NEW: Profit from this decision
    'compute_time': 0.0234             # ✨ NEW: Method computation time (seconds)
}
```

## 🔍 Data Analysis Examples

### Method Performance Comparison

```python
# Compare methods across key metrics
performance = decisions_df.groupby(['method', 'acceptance_function']).agg({
    'profit': ['mean', 'std', 'sum'],
    'sampled_decision': 'mean',        # Acceptance rate
    'compute_time': 'mean',            # Average computation time
    'price': 'mean'                    # Average price
}).round(4)

print("📊 Method Performance:")
display(performance)
```

### Profit Analysis

```python
# Find most profitable method
best_method = decisions_df.groupby('method')['profit'].mean().idxmax()
best_profit = decisions_df.groupby('method')['profit'].mean().max()

print(f"🏆 Best method: {best_method} (${best_profit:.2f} avg profit)")

# Analyze profit distribution
import matplotlib.pyplot as plt
decisions_df.boxplot(column='profit', by='method', figsize=(12, 6))
plt.title('💰 Profit Distribution by Method')
plt.show()
```

### Acceptance Function Analysis

```python
# Compare PL vs Sigmoid acceptance functions
func_comparison = decisions_df.groupby(['acceptance_function', 'method']).agg({
    'profit': 'mean',
    'sampled_decision': 'mean',
    'acceptance_prob': 'mean'
}).round(4)

print("📈 PL vs Sigmoid Comparison:")
display(func_comparison)
```

### Performance & Scalability

```python
# Analyze computation time by method
perf_analysis = decisions_df.groupby('method').agg({
    'compute_time': ['mean', 'std'],
    'profit': 'mean'
}).round(4)

# Calculate profit per second (efficiency)
perf_analysis['profit_per_second'] = (
    perf_analysis[('profit', 'mean')] / 
    perf_analysis[('compute_time', 'mean')]
)

print("⚡ Performance Analysis:")
display(perf_analysis)
```

### Decision Sampling Analysis

```python
# Analyze relationship between predicted and actual decisions
import numpy as np

# Create probability bins
decisions_df['prob_bin'] = pd.cut(
    decisions_df['acceptance_prob'], 
    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
    labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
)

# Calculate calibration
calibration = decisions_df.groupby('prob_bin').agg({
    'acceptance_prob': 'mean',    # Predicted
    'sampled_decision': 'mean',   # Actual
    'profit': 'mean'
}).round(4)

print("🎯 Probability Calibration:")
display(calibration)
```

## 📈 Visualization Examples

### Basic Comparison Charts

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Set up subplot grid
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Profit by method
sns.barplot(data=decisions_df, x='method', y='profit', ax=axes[0,0])
axes[0,0].set_title('💰 Average Profit by Method')
axes[0,0].tick_params(axis='x', rotation=45)

# 2. Acceptance rate by method
acceptance_rate = decisions_df.groupby('method')['sampled_decision'].mean()
acceptance_rate.plot(kind='bar', ax=axes[0,1])
axes[0,1].set_title('✅ Acceptance Rate by Method')
axes[0,1].set_ylabel('Rate')

# 3. Computation time by method
sns.boxplot(data=decisions_df, x='method', y='compute_time', ax=axes[1,0])
axes[1,0].set_title('⏱️ Computation Time Distribution')
axes[1,0].tick_params(axis='x', rotation=45)

# 4. Price vs acceptance scatter
sns.scatterplot(data=decisions_df.sample(1000), 
                x='price', y='acceptance_prob', 
                hue='method', alpha=0.6, ax=axes[1,1])
axes[1,1].set_title('💵 Price vs Acceptance Probability')

plt.tight_layout()
plt.show()
```

### Interactive Plotly Visualizations

```python
import plotly.express as px

# Interactive profit comparison
fig = px.box(decisions_df, x='method', y='profit', 
             color='acceptance_function',
             title='📊 Interactive Profit Distribution')
fig.show()

# 3D scatter plot
sample_data = decisions_df.sample(min(500, len(decisions_df)))
fig3d = px.scatter_3d(sample_data, x='price', y='acceptance_prob', z='profit',
                     color='method', title='🎯 3D Analysis: Price vs Acceptance vs Profit')
fig3d.show()
```

## 🗂️ Export and Reporting

### Export Results

```python
# Export summary results
method_summary = decisions_df.groupby(['method', 'acceptance_function']).agg({
    'profit': ['mean', 'std', 'sum'],
    'sampled_decision': 'mean',
    'compute_time': 'mean',
    'price': 'mean'
}).round(4)

# Save to CSV
method_summary.to_csv('method_performance_summary.csv')

# Export sample of detailed decisions
decisions_sample = decisions_df.sample(n=min(10000, len(decisions_df)))
decisions_sample.to_csv('detailed_decisions_sample.csv', index=False)
```

### Generate HTML Report

```python
# Create automated HTML report
html_report = f"""
<h1>Experiment Analysis Report</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

<h2>Summary</h2>
<ul>
<li>Total Decisions: {len(decisions_df):,}</li>
<li>Total Profit: ${decisions_df['profit'].sum():,.2f}</li>
<li>Best Method: {decisions_df.groupby('method')['profit'].mean().idxmax()}</li>
<li>Average Acceptance Rate: {decisions_df['sampled_decision'].mean()*100:.1f}%</li>
</ul>

<h2>Method Performance</h2>
{method_summary.to_html()}
"""

with open('experiment_report.html', 'w') as f:
    f.write(html_report)
```

## 🔧 Advanced Configuration

### Custom Experiment Configuration

```python
from src.core import ExperimentConfig
from src.core.types import VehicleType, Borough, PricingMethod
from datetime import date

# Create custom configuration
config = ExperimentConfig(
    processing_date=date(2019, 10, 6),
    vehicle_type=VehicleType.GREEN,
    boroughs=[Borough.MANHATTAN, Borough.BROOKLYN],
    methods=[
        PricingMethod.MINMAX_COSTFLOW,
        PricingMethod.MAPS,
        PricingMethod.LINUCB
    ],
    start_hour=10,
    end_hour=20,
    time_delta=30,  # 30-minute windows
    num_iter=100,
    num_workers=4,
    s3_results_bucket="taxi-benchmark"
)

# Run experiment
from src.experiments import ExperimentRunner
runner = ExperimentRunner(config)
summary = runner.run()
```

### Load Multiple Experiments

```python
# Compare multiple experiments
analyzer = ExperimentAnalyzer()
experiments = analyzer.list_experiments()

all_results = []
for exp_id in experiments[:3]:  # Compare last 3 experiments
    df = analyzer.load_decisions_data(exp_id)
    df['experiment_id'] = exp_id
    all_results.append(df)

combined_df = pd.concat(all_results, ignore_index=True)

# Cross-experiment analysis
experiment_comparison = combined_df.groupby(['experiment_id', 'method']).agg({
    'profit': 'mean',
    'sampled_decision': 'mean',
    'compute_time': 'mean'
}).round(4)

print("🔄 Cross-Experiment Comparison:")
display(experiment_comparison)
```

## 🎯 Best Practices

### 1. Experiment Design
- Use consistent time windows for fair comparison
- Run sufficient iterations (num_iter >= 100) for statistical significance
- Test multiple boroughs to understand geographic differences

### 2. Data Analysis
- Always check data quality before analysis
- Use sampling for large datasets in visualizations
- Compare both acceptance functions (PL vs Sigmoid)

### 3. Performance Optimization
- Use parallel processing for large experiments
- Monitor S3 costs for frequent uploads
- Cache frequently accessed data locally

### 4. Result Interpretation
- Consider both profit and acceptance rates
- Analyze computation time for scalability
- Look at profit variance for risk assessment

## 📞 Troubleshooting

### AWS/S3 Issues
```bash
# Check AWS credentials
aws configure list

# Test S3 access
aws s3 ls s3://taxi-benchmark/
```

### Data Loading Issues
```python
# Debug S3 loading
analyzer = ExperimentAnalyzer()
try:
    experiments = analyzer.list_experiments()
    print(f"Found {len(experiments)} experiments")
except Exception as e:
    print(f"Error: {e}")
```

### Memory Issues
```python
# Use sampling for large datasets
sample_size = min(50000, len(decisions_df))
decisions_sample = decisions_df.sample(n=sample_size)
```

---

🎉 **Happy Analyzing!** The enhanced framework provides everything you need for comprehensive taxi pricing analysis. 