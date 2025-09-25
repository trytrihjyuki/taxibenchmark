# 🚕 Taxi Pricing Benchmark

Simplified implementation of taxi pricing methods from the Hikima et al. paper.

## 🚀 Quick Start

```bash
# 1. Build Docker image (auto-detects platform)
./build.sh

# 2. Run experiment
./run.sh --processing-date 2019-10-06

# 3. Check results
ls experiments/
```

## 📋 Requirements

- Docker
- AWS credentials (in `.env` file or environment variables)

## ⚙️ Configuration

Create a `.env` file:
```bash
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=eu-north-1
S3_BUCKET=magisterka
```

## 🎯 Usage Examples

### Basic Experiment
```bash
./run.sh --processing-date 2019-10-06
```

### Full Configuration
```bash
./run.sh \
    --processing-date 2019-10-06 \
    --vehicle-type green \
    --boroughs Manhattan Brooklyn \
    --methods LP MinMaxCostFlow MAPS LinUCB \
    --start-hour 10 \
    --end-hour 11 \
    --time-delta 30 \
    --num-iter 100
```

## 🧮 Pricing Methods

All methods automatically evaluate with both **PL** and **Sigmoid** acceptance functions:

- **LP** - Linear Programming optimization
- **MinMaxCostFlow** - Min-cost flow algorithm (Hikima et al.)
- **MAPS** - Matching and Pricing in Shared economy
- **LinUCB** - Linear Upper Confidence Bound (contextual bandit)

## 📊 Output Structure

```
experiments/
└── run_20250826_123456_20191006/
    ├── decisions/
    │   ├── time_window_0000.parquet  # Matching decisions
    │   ├── time_window_0001.parquet
    │   └── ...
    └── experiment_summary.json       # Metrics and configuration
```

### Decision Files

Each Parquet file contains:
- `time_window_idx` - Time window index
- `borough` - NYC borough
- `method` - Pricing method used
- `acceptance_function` - PL or Sigmoid
- `requester_id` - Trip requester ID
- `price` - Offered price
- `acceptance_prob` - Acceptance probability

## 🔧 Development

### Without Docker
```bash
# Install dependencies
pip install -r requirements.txt

# Run directly
python run_experiment.py --processing-date 2019-10-06
```

### Architecture

Simplified single-process or parallel execution:
- Loads all data once in main process
- Distributes work to parallel workers (if configured)
- No S3 client serialization issues
- Direct parquet output for analysis

## 📈 Performance

- **Default**: Single worker (sequential)
- **Parallel**: Use `--num-workers N` for parallel processing
- **Data**: ~100 trips per time window for reasonable performance

## 🐛 Troubleshooting

### AWS Credentials
Ensure credentials are set via environment or `.env` file.

### Platform Issues (Apple Silicon)
The build script auto-detects ARM and builds native images for better performance.

### Memory Issues
Reduce `--num-iter` or process fewer boroughs/methods simultaneously.