# IPO Memory Training Scripts Guide

This guide explains how to use the IPO memory training scripts for SCATE memory corruption attacks.

## Overview

The training scripts provide a complete pipeline for training memory corruption attacks using IPO (Identity Preference Optimization) with ASTPrompter integration. The system includes:

- **Main training script** (`train_ipo_memory.py`)
- **Configuration management** (YAML-based)
- **Command-line interface** with logging support
- **Mock components** for testing and development

## Quick Start

### 1. Basic Training

```bash
# Run with default configuration
PYTHONPATH=src python3 src/astra_rl/training/train_ipo_memory.py \
    --config configs/ipo_memory_training.yaml
```

### 2. Custom Training

```bash
# Run with custom parameters
PYTHONPATH=src python3 src/astra_rl/training/train_ipo_memory.py \
    --config configs/ipo_memory_training.yaml \
    --device cuda \
    --run_name my_experiment
```

### 3. With Logging

```bash
# Run with wandb and tensorboard logging
PYTHONPATH=src python3 src/astra_rl/training/train_ipo_memory.py \
    --config configs/ipo_memory_training.yaml \
    --use_wandb \
    --use_tensorboard \
    --run_name logged_experiment
```

## Configuration

### Configuration File Structure

The training configuration is defined in `configs/ipo_memory_training.yaml`:

```yaml
# Attacker model configuration
attacker:
  model_name: "gpt2-medium"
  learning_rate: 0.000005
  weight_decay: 0.01

# Defender model configuration
defender:
  model_name: "gpt2-medium"
  temperature: 0.7

# Memory-specific configuration
memory:
  memory_capacity: 20
  memory_weight: 0.5
  injection_weight: 0.3
  persistence_weight: 0.4
  corruption_weight: 0.3

# Training configuration
training:
  num_epochs: 100
  batches_per_epoch: 10
  rollouts_per_batch: 8
  max_depth: 4
  beta: 0.1

# Evaluation configuration
evaluation:
  num_eval_episodes: 50

# Logging configuration
logging:
  use_wandb: false
  use_tensorboard: true
  log_level: "INFO"
```

### Key Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `training.num_epochs` | Number of training epochs | 100 |
| `training.batches_per_epoch` | Batches per epoch | 10 |
| `training.rollouts_per_batch` | Rollouts per batch | 8 |
| `training.max_depth` | Maximum rollout depth | 4 |
| `training.beta` | IPO beta parameter | 0.1 |
| `memory.memory_capacity` | Memory buffer capacity | 20 |
| `memory.memory_weight` | Weight for memory rewards | 0.5 |
| `evaluation.num_eval_episodes` | Episodes for evaluation | 50 |

## Command-Line Interface

### Available Options

```bash
python train_ipo_memory.py --help
```

| Option | Description | Default |
|--------|-------------|---------|
| `--config` | Path to config file | Required |
| `--device` | Device to use (cuda/cpu) | From config |
| `--run_name` | Name for this run | From config |
| `--use_wandb` | Enable wandb logging | False |
| `--use_tensorboard` | Enable tensorboard logging | False |
| `--debug` | Enable debug mode | False |

### Examples

#### Basic Training
```bash
PYTHONPATH=src python3 src/astra_rl/training/train_ipo_memory.py \
    --config configs/ipo_memory_training.yaml
```

#### GPU Training with Custom Name
```bash
PYTHONPATH=src python3 src/astra_rl/training/train_ipo_memory.py \
    --config configs/ipo_memory_training.yaml \
    --device cuda \
    --run_name gpu_experiment_v1
```

#### Debug Mode
```bash
PYTHONPATH=src python3 src/astra_rl/training/train_ipo_memory.py \
    --config configs/ipo_memory_training.yaml \
    --debug
```

## Training Process

### 1. Initialization
- Load configuration from YAML file
- Initialize mock attacker and defender models
- Set up logging (wandb/tensorboard)
- Create output directories

### 2. Training Loop
For each epoch:
1. **Rollout Generation**: Generate memory-aware rollouts with injection/trigger phases
2. **Preference Pair Creation**: Create memory-aware preference pairs
3. **IPO Loss Computation**: Compute IPO loss with memory corruption bonuses
4. **Model Update**: Update attacker model parameters
5. **Evaluation**: Evaluate model performance (every N epochs)
6. **Checkpointing**: Save model checkpoints (every N epochs)

### 3. Metrics Tracking
- **Loss**: IPO loss with memory bonuses
- **Success Rate**: Percentage of successful memory corruption attacks
- **Memory Corruption Score**: Average corruption detection score
- **Reward**: Average episode rewards

## Output Files

### Directory Structure
```
checkpoints/
├── run_name/
│   ├── epoch_10.pt
│   ├── epoch_20.pt
│   └── epoch_100.pt
results/
├── run_name/
│   ├── run_name.log
│   └── final_metrics.json
runs/
└── run_name/
    ├── events.out.tfevents.*
    └── ...
```

### Checkpoint Format
```python
{
    'epoch': int,
    'attacker_state_dict': dict,
    'optimizer_state_dict': dict,
    'config': dict,
    'timestamp': str,
    'best_success_rate': float,
    'final_metrics': dict
}
```

## Monitoring Training

### Tensorboard
```bash
# Start tensorboard
tensorboard --logdir runs/

# View at http://localhost:6006
```

### Wandb
```bash
# Enable wandb in config
logging:
  use_wandb: true
  wandb_project: "scate-memory"
```

### Log Files
Training logs are saved to `results/run_name/run_name.log` with detailed information about:
- Configuration parameters
- Training progress
- Evaluation results
- Error messages

## Customization

### Creating Custom Configurations

```python
import yaml

# Create custom config
custom_config = {
    'training': {
        'num_epochs': 50,
        'batches_per_epoch': 5,
        'rollouts_per_batch': 4
    },
    'memory': {
        'memory_capacity': 10,
        'memory_weight': 0.7
    },
    'device': {
        'device': 'cpu'
    }
}

# Save to file
with open('my_config.yaml', 'w') as f:
    yaml.dump(custom_config, f)
```

### Modifying Attack Parameters

```yaml
attack:
  injection_markers:
    - "by the way"
    - "note for"
    - "remember"
    - "keep in mind"
  
  trigger_keywords:
    - "what"
    - "how"
    - "when"
    - "where"
  
  corruption_keywords:
    - "marseille"
    - "false"
    - "incorrect"
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Make sure PYTHONPATH includes src directory
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
```

#### 2. CUDA Out of Memory
```yaml
# Reduce batch size in config
training:
  batches_per_epoch: 5
  rollouts_per_batch: 4

# Or use CPU
device:
  device: "cpu"
```

#### 3. Configuration Errors
```bash
# Validate config file
python -c "import yaml; yaml.safe_load(open('configs/ipo_memory_training.yaml'))"
```

### Debug Mode
```bash
# Enable debug logging
python train_ipo_memory.py --config config.yaml --debug
```

## Examples

### Quick Test Run
```bash
# Create a quick test config
cat > quick_test.yaml << EOF
training:
  num_epochs: 5
  batches_per_epoch: 2
  rollouts_per_batch: 2
memory:
  memory_capacity: 5
device:
  device: "cpu"
output:
  run_name: "quick_test"
EOF

# Run quick test
PYTHONPATH=src python3 src/astra_rl/training/train_ipo_memory.py \
    --config quick_test.yaml
```

### Production Training
```bash
# Full training with logging
PYTHONPATH=src python3 src/astra_rl/training/train_ipo_memory.py \
    --config configs/ipo_memory_training.yaml \
    --device cuda \
    --use_wandb \
    --use_tensorboard \
    --run_name production_run_v1
```

## Integration with ASTPrompter

The training scripts are designed to integrate with ASTPrompter's IPO implementation. To use with real ASTPrompter models:

1. Replace `MockAttackerPolicy` with real `AttackerPolicy`
2. Replace `MockDefenderModel` with real `DefenderModel`
3. Replace `MockProblem` with real ASTPrompter `Problem`
4. Update model loading in the training script

## Next Steps

1. **Run Training**: Start with the basic training command
2. **Monitor Progress**: Use tensorboard or wandb to track metrics
3. **Evaluate Results**: Check checkpoint files and evaluation metrics
4. **Customize**: Modify configurations for your specific experiments
5. **Scale Up**: Increase model sizes and training parameters for production runs
