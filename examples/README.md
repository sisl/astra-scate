# SCATE Examples

This directory contains examples and demonstrations for the SCATE memory corruption attack system.

## Directory Structure

```
examples/
├── README.md                           # This file
├── ast_basic.py                        # Basic AST example
├── ast_hf.py                          # Hugging Face integration example
├── ast_llama.py                       # Llama model example
├── ast_ppo.py                         # PPO training example
├── evaluator_demo.py                  # Evaluator demonstration
├── demos/                             # Comprehensive demo scripts
│   ├── extended_mdp_demo.py           # Extended MDP demo
│   ├── ipo_memory_integration_demo.py # IPO memory integration demo
│   ├── memory_attack_training_demo.py # Memory attack training demo
│   ├── memory_demo.py                 # Basic memory demo
│   ├── memory_reward_demo.py          # Memory reward demo
│   ├── monitoring_evaluation_demo.py  # Monitoring & evaluation demo
│   ├── training_script_demo.py        # Training script demo
│   └── mypy.ini                       # MyPy configuration for demos
└── output/                            # Generated outputs from demos
    ├── checkpoints/                   # Model checkpoints
    ├── results/                       # Training results
    ├── logs/                          # Training logs
    └── [demo outputs]                 # Various demo outputs
```

## Running Examples

### Basic Examples

```bash
# Basic AST example
python3 examples/ast_basic.py

# Hugging Face integration
python3 examples/ast_hf.py

# Llama model example
python3 examples/ast_llama.py

# PPO training example
python3 examples/ast_ppo.py
```

### Comprehensive Demos

```bash
# Memory system demo
python3 examples/demos/memory_demo.py

# Memory reward system demo
python3 examples/demos/memory_reward_demo.py

# Extended MDP demo
python3 examples/demos/extended_mdp_demo.py

# IPO memory integration demo
python3 examples/demos/ipo_memory_integration_demo.py

# Memory attack training demo
python3 examples/demos/memory_attack_training_demo.py

# Training script demo
python3 examples/demos/training_script_demo.py

# Monitoring & evaluation demo
python3 examples/demos/monitoring_evaluation_demo.py
```

## Output Files

All demo outputs are saved to `examples/output/`:

- **`checkpoints/`** - Model checkpoints from training
- **`results/`** - Training results and metrics
- **`logs/`** - Training logs and output files
- **`demo_*.png`** - Visualization plots
- **`demo_*.json`** - Reports and metrics data

## Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install torch numpy matplotlib pandas tqdm pyyaml
```

For optional features:
```bash
pip install tensorboard wandb plotly seaborn
```

## Notes

- All demo scripts are designed to run with mock data and don't require actual model files
- Output files are generated in `examples/output/` to keep the main directory clean
- Some demos may require specific configurations or model files (see individual demo documentation)
