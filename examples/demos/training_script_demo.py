#!/usr/bin/env python3
"""
Demonstration of IPO memory training scripts.
Shows how to use the training configuration and CLI interface.
"""

import os
import sys
import tempfile
import yaml
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from astra_rl.training.train_ipo_memory import (
    load_config,
    MockAttackerPolicy,
    MockDefenderModel,
    MockProblem,
)
from astra_rl.training.ipo_memory_trainer import MemoryIPOTrainer


def demonstrate_config_usage():
    """Demonstrate configuration file usage"""
    print("=== Configuration Usage Demo ===\n")

    # Load the main config
    config_path = Path(__file__).parent.parent / "configs" / "ipo_memory_training.yaml"
    config = load_config(str(config_path))

    print("✓ Loaded configuration from YAML file")
    print(f"  Config file: {config_path}")
    print(f"  Attacker model: {config['attacker']['model_name']}")
    print(f"  Training epochs: {config['training']['num_epochs']}")
    print(f"  Memory capacity: {config['memory']['memory_capacity']}")
    print(f"  Device: {config['device']['device']}")

    # Show how to modify config programmatically
    print("\n✓ Modifying configuration programmatically:")

    # Create a modified config for demo
    demo_config = config.copy()
    demo_config["training"]["num_epochs"] = 5  # Shorter training for demo
    demo_config["training"]["batches_per_epoch"] = 2
    demo_config["evaluation"]["num_eval_episodes"] = 10
    demo_config["device"]["device"] = "cpu"

    print(f"  Modified epochs: {demo_config['training']['num_epochs']}")
    print(f"  Modified batches: {demo_config['training']['batches_per_epoch']}")
    print(f"  Modified device: {demo_config['device']['device']}")

    return demo_config


def demonstrate_mock_components():
    """Demonstrate mock component usage"""
    print("\n=== Mock Components Demo ===\n")

    # Create mock components
    attacker_config = {"model_name": "gpt2-medium", "learning_rate": 0.000005}
    defender_config = {"model_name": "gpt2-medium"}

    attacker = MockAttackerPolicy(attacker_config, "cpu")
    defender = MockDefenderModel(defender_config, "cpu")
    problem = MockProblem()

    print("✓ Created mock components:")
    print(f"  Attacker: {attacker.model_name}")
    print(f"  Defender: {defender.model_name}")
    print(f"  Parameters: {len(list(attacker.parameters()))}")

    # Test text generation
    print("\n✓ Testing text generation:")

    # Test attacker generation
    injection_prompt = "Hello [INJECT]"
    injection_response = attacker.generate(injection_prompt)
    print(f"  Attacker injection: '{injection_response}'")

    trigger_prompt = "Hello [TRIGGER]"
    trigger_response = attacker.generate(trigger_prompt)
    print(f"  Attacker trigger: '{trigger_response}'")

    # Test defender generation
    defender_response = defender.generate("What's the capital of France?")
    print(f"  Defender response: '{defender_response}'")


def demonstrate_trainer_initialization():
    """Demonstrate trainer initialization"""
    print("\n=== Trainer Initialization Demo ===\n")

    # Create config
    config = {
        "memory": {
            "memory_capacity": 10,
            "memory_weight": 0.6,
            "injection_weight": 0.3,
            "persistence_weight": 0.4,
            "corruption_weight": 0.3,
            "coherence_bonus": 0.2,
        },
        "memory_weight": 0.6,
        "multi_session": True,
        "session_gap_steps": 0,
        "max_depth": 4,
    }

    # Initialize trainer
    problem = MockProblem()
    trainer = MemoryIPOTrainer(problem, config, beta=0.1)

    print("✓ MemoryIPOTrainer initialized:")
    print(f"  Memory weight: {trainer.memory_weight}")
    print(f"  Multi-session: {trainer.multi_session}")
    print(f"  Max depth: {trainer.max_depth}")
    print(f"  Beta: {trainer.beta}")

    # Test rollout generation
    print("\n✓ Testing rollout generation:")

    attacker = MockAttackerPolicy({"model_name": "gpt2-medium"}, "cpu")
    defender = MockDefenderModel({"model_name": "gpt2-medium"}, "cpu")

    rollouts = trainer.generate_rollout(attacker, defender, max_depth=4)

    print(f"  Generated {len(rollouts)} rollouts")
    for i, rollout in enumerate(rollouts):
        print(f"    Phase {i + 1}: {rollout.phase}")
        print(f"      Actions: {len(rollout.actions)}")
        print(f"      Responses: {len(rollout.responses)}")
        if rollout.actions:
            print(f"      Example: '{rollout.actions[0]}' → '{rollout.responses[0]}'")


def demonstrate_cli_usage():
    """Demonstrate CLI usage examples"""
    print("\n=== CLI Usage Examples ===\n")

    print("✓ Command-line interface examples:")
    print("  Basic training:")
    print("    python train_ipo_memory.py --config configs/ipo_memory_training.yaml")
    print()
    print("  With custom parameters:")
    print("    python train_ipo_memory.py \\")
    print("      --config configs/ipo_memory_training.yaml \\")
    print("      --device cuda \\")
    print("      --run_name my_experiment")
    print()
    print("  With logging:")
    print("    python train_ipo_memory.py \\")
    print("      --config configs/ipo_memory_training.yaml \\")
    print("      --use_wandb \\")
    print("      --use_tensorboard \\")
    print("      --run_name logged_experiment")
    print()
    print("  Debug mode:")
    print("    python train_ipo_memory.py \\")
    print("      --config configs/ipo_memory_training.yaml \\")
    print("      --debug")


def demonstrate_config_customization():
    """Demonstrate how to create custom configurations"""
    print("\n=== Configuration Customization Demo ===\n")

    # Create a custom config for quick testing
    custom_config = {
        "attacker": {
            "model_name": "gpt2-small",
            "learning_rate": 0.0001,
            "weight_decay": 0.01,
        },
        "defender": {"model_name": "gpt2-small"},
        "memory": {
            "memory_capacity": 5,
            "memory_weight": 0.7,
            "injection_weight": 0.3,
            "persistence_weight": 0.4,
            "corruption_weight": 0.3,
            "coherence_bonus": 0.2,
        },
        "training": {
            "num_epochs": 10,
            "batches_per_epoch": 5,
            "rollouts_per_batch": 4,
            "max_depth": 2,
            "beta": 0.2,
            "grad_clip": 1.0,
            "multi_session": True,
            "save_every": 5,
            "eval_every": 2,
        },
        "evaluation": {"num_eval_episodes": 20},
        "logging": {"use_wandb": False, "use_tensorboard": False, "log_level": "INFO"},
        "device": {"device": "cpu"},
        "output": {
            "run_name": "quick_test",
            "checkpoint_dir": "test_checkpoints",
            "results_dir": "test_results",
        },
        "attack": {"corruption_keywords": ["marseille", "false", "incorrect"]},
    }

    print("✓ Custom configuration for quick testing:")
    print(f"  Epochs: {custom_config['training']['num_epochs']}")
    print(f"  Memory capacity: {custom_config['memory']['memory_capacity']}")
    print(f"  Device: {custom_config['device']['device']}")
    print(f"  Run name: {custom_config['output']['run_name']}")

    # Save custom config to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(custom_config, f)
        temp_config_path = f.name

    print(f"\n✓ Saved custom config to: {temp_config_path}")

    # Load and verify the custom config
    loaded_config = load_config(temp_config_path)
    print(f"✓ Loaded custom config: {loaded_config['output']['run_name']}")

    # Clean up
    os.unlink(temp_config_path)
    print("✓ Cleaned up temporary file")


def main():
    """Run all demonstrations"""
    print("=== IPO Memory Training Scripts Demo ===\n")

    try:
        demonstrate_config_usage()
        demonstrate_mock_components()
        demonstrate_trainer_initialization()
        demonstrate_cli_usage()
        demonstrate_config_customization()

        print("\n=== Demo Summary ===")
        print("✓ Configuration loading and customization")
        print("✓ Mock component creation and usage")
        print("✓ Trainer initialization and rollout generation")
        print("✓ CLI interface examples")
        print("✓ Custom configuration creation")

        print("\n🎯 Next steps:")
        print(
            "  1. Run: python train_ipo_memory.py --config configs/ipo_memory_training.yaml"
        )
        print("  2. Monitor training with tensorboard or wandb")
        print("  3. Evaluate trained models")
        print("  4. Customize configurations for your experiments")

    except Exception as e:
        print(f"✗ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
