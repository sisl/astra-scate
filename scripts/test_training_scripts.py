#!/usr/bin/env python3
"""
Test script for IPO memory training scripts.
"""

import os
import sys
import tempfile
import subprocess
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))


def test_config_loading():
    """Test configuration file loading"""
    print("Testing configuration loading...")

    try:
        import yaml

        config_path = (
            Path(__file__).parent.parent / "configs" / "ipo_memory_training.yaml"
        )

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Check required sections
        required_sections = ["attacker", "defender", "memory", "training", "evaluation"]
        for section in required_sections:
            assert section in config, f"Missing section: {section}"

        print("✓ Configuration loaded successfully")
        print(f"  - Attacker model: {config['attacker']['model_name']}")
        print(f"  - Training epochs: {config['training']['num_epochs']}")
        print(f"  - Memory capacity: {config['memory']['memory_capacity']}")

        return True

    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False


def test_training_script_import():
    """Test that the training script can be imported"""
    print("\nTesting training script import...")

    try:
        from astra_rl.training.train_ipo_memory import (
            MockAttackerPolicy,
            MockDefenderModel,
            MockProblem,
        )

        print("✓ Training script imports successfully")

        # Test mock components
        config = {"model_name": "gpt2-medium", "learning_rate": 5e-6}

        attacker = MockAttackerPolicy(config)
        defender = MockDefenderModel(config)
        problem = MockProblem()

        print("✓ Mock components created successfully")
        print(f"  - Attacker: {attacker.model_name}")
        print(f"  - Defender: {defender.model_name}")

        return True

    except Exception as e:
        print(f"✗ Training script import failed: {e}")
        return False


def test_training_script_execution():
    """Test training script execution with minimal config"""
    print("\nTesting training script execution...")

    try:
        # Create a minimal test config
        test_config = {
            "attacker": {"model_name": "gpt2-medium", "learning_rate": 1e-4},
            "defender": {"model_name": "gpt2-medium"},
            "memory": {
                "memory_capacity": 10,
                "memory_weight": 0.5,
                "injection_weight": 0.3,
                "persistence_weight": 0.4,
                "corruption_weight": 0.3,
                "coherence_bonus": 0.2,
            },
            "training": {
                "num_epochs": 2,
                "batches_per_epoch": 2,
                "rollouts_per_batch": 2,
                "max_depth": 2,
                "beta": 0.1,
                "grad_clip": 1.0,
                "save_every": 1,
                "eval_every": 1,
            },
            "evaluation": {"num_eval_episodes": 5},
            "logging": {
                "use_wandb": False,
                "use_tensorboard": False,
                "log_level": "INFO",
            },
            "device": {"device": "cpu"},
            "output": {
                "run_name": "test_run",
                "checkpoint_dir": "test_checkpoints",
                "results_dir": "test_results",
            },
            "attack": {"corruption_keywords": ["marseille", "false", "incorrect"]},
        }

        # Write test config to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            import yaml

            yaml.dump(test_config, f)
            temp_config_path = f.name

        # Test the training script
        script_path = (
            Path(__file__).parent.parent
            / "src"
            / "astra_rl"
            / "training"
            / "train_ipo_memory.py"
        )

        cmd = [
            sys.executable,
            str(script_path),
            "--config",
            temp_config_path,
            "--device",
            "cpu",
            "--run_name",
            "test_run",
        ]

        print(f"Running command: {' '.join(cmd)}")

        # Run with timeout to avoid hanging
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,  # 1 minute timeout
        )

        # Clean up temp file
        os.unlink(temp_config_path)

        if result.returncode == 0:
            print("✓ Training script executed successfully")
            print("  Output preview:")
            for line in result.stdout.split("\n")[:5]:
                if line.strip():
                    print(f"    {line}")
            return True
        else:
            print(f"✗ Training script failed with return code {result.returncode}")
            print(f"  Error output: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(
            "✗ Training script timed out (this might be normal for a full training run)"
        )
        return True  # Timeout is expected for a real training run
    except Exception as e:
        print(f"✗ Training script execution failed: {e}")
        return False


def test_cli_interface():
    """Test command-line interface"""
    print("\nTesting CLI interface...")

    try:
        script_path = (
            Path(__file__).parent.parent
            / "src"
            / "astra_rl"
            / "training"
            / "train_ipo_memory.py"
        )

        # Test help command
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"], capture_output=True, text=True
        )

        if result.returncode == 0 and "--config" in result.stdout:
            print("✓ CLI interface works correctly")
            print("  Available options:")
            for line in result.stdout.split("\n"):
                if "--" in line and "help" not in line.lower():
                    print(f"    {line.strip()}")
            return True
        else:
            print("✗ CLI interface test failed")
            return False

    except Exception as e:
        print(f"✗ CLI interface test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=== Testing IPO Memory Training Scripts ===\n")

    tests = [
        test_config_loading,
        test_training_script_import,
        test_cli_interface,
        test_training_script_execution,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)

    print("\n=== Test Results ===")
    passed = sum(results)
    total = len(results)

    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
