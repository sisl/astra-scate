#!/usr/bin/env python3
"""
Demonstration of SCATE monitoring and evaluation tools.
Shows how to use training monitoring, model evaluation, and metrics tracking.
"""

import os
import sys
import time
import json
import numpy as np
import tempfile
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from astra_rl.training.monitor_training import TrainingMonitor
from astra_rl.training.evaluate_trained_model import ModelEvaluator
from astra_rl.training.metrics_tracker import MetricsTracker
from astra_rl.training.simple_evaluation_dashboard import SimpleEvaluationDashboard


def demonstrate_metrics_tracking():
    """Demonstrate real-time metrics tracking"""
    print("=== Metrics Tracking Demo ===\n")

    # Create metrics tracker
    tracker = MetricsTracker("demo_run", "examples/output/demo_metrics")

    print("✓ Created metrics tracker")
    print("  - Real-time monitoring enabled")
    print("  - Episode metrics tracking")
    print("  - Training step metrics tracking")
    print("  - Evaluation metrics tracking")

    # Simulate training data
    print("\n✓ Simulating training data...")

    for episode in range(20):
        # Simulate episode
        episode_data = {
            "reward": np.random.random() * 2 - 1,  # -1 to 1
            "success": np.random.random() > 0.7,  # 30% success rate
            "corruption_score": np.random.random(),
            "injection_success": np.random.random() > 0.5,
            "persistence_score": np.random.random(),
            "episode_time": np.random.random() * 10,
        }

        tracker.log_episode(episode_data)

        # Simulate training step
        if episode % 2 == 0:  # Every other episode
            step_data = {
                "loss": np.random.random() * 2,
                "grad_norm": np.random.random() * 5,
                "batch_time": np.random.random() * 5,
            }
            tracker.log_training_step(step_data)

        # Simulate evaluation
        if episode % 5 == 0:  # Every 5 episodes
            eval_data = {
                "success_rate": np.random.random(),
                "corruption_score": np.random.random(),
                "eval_time": np.random.random() * 30,
            }
            tracker.log_evaluation(eval_data)

    # Get current metrics
    current_metrics = tracker.get_current_metrics()
    print("\n✓ Current metrics summary:")
    print(f"  Episodes: {current_metrics.get('episodes', {}).get('total', 0)}")
    print(
        f"  Avg reward: {current_metrics.get('episodes', {}).get('avg_reward', 0):.3f}"
    )
    print(
        f"  Success rate: {current_metrics.get('episodes', {}).get('avg_success', 0):.3f}"
    )
    print(
        f"  Training steps: {current_metrics.get('training', {}).get('total_steps', 0)}"
    )
    print(
        f"  Evaluations: {current_metrics.get('evaluation', {}).get('total_evals', 0)}"
    )

    # Generate plots
    print("\n✓ Generating plots...")
    tracker.plot_training_progress("examples/output/demo_training_progress.png")
    tracker.plot_performance_analysis("examples/output/demo_performance_analysis.png")

    # Generate report
    report = tracker.generate_report()
    print("✓ Generated comprehensive report")
    print(f"  Total episodes: {report['summary']['episodes']['total']}")
    print(f"  Success rate: {report['summary']['episodes']['success_rate']:.3f}")
    print(f"  Avg reward: {report['summary']['episodes']['avg_reward']:.3f}")

    # Stop monitoring
    tracker.stop_monitoring()
    print("✓ Metrics tracking completed")

    return tracker


def demonstrate_training_monitoring():
    """Demonstrate training monitoring capabilities"""
    print("\n=== Training Monitoring Demo ===\n")

    # Create mock training data
    mock_metrics = {
        "Loss/train": {
            "step": list(range(0, 50, 2)),
            "value": [2.0 - i * 0.03 + np.random.normal(0, 0.1) for i in range(25)],
            "wall_time": [time.time() + i for i in range(25)],
        },
        "Reward/train": {
            "step": list(range(0, 50, 2)),
            "value": [0.1 + i * 0.02 + np.random.normal(0, 0.05) for i in range(25)],
            "wall_time": [time.time() + i for i in range(25)],
        },
        "Success/train": {
            "step": list(range(0, 50, 2)),
            "value": [0.1 + i * 0.015 + np.random.normal(0, 0.02) for i in range(25)],
            "wall_time": [time.time() + i for i in range(25)],
        },
        "Eval/success_rate": {
            "step": list(range(0, 50, 10)),
            "value": [0.2 + i * 0.1 + np.random.normal(0, 0.05) for i in range(5)],
            "wall_time": [time.time() + i for i in range(5)],
        },
        "Eval/avg_corruption_score": {
            "step": list(range(0, 50, 10)),
            "value": [0.1 + i * 0.08 + np.random.normal(0, 0.03) for i in range(5)],
            "wall_time": [time.time() + i for i in range(5)],
        },
    }

    # Convert to DataFrame format
    import pandas as pd

    metrics_dfs = {}
    for name, data in mock_metrics.items():
        metrics_dfs[name] = pd.DataFrame(data)

    # Create monitor
    monitor = TrainingMonitor(".", "demo_run")

    print("✓ Created training monitor")
    print("  - Tensorboard log loading")
    print("  - Training progress visualization")
    print("  - Memory analysis")
    print("  - Performance recommendations")

    # Generate plots
    print("\n✓ Generating training progress plots...")
    monitor.plot_training_progress(
        metrics_dfs, "examples/output/demo_training_progress.png"
    )
    monitor.plot_memory_analysis(
        metrics_dfs, "examples/output/demo_memory_analysis.png"
    )

    # Generate report
    print("✓ Generating training report...")
    report = monitor.generate_training_report(metrics_dfs)

    print(f"  Final loss: {report['summary'].get('final_loss', 0):.3f}")
    print(f"  Final success rate: {report['summary'].get('final_success_rate', 0):.3f}")
    print(f"  Final reward: {report['summary'].get('final_reward', 0):.3f}")

    if report["recommendations"]:
        print(f"  Recommendations: {len(report['recommendations'])}")
        for i, rec in enumerate(report["recommendations"][:2], 1):
            print(f"    {i}. {rec}")

    # Save report
    with open("examples/output/demo_training_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("✓ Training report saved")

    return monitor


def demonstrate_model_evaluation():
    """Demonstrate model evaluation capabilities"""
    print("\n=== Model Evaluation Demo ===\n")

    # Create mock checkpoint and config
    import torch

    mock_checkpoint = {
        "epoch": 50,
        "attacker_state_dict": {
            "param_0": torch.randn(100, 100),
            "param_1": torch.randn(100, 100),
            "param_2": torch.randn(100, 100),
        },
        "optimizer_state_dict": {"state": "mock_optimizer"},
        "config": {
            "attacker": {"model_name": "gpt2-medium"},
            "defender": {"model_name": "gpt2-medium"},
            "training": {"beta": 0.1},
            "attack": {"corruption_keywords": ["marseille", "false", "incorrect"]},
        },
    }

    mock_config = {
        "attacker": {"model_name": "gpt2-medium"},
        "defender": {"model_name": "gpt2-medium"},
        "training": {"beta": 0.1, "max_depth": 4},
        "memory": {
            "memory_capacity": 20,
            "memory_weight": 0.5,
            "injection_weight": 0.3,
            "persistence_weight": 0.4,
            "corruption_weight": 0.3,
            "coherence_bonus": 0.2,
        },
        "attack": {"corruption_keywords": ["marseille", "false", "incorrect"]},
        "device": {"device": "cpu"},
    }

    # Save mock checkpoint
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        import torch

        torch.save(mock_checkpoint, f.name)
        checkpoint_path = f.name

    try:
        # Create evaluator
        evaluator = ModelEvaluator(checkpoint_path, mock_config)

        print("✓ Created model evaluator")
        print("  - Checkpoint loading")
        print("  - Attack success evaluation")
        print("  - Pattern analysis")
        print("  - Performance metrics")

        # Run evaluation
        print("\n✓ Running model evaluation...")
        results = evaluator.evaluate_attack_success(num_episodes=20)

        summary = results["summary"]
        print(f"  Success rate: {summary['success_rate']:.2%}")
        print(f"  Injection success rate: {summary['injection_success_rate']:.2%}")
        print(f"  Avg corruption score: {summary['avg_corruption_score']:.3f}")
        print(f"  Avg reward: {summary['avg_reward']:.3f}")

        # Analyze patterns
        pattern_analysis = evaluator.evaluate_attack_patterns(results)
        print(f"  Total patterns: {pattern_analysis['total_patterns']}")
        print(f"  Unique patterns: {pattern_analysis['unique_patterns']}")

        # Generate plots
        print("\n✓ Generating evaluation plots...")
        evaluator.plot_evaluation_results(
            results, "examples/output/demo_evaluation_results.png"
        )

        # Generate report
        print("✓ Generating evaluation report...")
        report = evaluator.generate_evaluation_report(results)

        if report["recommendations"]:
            print(f"  Recommendations: {len(report['recommendations'])}")
            for i, rec in enumerate(report["recommendations"][:2], 1):
                print(f"    {i}. {rec}")

        # Save report
        with open("examples/output/demo_evaluation_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print("✓ Evaluation report saved")

    finally:
        # Clean up
        os.unlink(checkpoint_path)

    return results


def demonstrate_evaluation_dashboard():
    """Demonstrate evaluation dashboard"""
    print("\n=== Evaluation Dashboard Demo ===\n")

    # Create mock evaluation data
    mock_data = {
        "episodes": [
            {
                "reward": np.random.random() * 2 - 1,
                "success": np.random.random() > 0.7,
                "corruption_score": np.random.random(),
                "injection_success": np.random.random() > 0.5,
                "persistence_score": np.random.random(),
                "pattern": f"pattern_{i % 3}",
            }
            for i in range(50)
        ],
        "training": [
            {
                "loss": 2.0 - i * 0.05 + np.random.normal(0, 0.1),
                "grad_norm": np.random.random() * 5,
            }
            for i in range(25)
        ],
        "evaluation": [
            {
                "success_rate": 0.2 + i * 0.1 + np.random.normal(0, 0.05),
                "corruption_score": 0.1 + i * 0.08 + np.random.normal(0, 0.03),
            }
            for i in range(5)
        ],
        "performance": {
            "total_time": 3600,
            "episode_times": [np.random.random() * 10 for _ in range(20)],
            "batch_times": [np.random.random() * 5 for _ in range(20)],
            "avg_episode_time": 5.0,
            "avg_batch_time": 2.5,
            "episodes_per_hour": 12.0,
        },
    }

    # Create results directory
    results_dir = Path("examples/output/demo_results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save mock data
    with open(results_dir / "demo_run_episodes.json", "w") as f:
        json.dump({"episode": mock_data["episodes"]}, f)

    with open(results_dir / "demo_run_training.json", "w") as f:
        json.dump({"training": mock_data["training"]}, f)

    with open(results_dir / "demo_run_evaluation.json", "w") as f:
        json.dump({"evaluation": mock_data["evaluation"]}, f)

    with open(results_dir / "demo_run_performance.json", "w") as f:
        json.dump(mock_data["performance"], f)

    # Create dashboard
    dashboard = SimpleEvaluationDashboard("examples/output/demo_results", "demo_run")

    print("✓ Created evaluation dashboard")
    print("  - Interactive visualizations")
    print("  - Training overview")
    print("  - Memory analysis")
    print("  - Performance analysis")
    print("  - Summary reports")

    # Generate dashboard
    print("\n✓ Generating dashboard...")
    dashboard_dir = dashboard.save_dashboard("examples/output/demo_dashboard")

    print(f"✓ Dashboard saved to: {dashboard_dir}")
    print("  View the PNG files to see the dashboard plots")

    # Generate summary report
    report = dashboard.create_summary_report()
    print("\n✓ Summary report generated:")
    print(f"  Episodes: {report['summary']['episodes']['total']}")
    print(f"  Success rate: {report['summary']['episodes']['success_rate']:.3f}")
    print(f"  Training steps: {report['summary']['training']['total_steps']}")
    print(f"  Evaluations: {report['summary']['evaluation']['total_evals']}")

    return dashboard


def demonstrate_integration():
    """Demonstrate integration of all monitoring and evaluation tools"""
    print("\n=== Integration Demo ===\n")

    print("✓ Demonstrating complete monitoring and evaluation pipeline:")
    print("  1. Real-time metrics tracking during training")
    print("  2. Training progress monitoring and visualization")
    print("  3. Model evaluation with comprehensive metrics")
    print("  4. Interactive dashboard for analysis")

    # Simulate complete pipeline
    print("\n✓ Simulating complete training and evaluation pipeline...")

    # Step 1: Metrics tracking during training
    tracker = MetricsTracker("integration_demo", "examples/output/integration_metrics")

    for epoch in range(10):
        # Simulate training epoch
        for batch in range(5):
            tracker.log_training_step(
                {
                    "loss": 2.0 - epoch * 0.1 + np.random.normal(0, 0.05),
                    "grad_norm": np.random.random() * 3,
                    "batch_time": np.random.random() * 2,
                }
            )

        # Simulate episode
        tracker.log_episode(
            {
                "reward": 0.1 + epoch * 0.05 + np.random.normal(0, 0.1),
                "success": np.random.random() > (0.8 - epoch * 0.05),
                "corruption_score": 0.1 + epoch * 0.08 + np.random.normal(0, 0.05),
                "episode_time": np.random.random() * 8,
            }
        )

        # Simulate evaluation every 3 epochs
        if epoch % 3 == 0:
            tracker.log_evaluation(
                {
                    "success_rate": 0.2 + epoch * 0.08 + np.random.normal(0, 0.05),
                    "corruption_score": 0.1 + epoch * 0.06 + np.random.normal(0, 0.03),
                    "eval_time": np.random.random() * 20,
                }
            )

    # Step 2: Generate monitoring plots
    tracker.plot_training_progress("examples/output/integration_training_progress.png")
    tracker.plot_performance_analysis(
        "examples/output/integration_performance_analysis.png"
    )

    # Step 3: Generate comprehensive report
    report = tracker.generate_report()

    print("\n✓ Integration demo completed:")
    print(f"  Total episodes: {report['summary']['episodes']['total']}")
    print(f"  Success rate: {report['summary']['episodes']['success_rate']:.3f}")
    print(f"  Training steps: {report['summary']['training']['total_steps']}")
    print(f"  Evaluations: {report['summary']['evaluation']['total_evals']}")

    # Stop tracking
    tracker.stop_monitoring()

    print("\n✓ All monitoring and evaluation tools demonstrated successfully!")


def main():
    """Run all demonstrations"""
    print("=== SCATE Monitoring & Evaluation Tools Demo ===\n")

    try:
        # Demonstrate each component
        demonstrate_metrics_tracking()
        demonstrate_training_monitoring()
        demonstrate_model_evaluation()
        demonstrate_evaluation_dashboard()
        demonstrate_integration()

        print("\n=== Demo Summary ===")
        print("✓ Real-time metrics tracking")
        print("✓ Training progress monitoring")
        print("✓ Model evaluation and analysis")
        print("✓ Interactive evaluation dashboard")
        print("✓ Complete integration pipeline")

        print("\n🎯 Key Features Demonstrated:")
        print("  - Real-time metrics collection and visualization")
        print("  - Training progress monitoring with recommendations")
        print("  - Comprehensive model evaluation with pattern analysis")
        print("  - Interactive dashboard for detailed analysis")
        print("  - Complete integration of all monitoring tools")

        print("\n📁 Generated Files:")
        print("  - examples/output/demo_training_progress.png")
        print("  - examples/output/demo_performance_analysis.png")
        print("  - examples/output/demo_memory_analysis.png")
        print("  - examples/output/demo_evaluation_results.png")
        print("  - examples/output/demo_dashboard/")
        print("  - Various JSON reports and metrics files in examples/output/")

        print("\n🚀 Next Steps:")
        print("  1. Use metrics tracking during actual training")
        print("  2. Monitor training progress with real-time visualization")
        print("  3. Evaluate trained models with comprehensive analysis")
        print("  4. Create interactive dashboards for detailed insights")
        print("  5. Integrate all tools into your training pipeline")

    except Exception as e:
        print(f"✗ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
