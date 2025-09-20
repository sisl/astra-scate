#!/usr/bin/env python3
"""
Simple evaluation dashboard for SCATE training.
Provides basic visualization and analysis of training results using matplotlib.
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add src to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


class SimpleEvaluationDashboard:
    """Simple evaluation dashboard for SCATE training"""

    def __init__(self, results_dir: str, run_name: str):
        """Initialize evaluation dashboard

        Args:
            results_dir: Directory containing evaluation results
            run_name: Name of the training run
        """
        self.results_dir = Path(results_dir)
        self.run_name = run_name
        self.output_dir = self.results_dir / "dashboard"
        self.output_dir.mkdir(exist_ok=True)

        # Load data
        self.data = self._load_evaluation_data()

    def _load_evaluation_data(self) -> Dict[str, Any]:
        """Load evaluation data from files

        Returns:
            Dictionary containing evaluation data
        """
        data = {
            "metrics": {},
            "episodes": [],
            "training": [],
            "evaluation": [],
            "performance": {},
        }

        # Load metrics file
        metrics_file = self.results_dir / f"{self.run_name}_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                data["metrics"] = json.load(f)

        # Load episode data
        episode_file = self.results_dir / f"{self.run_name}_episodes.json"
        if episode_file.exists():
            with open(episode_file, "r") as f:
                episode_data = json.load(f)
                data["episodes"] = episode_data.get("episode", [])

        # Load training data
        training_file = self.results_dir / f"{self.run_name}_training.json"
        if training_file.exists():
            with open(training_file, "r") as f:
                training_data = json.load(f)
                data["training"] = training_data.get("training", [])

        # Load evaluation data
        eval_file = self.results_dir / f"{self.run_name}_evaluation.json"
        if eval_file.exists():
            with open(eval_file, "r") as f:
                eval_data = json.load(f)
                data["evaluation"] = eval_data.get("evaluation", [])

        # Load performance data
        perf_file = self.results_dir / f"{self.run_name}_performance.json"
        if perf_file.exists():
            with open(perf_file, "r") as f:
                data["performance"] = json.load(f)

        return data

    def create_training_overview(self, output_path: Optional[str] = None) -> None:
        """Create training overview dashboard

        Args:
            output_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Training Overview - {self.run_name}", fontsize=16)

        # Episode rewards
        if self.data["episodes"]:
            episodes = self.data["episodes"]
            rewards = [e.get("reward", 0) for e in episodes]
            episode_ids = list(range(len(episodes)))

            axes[0, 0].plot(episode_ids, rewards, "b-", alpha=0.7, linewidth=2)
            axes[0, 0].set_title("Episode Rewards")
            axes[0, 0].set_xlabel("Episode")
            axes[0, 0].set_ylabel("Reward")
            axes[0, 0].grid(True, alpha=0.3)

            # Add trend line
            if len(rewards) > 1:
                z = np.polyfit(episode_ids, rewards, 1)
                p = np.poly1d(z)
                trend = p(episode_ids)
                axes[0, 0].plot(episode_ids, trend, "r--", alpha=0.7, label="Trend")
                axes[0, 0].legend()

        # Success rate
        if self.data["episodes"]:
            successes = [e.get("success", 0) for e in episodes]
            success_rate = np.cumsum(successes) / (np.arange(len(successes)) + 1)

            axes[0, 1].plot(episode_ids, success_rate, "g-", alpha=0.7, linewidth=2)
            axes[0, 1].set_title("Success Rate")
            axes[0, 1].set_xlabel("Episode")
            axes[0, 1].set_ylabel("Success Rate")
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].grid(True, alpha=0.3)

        # Training loss
        if self.data["training"]:
            training = self.data["training"]
            losses = [t.get("loss", 0) for t in training]
            step_ids = list(range(len(training)))

            axes[1, 0].plot(step_ids, losses, "r-", alpha=0.7, linewidth=2)
            axes[1, 0].set_title("Training Loss")
            axes[1, 0].set_xlabel("Step")
            axes[1, 0].set_ylabel("Loss")
            axes[1, 0].grid(True, alpha=0.3)

        # Evaluation metrics
        if self.data["evaluation"]:
            evaluations = self.data["evaluation"]
            success_rates = [e.get("success_rate", 0) for e in evaluations]
            corruption_scores = [e.get("corruption_score", 0) for e in evaluations]
            eval_ids = list(range(len(evaluations)))

            axes[1, 1].plot(
                eval_ids, success_rates, "b-o", label="Success Rate", alpha=0.7
            )
            axes[1, 1].plot(
                eval_ids, corruption_scores, "r-s", label="Corruption Score", alpha=0.7
            )
            axes[1, 1].set_title("Evaluation Metrics")
            axes[1, 1].set_xlabel("Evaluation")
            axes[1, 1].set_ylabel("Value")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "No evaluation metrics available",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )
            axes[1, 1].set_title("Evaluation Metrics")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Training overview saved to: {output_path}")
        else:
            plt.show()

    def create_memory_analysis(self, output_path: Optional[str] = None) -> None:
        """Create memory analysis dashboard

        Args:
            output_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Memory Analysis - {self.run_name}", fontsize=16)

        # Memory corruption over time
        if self.data["episodes"]:
            episodes = self.data["episodes"]
            corruption_scores = [e.get("corruption_score", 0) for e in episodes]
            episode_ids = list(range(len(episodes)))

            axes[0, 0].plot(
                episode_ids, corruption_scores, "r-", alpha=0.7, linewidth=2
            )
            axes[0, 0].set_title("Memory Corruption Over Time")
            axes[0, 0].set_xlabel("Episode")
            axes[0, 0].set_ylabel("Corruption Score")
            axes[0, 0].set_ylim(0, 1)
            axes[0, 0].grid(True, alpha=0.3)

        # Attack pattern distribution
        if self.data["episodes"]:
            patterns = [
                e.get("pattern", "unknown") for e in episodes if e.get("success", False)
            ]
            if patterns:
                pattern_counts = pd.Series(patterns).value_counts()

                axes[0, 1].bar(
                    range(len(pattern_counts)),
                    pattern_counts.values,
                    color="skyblue",
                    alpha=0.7,
                )
                axes[0, 1].set_title("Attack Pattern Distribution")
                axes[0, 1].set_xlabel("Pattern")
                axes[0, 1].set_ylabel("Count")
                axes[0, 1].set_xticks(range(len(pattern_counts)))
                axes[0, 1].set_xticklabels(pattern_counts.index, rotation=45)
                axes[0, 1].grid(True, alpha=0.3)
            else:
                axes[0, 1].text(
                    0.5,
                    0.5,
                    "No successful attacks found",
                    ha="center",
                    va="center",
                    transform=axes[0, 1].transAxes,
                )
                axes[0, 1].set_title("Attack Pattern Distribution")

        # Injection success rate
        if self.data["episodes"]:
            injection_success = [e.get("injection_success", 0) for e in episodes]
            injection_rate = np.cumsum(injection_success) / (
                np.arange(len(injection_success)) + 1
            )

            axes[1, 0].plot(episode_ids, injection_rate, "g-", alpha=0.7, linewidth=2)
            axes[1, 0].set_title("Injection Success Rate")
            axes[1, 0].set_xlabel("Episode")
            axes[1, 0].set_ylabel("Injection Success Rate")
            axes[1, 0].set_ylim(0, 1)
            axes[1, 0].grid(True, alpha=0.3)

        # Memory persistence
        if self.data["episodes"]:
            persistence_scores = [e.get("persistence_score", 0) for e in episodes]

            axes[1, 1].plot(
                episode_ids, persistence_scores, "purple", alpha=0.7, linewidth=2
            )
            axes[1, 1].set_title("Memory Persistence")
            axes[1, 1].set_xlabel("Episode")
            axes[1, 1].set_ylabel("Persistence Score")
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Memory analysis saved to: {output_path}")
        else:
            plt.show()

    def create_performance_analysis(self, output_path: Optional[str] = None) -> None:
        """Create performance analysis dashboard

        Args:
            output_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Performance Analysis - {self.run_name}", fontsize=16)

        # Episode times
        if self.data["performance"].get("episode_times"):
            episode_times = self.data["performance"]["episode_times"]
            episode_ids = list(range(len(episode_times)))

            axes[0, 0].plot(episode_ids, episode_times, "b-", alpha=0.7, linewidth=2)
            axes[0, 0].set_title("Episode Times")
            axes[0, 0].set_xlabel("Episode")
            axes[0, 0].set_ylabel("Time (seconds)")
            axes[0, 0].grid(True, alpha=0.3)

        # Batch times
        if self.data["performance"].get("batch_times"):
            batch_times = self.data["performance"]["batch_times"]
            batch_ids = list(range(len(batch_times)))

            axes[0, 1].plot(batch_ids, batch_times, "g-", alpha=0.7, linewidth=2)
            axes[0, 1].set_title("Batch Times")
            axes[0, 1].set_xlabel("Batch")
            axes[0, 1].set_ylabel("Time (seconds)")
            axes[0, 1].grid(True, alpha=0.3)

        # Performance summary
        if self.data["performance"]:
            perf = self.data["performance"]
            metrics = [
                "Total Time (h)",
                "Avg Episode (s)",
                "Avg Batch (s)",
                "Episodes/Hour",
            ]
            values = [
                perf.get("total_time", 0) / 3600,  # Convert to hours
                perf.get("avg_episode_time", 0),
                perf.get("avg_batch_time", 0),
                perf.get("episodes_per_hour", 0),
            ]

            bars = axes[1, 0].bar(
                metrics, values, color=["blue", "green", "orange", "purple"], alpha=0.7
            )
            axes[1, 0].set_title("Performance Summary")
            axes[1, 0].set_ylabel("Value")
            axes[1, 0].tick_params(axis="x", rotation=45)

            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[1, 0].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                )

        # Efficiency metrics
        if self.data["episodes"] and self.data["training"]:
            episodes = self.data["episodes"]
            training = self.data["training"]

            # Calculate efficiency metrics
            total_episodes = len(episodes)
            total_steps = len(training)
            success_rate = np.mean([e.get("success", 0) for e in episodes])
            avg_reward = np.mean([e.get("reward", 0) for e in episodes])

            efficiency_metrics = ["Episodes", "Steps", "Success Rate", "Avg Reward"]
            efficiency_values = [total_episodes, total_steps, success_rate, avg_reward]

            bars = axes[1, 1].bar(
                efficiency_metrics,
                efficiency_values,
                color=["blue", "green", "orange", "purple"],
                alpha=0.7,
            )
            axes[1, 1].set_title("Efficiency Metrics")
            axes[1, 1].set_ylabel("Value")
            axes[1, 1].tick_params(axis="x", rotation=45)

            # Add value labels
            for bar, value in zip(bars, efficiency_values):
                height = bar.get_height()
                axes[1, 1].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Performance analysis saved to: {output_path}")
        else:
            plt.show()

    def create_summary_report(self) -> Dict[str, Any]:
        """Create summary report

        Returns:
            Summary report dictionary
        """
        report = {
            "run_name": self.run_name,
            "timestamp": datetime.now().isoformat(),
            "summary": {},
            "recommendations": [],
        }

        # Episode summary
        if self.data["episodes"]:
            episodes = self.data["episodes"]
            rewards = [e.get("reward", 0) for e in episodes]
            successes = [e.get("success", 0) for e in episodes]

            report["summary"]["episodes"] = {
                "total": len(episodes),
                "avg_reward": np.mean(rewards),
                "std_reward": np.std(rewards),
                "success_rate": np.mean(successes),
                "best_reward": np.max(rewards),
                "worst_reward": np.min(rewards),
            }

        # Training summary
        if self.data["training"]:
            training = self.data["training"]
            losses = [t.get("loss", 0) for t in training]

            report["summary"]["training"] = {
                "total_steps": len(training),
                "avg_loss": np.mean(losses),
                "std_loss": np.std(losses),
                "min_loss": np.min(losses),
                "max_loss": np.max(losses),
            }

        # Evaluation summary
        if self.data["evaluation"]:
            evaluations = self.data["evaluation"]
            success_rates = [e.get("success_rate", 0) for e in evaluations]
            corruption_scores = [e.get("corruption_score", 0) for e in evaluations]

            report["summary"]["evaluation"] = {
                "total_evals": len(evaluations),
                "avg_success_rate": np.mean(success_rates),
                "avg_corruption_score": np.mean(corruption_scores),
                "best_success_rate": np.max(success_rates),
                "latest_success_rate": success_rates[-1] if success_rates else 0,
            }

        # Generate recommendations
        if "episodes" in report["summary"]:
            episodes = report["summary"]["episodes"]
            if episodes["success_rate"] < 0.1:
                report["recommendations"].append(
                    "Very low success rate. Consider retraining with different hyperparameters."
                )
            elif episodes["success_rate"] < 0.3:
                report["recommendations"].append(
                    "Low success rate. Consider increasing training epochs or adjusting learning rate."
                )
            elif episodes["success_rate"] > 0.7:
                report["recommendations"].append(
                    "High success rate! Model is performing well."
                )

        if "training" in report["summary"]:
            training = report["summary"]["training"]
            if training["avg_loss"] > 2.0:
                report["recommendations"].append(
                    "High average loss. Consider reducing learning rate or increasing model capacity."
                )

        return report

    def save_dashboard(self, output_dir: Optional[str] = None) -> str:
        """Save dashboard plots

        Args:
            output_dir: Directory to save plots

        Returns:
            Directory where plots were saved
        """
        if output_dir is None:
            output_dir = self.output_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)

        # Create plots
        self.create_training_overview(
            str(output_dir / f"{self.run_name}_training_overview.png")
        )
        self.create_memory_analysis(
            str(output_dir / f"{self.run_name}_memory_analysis.png")
        )
        self.create_performance_analysis(
            str(output_dir / f"{self.run_name}_performance_analysis.png")
        )

        # Generate report
        report = self.create_summary_report()
        report_path = output_dir / f"{self.run_name}_summary_report.json"

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Dashboard saved to: {output_dir}")
        return str(output_dir)


def main():
    """Main dashboard function"""
    parser = argparse.ArgumentParser(description="Create SCATE evaluation dashboard")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing evaluation results",
    )
    parser.add_argument(
        "--run_name", type=str, required=True, help="Name of the training run"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Directory to save dashboard plots"
    )

    args = parser.parse_args()

    # Create dashboard
    dashboard = SimpleEvaluationDashboard(args.results_dir, args.run_name)

    # Save dashboard
    output_dir = dashboard.save_dashboard(args.output_dir)

    print("Dashboard created successfully!")
    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
