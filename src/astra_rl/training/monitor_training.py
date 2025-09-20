#!/usr/bin/env python3
"""
Monitor ongoing training and visualize results.
Provides real-time monitoring and analysis of IPO memory training progress.
"""

import argparse
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import time
from datetime import datetime

# Add src to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    from tensorboard.backend.event_processing import event_accumulator

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: tensorboard not available. Install with: pip install tensorboard")

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")


class TrainingMonitor:
    """Monitor and visualize training progress"""

    def __init__(self, log_dir: str, run_name: str):
        """Initialize training monitor

        Args:
            log_dir: Directory containing training logs
            run_name: Name of the training run
        """
        self.log_dir = Path(log_dir)
        self.run_name = run_name
        self.tensorboard_dir = self.log_dir / "runs" / run_name
        self.results_dir = self.log_dir / "results" / run_name

    def load_tensorboard_logs(self) -> Dict[str, pd.DataFrame]:
        """Load metrics from tensorboard logs"""
        if not TENSORBOARD_AVAILABLE:
            print("Tensorboard not available, cannot load logs")
            return {}

        if not self.tensorboard_dir.exists():
            print(f"Tensorboard directory not found: {self.tensorboard_dir}")
            return {}

        ea = event_accumulator.EventAccumulator(str(self.tensorboard_dir))
        ea.Reload()

        metrics = {}
        for tag in ea.Tags()["scalars"]:
            events = ea.Scalars(tag)
            metrics[tag] = pd.DataFrame(
                [
                    {"step": e.step, "value": e.value, "wall_time": e.wall_time}
                    for e in events
                ]
            )

        return metrics

    def load_wandb_logs(
        self, project: str = "scate-memory"
    ) -> Optional[Dict[str, Any]]:
        """Load metrics from wandb (if available)"""
        if not WANDB_AVAILABLE:
            print("Wandb not available")
            return None

        try:
            # This would require the actual wandb run ID
            # For now, return None as we don't have the run ID
            print("Wandb integration requires run ID - not implemented in this demo")
            return None
        except Exception as e:
            print(f"Error loading wandb logs: {e}")
            return None

    def load_training_logs(self) -> List[Dict[str, Any]]:
        """Load training logs from file"""
        log_file = self.results_dir / f"{self.run_name}.log"
        if not log_file.exists():
            print(f"Log file not found: {log_file}")
            return []

        logs = []
        with open(log_file, "r") as f:
            for line in f:
                if "Epoch" in line and "Loss:" in line:
                    # Parse epoch information
                    try:
                        parts = line.strip().split()
                        epoch = int(parts[1].split("/")[0])
                        loss = float(parts[3])
                        reward = float(parts[5])
                        success = float(parts[7].rstrip("%")) / 100

                        logs.append(
                            {
                                "epoch": epoch,
                                "loss": loss,
                                "reward": reward,
                                "success_rate": success,
                                "timestamp": datetime.now(),
                            }
                        )
                    except (ValueError, IndexError):
                        continue

        return logs

    def plot_training_progress(
        self, metrics: Dict[str, pd.DataFrame], output_path: Optional[str] = None
    ) -> None:
        """Plot training metrics"""
        if not metrics:
            print("No metrics available for plotting")
            return

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Training Progress - {self.run_name}", fontsize=16)

        # Loss plot
        if "Loss/train" in metrics:
            ax = axes[0, 0]
            df = metrics["Loss/train"]
            ax.plot(df["step"], df["value"], "b-", linewidth=2)
            ax.set_title("Training Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)

            # Add trend line
            if len(df) > 1:
                z = np.polyfit(df["step"], df["value"], 1)
                p = np.poly1d(z)
                ax.plot(df["step"], p(df["step"]), "r--", alpha=0.7, label="Trend")
                ax.legend()

        # Reward plot
        if "Reward/train" in metrics:
            ax = axes[0, 1]
            df = metrics["Reward/train"]
            ax.plot(df["step"], df["value"], "g-", linewidth=2)
            ax.set_title("Average Reward")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Reward")
            ax.grid(True, alpha=0.3)

            # Add trend line
            if len(df) > 1:
                z = np.polyfit(df["step"], df["value"], 1)
                p = np.poly1d(z)
                ax.plot(df["step"], p(df["step"]), "r--", alpha=0.7, label="Trend")
                ax.legend()

        # Success rate plot
        if "Success/train" in metrics:
            ax = axes[1, 0]
            df = metrics["Success/train"]
            ax.plot(df["step"], df["value"], "m-", linewidth=2)
            ax.set_title("Attack Success Rate")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Success Rate")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

            # Add trend line
            if len(df) > 1:
                z = np.polyfit(df["step"], df["value"], 1)
                p = np.poly1d(z)
                ax.plot(df["step"], p(df["step"]), "r--", alpha=0.7, label="Trend")
                ax.legend()

        # Evaluation metrics plot
        eval_metrics = [k for k in metrics.keys() if k.startswith("Eval/")]
        if eval_metrics:
            ax = axes[1, 1]
            for metric in eval_metrics:
                df = metrics[metric]
                ax.plot(
                    df["step"],
                    df["value"],
                    label=metric.replace("Eval/", ""),
                    linewidth=2,
                )
            ax.set_title("Evaluation Metrics")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            # If no eval metrics, show memory stats
            ax = axes[1, 1]
            ax.text(
                0.5,
                0.5,
                "No evaluation metrics available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Evaluation Metrics")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {output_path}")
        else:
            plt.show()

    def plot_memory_analysis(
        self, metrics: Dict[str, pd.DataFrame], output_path: Optional[str] = None
    ) -> None:
        """Plot memory-specific analysis"""
        if not metrics:
            print("No metrics available for memory analysis")
            return

        # Create memory-specific plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Memory Analysis - {self.run_name}", fontsize=16)

        # Memory corruption score
        if "Eval/avg_corruption_score" in metrics:
            ax = axes[0, 0]
            df = metrics["Eval/avg_corruption_score"]
            ax.plot(df["step"], df["value"], "r-", linewidth=2, marker="o")
            ax.set_title("Memory Corruption Score")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Corruption Score")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

        # Success rate over time
        if "Eval/success_rate" in metrics:
            ax = axes[0, 1]
            df = metrics["Eval/success_rate"]
            ax.plot(df["step"], df["value"], "b-", linewidth=2, marker="s")
            ax.set_title("Attack Success Rate")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Success Rate")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

        # Reward distribution
        if "Reward/train" in metrics:
            ax = axes[1, 0]
            df = metrics["Reward/train"]
            ax.hist(df["value"], bins=20, alpha=0.7, color="green")
            ax.set_title("Reward Distribution")
            ax.set_xlabel("Reward Value")
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)

        # Training stability (loss variance)
        if "Loss/train" in metrics:
            ax = axes[1, 1]
            df = metrics["Loss/train"]
            # Calculate rolling variance
            window_size = min(10, len(df) // 4)
            if window_size > 1:
                rolling_var = df["value"].rolling(window=window_size).var()
                ax.plot(df["step"], rolling_var, "purple", linewidth=2)
                ax.set_title("Training Stability (Loss Variance)")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss Variance")
                ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Memory analysis plot saved to: {output_path}")
        else:
            plt.show()

    def generate_training_report(
        self, metrics: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        report = {
            "run_name": self.run_name,
            "timestamp": datetime.now().isoformat(),
            "summary": {},
            "metrics": {},
            "recommendations": [],
        }

        # Extract key metrics
        if "Loss/train" in metrics:
            loss_df = metrics["Loss/train"]
            report["summary"]["final_loss"] = float(loss_df["value"].iloc[-1])
            report["summary"]["min_loss"] = float(loss_df["value"].min())
            report["summary"]["loss_improvement"] = float(
                loss_df["value"].iloc[0] - loss_df["value"].iloc[-1]
            )

        if "Success/train" in metrics:
            success_df = metrics["Success/train"]
            report["summary"]["final_success_rate"] = float(
                success_df["value"].iloc[-1]
            )
            report["summary"]["max_success_rate"] = float(success_df["value"].max())

        if "Reward/train" in metrics:
            reward_df = metrics["Reward/train"]
            report["summary"]["final_reward"] = float(reward_df["value"].iloc[-1])
            report["summary"]["avg_reward"] = float(reward_df["value"].mean())

        # Evaluation metrics
        eval_metrics = [k for k in metrics.keys() if k.startswith("Eval/")]
        for metric in eval_metrics:
            df = metrics[metric]
            report["metrics"][metric] = {
                "final": float(df["value"].iloc[-1]),
                "max": float(df["value"].max()),
                "min": float(df["value"].min()),
                "mean": float(df["value"].mean()),
            }

        # Generate recommendations
        if "summary" in report:
            summary = report["summary"]

            if summary.get("final_success_rate", 0) < 0.1:
                report["recommendations"].append(
                    "Low success rate detected. Consider increasing training epochs or adjusting learning rate."
                )

            if summary.get("loss_improvement", 0) < 0.1:
                report["recommendations"].append(
                    "Minimal loss improvement. Consider adjusting hyperparameters or model architecture."
                )

            if summary.get("final_loss", float("inf")) > 2.0:
                report["recommendations"].append(
                    "High final loss. Consider reducing learning rate or increasing model capacity."
                )

        return report

    def save_report(self, report: Dict[str, Any], output_path: str) -> None:
        """Save training report to file"""
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Training report saved to: {output_path}")

    def print_latest_metrics(self, metrics: Dict[str, pd.DataFrame]) -> None:
        """Print latest metrics to console"""
        print(f"\n=== Latest Training Metrics - {self.run_name} ===")

        for name, df in metrics.items():
            if len(df) > 0:
                latest = df.iloc[-1]
                print(f"{name}: {latest['value']:.4f} (step {latest['step']})")

        print("=" * 50)


def main():
    """Main monitoring function"""
    parser = argparse.ArgumentParser(description="Monitor SCATE training progress")
    parser.add_argument(
        "--log_dir", type=str, default=".", help="Directory containing training logs"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Name of the training run to monitor",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="monitoring_output",
        help="Directory to save monitoring outputs",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Generate training progress plots"
    )
    parser.add_argument(
        "--memory_analysis", action="store_true", help="Generate memory analysis plots"
    )
    parser.add_argument(
        "--report", action="store_true", help="Generate training report"
    )
    parser.add_argument(
        "--watch", action="store_true", help="Watch training in real-time"
    )
    parser.add_argument(
        "--interval", type=int, default=30, help="Watch interval in seconds"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Initialize monitor
    monitor = TrainingMonitor(args.log_dir, args.run_name)

    if args.watch:
        # Real-time monitoring
        print(f"Watching training run: {args.run_name}")
        print(f"Refresh interval: {args.interval} seconds")
        print("Press Ctrl+C to stop")

        try:
            while True:
                metrics = monitor.load_tensorboard_logs()
                if metrics:
                    monitor.print_latest_metrics(metrics)
                else:
                    print("No metrics available yet...")

                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped")

    else:
        # One-time analysis
        print(f"Analyzing training run: {args.run_name}")

        # Load metrics
        metrics = monitor.load_tensorboard_logs()

        if not metrics:
            print("No metrics found. Make sure tensorboard logging is enabled.")
            return

        # Print latest metrics
        monitor.print_latest_metrics(metrics)

        # Generate plots
        if args.plot:
            plot_path = output_dir / f"{args.run_name}_training_progress.png"
            monitor.plot_training_progress(metrics, str(plot_path))

        if args.memory_analysis:
            memory_plot_path = output_dir / f"{args.run_name}_memory_analysis.png"
            monitor.plot_memory_analysis(metrics, str(memory_plot_path))

        # Generate report
        if args.report:
            report = monitor.generate_training_report(metrics)
            report_path = output_dir / f"{args.run_name}_report.json"
            monitor.save_report(report, str(report_path))

            # Print recommendations
            if report["recommendations"]:
                print("\n=== Recommendations ===")
                for i, rec in enumerate(report["recommendations"], 1):
                    print(f"{i}. {rec}")


if __name__ == "__main__":
    main()
