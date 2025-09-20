"""
Comprehensive metrics tracking for SCATE training.
Provides real-time metrics collection, analysis, and visualization.
"""

import time
import json
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict, deque

try:
    import seaborn as sns

    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
import threading
import queue


class MetricsTracker:
    """Real-time metrics tracking and analysis"""

    def __init__(self, run_name: str, output_dir: str = "metrics_output"):
        """Initialize metrics tracker

        Args:
            run_name: Name of the training run
            output_dir: Directory to save metrics
        """
        self.run_name = run_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Metrics storage
        self.metrics = defaultdict(list)
        self.episode_metrics = defaultdict(list)
        self.training_metrics = defaultdict(list)
        self.evaluation_metrics = defaultdict(list)

        # Real-time tracking
        self.realtime_queue = queue.Queue()
        self.is_monitoring = False
        self.monitor_thread = None

        # Performance tracking
        self.performance_stats = {
            "start_time": time.time(),
            "episode_times": deque(maxlen=100),
            "batch_times": deque(maxlen=100),
            "evaluation_times": deque(maxlen=10),
        }

        # Initialize tracking
        self._initialize_tracking()

    def _initialize_tracking(self):
        """Initialize tracking systems"""
        # Create metrics files
        self.metrics_file = self.output_dir / f"{self.run_name}_metrics.json"
        self.episode_file = self.output_dir / f"{self.run_name}_episodes.json"
        self.performance_file = self.output_dir / f"{self.run_name}_performance.json"

        # Initialize real-time monitoring
        self._start_realtime_monitoring()

    def _start_realtime_monitoring(self):
        """Start real-time monitoring thread"""
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_realtime)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def _monitor_realtime(self):
        """Real-time monitoring loop"""
        while self.is_monitoring:
            try:
                # Process real-time metrics
                while not self.realtime_queue.empty():
                    metric_data = self.realtime_queue.get_nowait()
                    self._process_realtime_metric(metric_data)

                time.sleep(0.1)  # 10Hz monitoring
            except Exception as e:
                print(f"Error in real-time monitoring: {e}")

    def _process_realtime_metric(self, metric_data: Dict[str, Any]):
        """Process real-time metric data"""
        metric_type = metric_data.get("type", "unknown")
        timestamp = metric_data.get("timestamp", time.time())
        data = metric_data.get("data", {})

        # Store metric
        self.metrics[metric_type].append({"timestamp": timestamp, "data": data})

        # Update specific metric collections
        if metric_type == "episode":
            self.episode_metrics[metric_type].append(data)
        elif metric_type == "training":
            self.training_metrics[metric_type].append(data)
        elif metric_type == "evaluation":
            self.evaluation_metrics[metric_type].append(data)

    def log_episode(self, episode_data: Dict[str, Any]):
        """Log episode metrics

        Args:
            episode_data: Episode metrics including rewards, success, etc.
        """
        episode_data["timestamp"] = time.time()
        episode_data["episode_id"] = len(self.episode_metrics["episode"])

        # Add to real-time queue
        self.realtime_queue.put(
            {"type": "episode", "timestamp": time.time(), "data": episode_data}
        )

        # Track episode time
        if "episode_time" in episode_data:
            self.performance_stats["episode_times"].append(episode_data["episode_time"])

    def log_training_step(self, step_data: Dict[str, Any]):
        """Log training step metrics

        Args:
            step_data: Training step metrics including loss, gradients, etc.
        """
        step_data["timestamp"] = time.time()
        step_data["step_id"] = len(self.training_metrics["training"])

        # Add to real-time queue
        self.realtime_queue.put(
            {"type": "training", "timestamp": time.time(), "data": step_data}
        )

        # Track batch time
        if "batch_time" in step_data:
            self.performance_stats["batch_times"].append(step_data["batch_time"])

    def log_evaluation(self, eval_data: Dict[str, Any]):
        """Log evaluation metrics

        Args:
            eval_data: Evaluation metrics including success rate, corruption score, etc.
        """
        eval_data["timestamp"] = time.time()
        eval_data["eval_id"] = len(self.evaluation_metrics["evaluation"])

        # Add to real-time queue
        self.realtime_queue.put(
            {"type": "evaluation", "timestamp": time.time(), "data": eval_data}
        )

        # Track evaluation time
        if "eval_time" in eval_data:
            self.performance_stats["evaluation_times"].append(eval_data["eval_time"])

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics summary

        Returns:
            Dictionary containing current metrics
        """
        current_metrics = {}

        # Episode metrics
        if self.episode_metrics["episode"]:
            episodes = self.episode_metrics["episode"]
            current_metrics["episodes"] = {
                "total": len(episodes),
                "avg_reward": np.mean([e.get("reward", 0) for e in episodes]),
                "avg_success": np.mean([e.get("success", 0) for e in episodes]),
                "recent_reward": np.mean([e.get("reward", 0) for e in episodes[-10:]]),
                "recent_success": np.mean(
                    [e.get("success", 0) for e in episodes[-10:]]
                ),
            }

        # Training metrics
        if self.training_metrics["training"]:
            training = self.training_metrics["training"]
            current_metrics["training"] = {
                "total_steps": len(training),
                "avg_loss": np.mean([t.get("loss", 0) for t in training]),
                "recent_loss": np.mean([t.get("loss", 0) for t in training[-10:]]),
                "avg_grad_norm": np.mean([t.get("grad_norm", 0) for t in training]),
            }

        # Evaluation metrics
        if self.evaluation_metrics["evaluation"]:
            evaluations = self.evaluation_metrics["evaluation"]
            current_metrics["evaluation"] = {
                "total_evals": len(evaluations),
                "avg_success_rate": np.mean(
                    [e.get("success_rate", 0) for e in evaluations]
                ),
                "avg_corruption_score": np.mean(
                    [e.get("corruption_score", 0) for e in evaluations]
                ),
                "latest_success_rate": evaluations[-1].get("success_rate", 0)
                if evaluations
                else 0,
            }

        # Performance metrics
        current_metrics["performance"] = self._get_performance_summary()

        return current_metrics

    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary

        Returns:
            Performance statistics
        """
        total_time = time.time() - self.performance_stats["start_time"]

        performance = {
            "total_time": total_time,
            "avg_episode_time": np.mean(self.performance_stats["episode_times"])
            if self.performance_stats["episode_times"]
            else 0,
            "avg_batch_time": np.mean(self.performance_stats["batch_times"])
            if self.performance_stats["batch_times"]
            else 0,
            "avg_eval_time": np.mean(self.performance_stats["evaluation_times"])
            if self.performance_stats["evaluation_times"]
            else 0,
            "episodes_per_hour": len(self.episode_metrics["episode"])
            / (total_time / 3600)
            if total_time > 0
            else 0,
        }

        return performance

    def save_metrics(self):
        """Save all metrics to files"""
        # Save main metrics
        with open(self.metrics_file, "w") as f:
            json.dump(dict(self.metrics), f, indent=2)

        # Save episode metrics
        with open(self.episode_file, "w") as f:
            json.dump(dict(self.episode_metrics), f, indent=2)

        # Save performance stats (convert deque to list)
        performance_stats_serializable = {
            "start_time": self.performance_stats["start_time"],
            "episode_times": list(self.performance_stats["episode_times"]),
            "batch_times": list(self.performance_stats["batch_times"]),
            "evaluation_times": list(self.performance_stats["evaluation_times"]),
        }

        with open(self.performance_file, "w") as f:
            json.dump(performance_stats_serializable, f, indent=2)

        print(f"Metrics saved to {self.output_dir}")

    def plot_training_progress(self, output_path: Optional[str] = None) -> None:
        """Plot training progress

        Args:
            output_path: Path to save plot
        """
        if not self.episode_metrics["episode"]:
            print("No episode metrics available for plotting")
            return

        episodes = self.episode_metrics["episode"]

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Training Progress - {self.run_name}", fontsize=16)

        # Episode rewards
        ax = axes[0, 0]
        rewards = [e.get("reward", 0) for e in episodes]
        ax.plot(rewards, "b-", alpha=0.7)
        ax.set_title("Episode Rewards")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.grid(True, alpha=0.3)

        # Success rate
        ax = axes[0, 1]
        success_rates = [e.get("success", 0) for e in episodes]
        ax.plot(success_rates, "g-", alpha=0.7)
        ax.set_title("Success Rate")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Success Rate")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        # Training loss
        if self.training_metrics["training"]:
            ax = axes[1, 0]
            training = self.training_metrics["training"]
            losses = [t.get("loss", 0) for t in training]
            ax.plot(losses, "r-", alpha=0.7)
            ax.set_title("Training Loss")
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)

        # Evaluation metrics
        if self.evaluation_metrics["evaluation"]:
            ax = axes[1, 1]
            evaluations = self.evaluation_metrics["evaluation"]
            success_rates = [e.get("success_rate", 0) for e in evaluations]
            corruption_scores = [e.get("corruption_score", 0) for e in evaluations]

            ax.plot(success_rates, "b-", label="Success Rate", alpha=0.7)
            ax.plot(corruption_scores, "r-", label="Corruption Score", alpha=0.7)
            ax.set_title("Evaluation Metrics")
            ax.set_xlabel("Evaluation")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Training progress plot saved to: {output_path}")
        else:
            plt.show()

    def plot_performance_analysis(self, output_path: Optional[str] = None) -> None:
        """Plot performance analysis

        Args:
            output_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Performance Analysis - {self.run_name}", fontsize=16)

        # Episode times
        ax = axes[0, 0]
        episode_times = list(self.performance_stats["episode_times"])
        if episode_times:
            ax.plot(episode_times, "b-", alpha=0.7)
            ax.set_title("Episode Times")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Time (seconds)")
            ax.grid(True, alpha=0.3)

        # Batch times
        ax = axes[0, 1]
        batch_times = list(self.performance_stats["batch_times"])
        if batch_times:
            ax.plot(batch_times, "g-", alpha=0.7)
            ax.set_title("Batch Times")
            ax.set_xlabel("Batch")
            ax.set_ylabel("Time (seconds)")
            ax.grid(True, alpha=0.3)

        # Performance summary
        ax = axes[1, 0]
        performance = self._get_performance_summary()
        metrics = ["Total Time", "Avg Episode", "Avg Batch", "Episodes/Hour"]
        values = [
            performance["total_time"] / 3600,  # Convert to hours
            performance["avg_episode_time"],
            performance["avg_batch_time"],
            performance["episodes_per_hour"],
        ]

        bars = ax.bar(metrics, values, color=["blue", "green", "orange", "purple"])
        ax.set_title("Performance Summary")
        ax.set_ylabel("Value")

        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )

        # Memory usage (if available)
        ax = axes[1, 1]
        ax.text(
            0.5,
            0.5,
            "Memory usage tracking\nnot implemented",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Memory Usage")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Performance analysis plot saved to: {output_path}")
        else:
            plt.show()

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive metrics report

        Returns:
            Comprehensive metrics report
        """
        current_metrics = self.get_current_metrics()

        report = {
            "run_name": self.run_name,
            "timestamp": datetime.now().isoformat(),
            "current_metrics": current_metrics,
            "performance_stats": self.performance_stats,
            "summary": self._generate_summary(),
        }

        return report

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics

        Returns:
            Summary statistics
        """
        summary = {}

        # Episode summary
        if self.episode_metrics["episode"]:
            episodes = self.episode_metrics["episode"]
            rewards = [e.get("reward", 0) for e in episodes]
            successes = [e.get("success", 0) for e in episodes]

            summary["episodes"] = {
                "total": len(episodes),
                "avg_reward": np.mean(rewards),
                "std_reward": np.std(rewards),
                "success_rate": np.mean(successes),
                "best_reward": np.max(rewards),
                "worst_reward": np.min(rewards),
            }

        # Training summary
        if self.training_metrics["training"]:
            training = self.training_metrics["training"]
            losses = [t.get("loss", 0) for t in training]

            summary["training"] = {
                "total_steps": len(training),
                "avg_loss": np.mean(losses),
                "std_loss": np.std(losses),
                "min_loss": np.min(losses),
                "max_loss": np.max(losses),
            }

        # Evaluation summary
        if self.evaluation_metrics["evaluation"]:
            evaluations = self.evaluation_metrics["evaluation"]
            success_rates = [e.get("success_rate", 0) for e in evaluations]
            corruption_scores = [e.get("corruption_score", 0) for e in evaluations]

            summary["evaluation"] = {
                "total_evals": len(evaluations),
                "avg_success_rate": np.mean(success_rates),
                "avg_corruption_score": np.mean(corruption_scores),
                "best_success_rate": np.max(success_rates),
                "latest_success_rate": success_rates[-1] if success_rates else 0,
            }

        return summary

    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

        # Save final metrics
        self.save_metrics()
        print("Monitoring stopped and metrics saved")


def create_metrics_tracker(
    run_name: str, output_dir: str = "metrics_output"
) -> MetricsTracker:
    """Create a metrics tracker instance

    Args:
        run_name: Name of the training run
        output_dir: Directory to save metrics

    Returns:
        MetricsTracker instance
    """
    return MetricsTracker(run_name, output_dir)


# Example usage
if __name__ == "__main__":
    # Create tracker
    tracker = create_metrics_tracker("test_run")

    # Simulate some metrics
    for i in range(10):
        tracker.log_episode(
            {
                "reward": np.random.random(),
                "success": np.random.random() > 0.7,
                "episode_time": np.random.random() * 10,
            }
        )

        tracker.log_training_step(
            {
                "loss": np.random.random() * 2,
                "grad_norm": np.random.random(),
                "batch_time": np.random.random() * 5,
            }
        )

    # Generate plots and report
    tracker.plot_training_progress("training_progress.png")
    tracker.plot_performance_analysis("performance_analysis.png")

    report = tracker.generate_report()
    print("Generated report:", json.dumps(report, indent=2))

    # Stop monitoring
    tracker.stop_monitoring()
