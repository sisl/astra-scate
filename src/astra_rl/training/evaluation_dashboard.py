#!/usr/bin/env python3
"""
Evaluation dashboard for SCATE training.
Provides interactive visualization and analysis of training results.
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd

try:
    import seaborn as sns

    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available. Install with: pip install plotly")

# Add src to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


class EvaluationDashboard:
    """Interactive evaluation dashboard for SCATE training"""

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

    def create_training_overview(self):
        """Create training overview dashboard

        Returns:
            Plotly figure with training overview
        """
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Episode Rewards",
                "Success Rate",
                "Training Loss",
                "Evaluation Metrics",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        # Episode rewards
        if self.data["episodes"]:
            episodes = self.data["episodes"]
            rewards = [e.get("reward", 0) for e in episodes]
            episode_ids = list(range(len(episodes)))

            fig.add_trace(
                go.Scatter(x=episode_ids, y=rewards, mode="lines", name="Reward"),
                row=1,
                col=1,
            )

            # Add trend line
            if len(rewards) > 1:
                z = np.polyfit(episode_ids, rewards, 1)
                p = np.poly1d(z)
                trend = p(episode_ids)
                fig.add_trace(
                    go.Scatter(
                        x=episode_ids,
                        y=trend,
                        mode="lines",
                        name="Trend",
                        line=dict(dash="dash"),
                    ),
                    row=1,
                    col=1,
                )

        # Success rate
        if self.data["episodes"]:
            successes = [e.get("success", 0) for e in episodes]
            success_rate = np.cumsum(successes) / (np.arange(len(successes)) + 1)

            fig.add_trace(
                go.Scatter(
                    x=episode_ids, y=success_rate, mode="lines", name="Success Rate"
                ),
                row=1,
                col=2,
            )

        # Training loss
        if self.data["training"]:
            training = self.data["training"]
            losses = [t.get("loss", 0) for t in training]
            step_ids = list(range(len(training)))

            fig.add_trace(
                go.Scatter(x=step_ids, y=losses, mode="lines", name="Loss"),
                row=2,
                col=1,
            )

        # Evaluation metrics
        if self.data["evaluation"]:
            evaluations = self.data["evaluation"]
            success_rates = [e.get("success_rate", 0) for e in evaluations]
            corruption_scores = [e.get("corruption_score", 0) for e in evaluations]
            eval_ids = list(range(len(evaluations)))

            fig.add_trace(
                go.Scatter(
                    x=eval_ids,
                    y=success_rates,
                    mode="lines+markers",
                    name="Success Rate",
                ),
                row=2,
                col=2,
            )
            fig.add_trace(
                go.Scatter(
                    x=eval_ids,
                    y=corruption_scores,
                    mode="lines+markers",
                    name="Corruption Score",
                ),
                row=2,
                col=2,
            )

        # Update layout
        fig.update_layout(
            title=f"Training Overview - {self.run_name}", height=800, showlegend=True
        )

        return fig

    def create_memory_analysis(self) -> go.Figure:
        """Create memory analysis dashboard

        Returns:
            Plotly figure with memory analysis
        """
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Memory Corruption Over Time",
                "Attack Pattern Distribution",
                "Injection Success Rate",
                "Memory Persistence",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        # Memory corruption over time
        if self.data["episodes"]:
            episodes = self.data["episodes"]
            corruption_scores = [e.get("corruption_score", 0) for e in episodes]
            episode_ids = list(range(len(episodes)))

            fig.add_trace(
                go.Scatter(
                    x=episode_ids,
                    y=corruption_scores,
                    mode="lines",
                    name="Corruption Score",
                ),
                row=1,
                col=1,
            )

        # Attack pattern distribution
        if self.data["episodes"]:
            patterns = [
                e.get("pattern", "unknown") for e in episodes if e.get("success", False)
            ]
            if patterns:
                pattern_counts = pd.Series(patterns).value_counts()

                fig.add_trace(
                    go.Bar(
                        x=pattern_counts.index,
                        y=pattern_counts.values,
                        name="Pattern Count",
                    ),
                    row=1,
                    col=2,
                )

        # Injection success rate
        if self.data["episodes"]:
            injection_success = [e.get("injection_success", 0) for e in episodes]
            injection_rate = np.cumsum(injection_success) / (
                np.arange(len(injection_success)) + 1
            )

            fig.add_trace(
                go.Scatter(
                    x=episode_ids,
                    y=injection_rate,
                    mode="lines",
                    name="Injection Success",
                ),
                row=2,
                col=1,
            )

        # Memory persistence
        if self.data["episodes"]:
            persistence_scores = [e.get("persistence_score", 0) for e in episodes]

            fig.add_trace(
                go.Scatter(
                    x=episode_ids,
                    y=persistence_scores,
                    mode="lines",
                    name="Persistence",
                ),
                row=2,
                col=2,
            )

        # Update layout
        fig.update_layout(
            title=f"Memory Analysis - {self.run_name}", height=800, showlegend=True
        )

        return fig

    def create_performance_analysis(self) -> go.Figure:
        """Create performance analysis dashboard

        Returns:
            Plotly figure with performance analysis
        """
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Episode Times",
                "Batch Times",
                "Performance Summary",
                "Efficiency Metrics",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        # Episode times
        if self.data["performance"].get("episode_times"):
            episode_times = self.data["performance"]["episode_times"]
            episode_ids = list(range(len(episode_times)))

            fig.add_trace(
                go.Scatter(
                    x=episode_ids, y=episode_times, mode="lines", name="Episode Time"
                ),
                row=1,
                col=1,
            )

        # Batch times
        if self.data["performance"].get("batch_times"):
            batch_times = self.data["performance"]["batch_times"]
            batch_ids = list(range(len(batch_times)))

            fig.add_trace(
                go.Scatter(x=batch_ids, y=batch_times, mode="lines", name="Batch Time"),
                row=1,
                col=2,
            )

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
                perf.get("total_time", 0) / 3600,
                perf.get("avg_episode_time", 0),
                perf.get("avg_batch_time", 0),
                perf.get("episodes_per_hour", 0),
            ]

            fig.add_trace(go.Bar(x=metrics, y=values, name="Performance"), row=2, col=1)

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

            fig.add_trace(
                go.Bar(x=efficiency_metrics, y=efficiency_values, name="Efficiency"),
                row=2,
                col=2,
            )

        # Update layout
        fig.update_layout(
            title=f"Performance Analysis - {self.run_name}", height=800, showlegend=True
        )

        return fig

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

    def save_dashboard(self, output_path: Optional[str] = None) -> str:
        """Save interactive dashboard

        Args:
            output_path: Path to save dashboard

        Returns:
            Path to saved dashboard
        """
        if output_path is None:
            output_path = self.output_dir / f"{self.run_name}_dashboard.html"

        # Create dashboard
        dashboard_html = self._create_dashboard_html()

        # Save to file
        with open(output_path, "w") as f:
            f.write(dashboard_html)

        print(f"Dashboard saved to: {output_path}")
        return str(output_path)

    def _create_dashboard_html(self) -> str:
        """Create HTML dashboard

        Returns:
            HTML content
        """
        # Create figures
        overview_fig = self.create_training_overview()
        memory_fig = self.create_memory_analysis()
        performance_fig = self.create_performance_analysis()

        # Generate report
        report = self.create_summary_report()

        # Create HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SCATE Training Dashboard - {self.run_name}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .section {{ margin-bottom: 40px; }}
                .summary {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; }}
                .recommendations {{ background-color: #e8f4fd; padding: 20px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>SCATE Training Dashboard</h1>
                <h2>{self.run_name}</h2>
                <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <div class="section">
                <h3>Training Overview</h3>
                <div id="overview"></div>
            </div>
            
            <div class="section">
                <h3>Memory Analysis</h3>
                <div id="memory"></div>
            </div>
            
            <div class="section">
                <h3>Performance Analysis</h3>
                <div id="performance"></div>
            </div>
            
            <div class="section">
                <h3>Summary Report</h3>
                <div class="summary">
                    <pre>{json.dumps(report, indent=2)}</pre>
                </div>
            </div>
            
            <script>
                // Render plots
                Plotly.newPlot('overview', {overview_fig.to_json()});
                Plotly.newPlot('memory', {memory_fig.to_json()});
                Plotly.newPlot('performance', {performance_fig.to_json()});
            </script>
        </body>
        </html>
        """

        return html_content


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
        "--output_path", type=str, default=None, help="Path to save dashboard HTML"
    )

    args = parser.parse_args()

    # Create dashboard
    dashboard = EvaluationDashboard(args.results_dir, args.run_name)

    # Save dashboard
    output_path = dashboard.save_dashboard(args.output_path)

    print("Dashboard created successfully!")
    print(f"Open {output_path} in your browser to view the dashboard.")


if __name__ == "__main__":
    main()
