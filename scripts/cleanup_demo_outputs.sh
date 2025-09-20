#!/bin/bash
# Cleanup script for demo outputs
# This script organizes demo outputs and cleans up the main directory

echo "🧹 Cleaning up demo outputs..."

# Create organized output directories
mkdir -p examples/output/{checkpoints,results,logs,demos}

# Move any remaining demo files to the organized structure
echo "📁 Moving demo outputs to organized structure..."

# Move PNG files
find . -maxdepth 1 -name "demo_*.png" -exec mv {} examples/output/ \;
find . -maxdepth 1 -name "integration_*.png" -exec mv {} examples/output/ \;

# Move JSON files
find . -maxdepth 1 -name "demo_*.json" -exec mv {} examples/output/ \;

# Move directories
[ -d "demo_dashboard" ] && mv demo_dashboard examples/output/
[ -d "demo_metrics" ] && mv demo_metrics examples/output/
[ -d "demo_results" ] && mv demo_results examples/output/
[ -d "integration_metrics" ] && mv integration_metrics examples/output/

# Move checkpoints and results
[ -d "checkpoints" ] && mv checkpoints/* examples/output/checkpoints/ 2>/dev/null
[ -d "results" ] && mv results/* examples/output/results/ 2>/dev/null

# Clean up empty directories
rmdir checkpoints results 2>/dev/null

# Move log files
find . -name "*.log" -exec mv {} examples/output/logs/ \;

echo "✅ Demo outputs cleaned up and organized!"
echo "📂 All demo outputs are now in examples/output/"
echo ""
echo "Directory structure:"
echo "  examples/output/"
echo "  ├── checkpoints/     # Model checkpoints"
echo "  ├── results/         # Training results"
echo "  ├── logs/           # Training logs"
echo "  ├── demos/          # Demo outputs"
echo "  └── *.png, *.json   # Visualization files"
