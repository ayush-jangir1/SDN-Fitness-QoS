#!/bin/bash
#
# SDN Fitness Center - Complete Simulation Runner
# ================================================
# This script automates the entire simulation process:
# 1. Starts the Ryu controller
# 2. Creates the Mininet topology
# 3. Launches client simulators and dashboard
# 4. Collects data for specified duration
# 5. Runs analysis on collected data
#
# Usage: ./run_simulation.sh [duration_seconds]
#
# Authors: SDN Fitness Center Research Team
# License: MIT

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_DIR/src"
DATA_DIR="$PROJECT_DIR/data"
RESULTS_DIR="$PROJECT_DIR/results"
LOG_DIR="$PROJECT_DIR/logs"

# Default simulation duration (seconds)
DURATION=${1:-120}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Create directories
mkdir -p "$DATA_DIR" "$RESULTS_DIR" "$LOG_DIR"

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    
    # Kill background processes
    if [ ! -z "$CONTROLLER_PID" ]; then
        kill $CONTROLLER_PID 2>/dev/null || true
    fi
    
    # Stop Mininet
    sudo mn -c 2>/dev/null || true
    
    # Kill any remaining Python processes from our simulation
    pkill -f "wristband_client.py" 2>/dev/null || true
    pkill -f "dashboard.py" 2>/dev/null || true
    pkill -f "ryu-manager" 2>/dev/null || true
    
    log_info "Cleanup complete"
}

# Set trap for cleanup
trap cleanup EXIT

# Print banner
echo ""
echo "========================================================"
echo "     SDN FITNESS CENTER - SIMULATION RUNNER"
echo "========================================================"
echo ""
echo "Configuration:"
echo "  Duration: ${DURATION} seconds"
echo "  Data Dir: ${DATA_DIR}"
echo "  Results Dir: ${RESULTS_DIR}"
echo ""

# Step 1: Check prerequisites
log_step "Checking prerequisites..."

# Check for Mininet
if ! command -v mn &> /dev/null; then
    log_error "Mininet is not installed. Please install it first."
    exit 1
fi

# Check for Ryu
if ! command -v ryu-manager &> /dev/null; then
    log_warn "Ryu is not installed. Installing via pip..."
    pip install ryu --break-system-packages 2>/dev/null || pip install ryu
fi

# Check for Python packages
python3 -c "import numpy, pandas, matplotlib" 2>/dev/null || {
    log_warn "Installing required Python packages..."
    pip install numpy pandas matplotlib seaborn scipy --break-system-packages 2>/dev/null || \
    pip install numpy pandas matplotlib seaborn scipy
}

log_info "Prerequisites check passed"

# Step 2: Start Ryu Controller
log_step "Starting Ryu SDN Controller..."

ryu-manager "$SRC_DIR/controller.py" \
    --observe-links \
    --ofp-tcp-listen-port 6633 \
    > "$LOG_DIR/controller.log" 2>&1 &
CONTROLLER_PID=$!

sleep 3

if ! kill -0 $CONTROLLER_PID 2>/dev/null; then
    log_error "Failed to start Ryu controller. Check $LOG_DIR/controller.log"
    exit 1
fi

log_info "Ryu controller started (PID: $CONTROLLER_PID)"
log_info "Controller REST API: http://localhost:8080"

# Step 3: Create Mininet Network
log_step "Creating Mininet topology..."

# Create a script that will run inside Mininet
cat > /tmp/mininet_commands.py << 'EOF'
#!/usr/bin/env python3
import time
import subprocess
import os

# Wait for network to be ready
time.sleep(2)

# Get references to hosts
dashboard = net.get('dashboard')
client1 = net.get('client1')
client2 = net.get('client2')
client3 = net.get('client3')
client4 = net.get('client4')

# Start dashboard
print("Starting dashboard server...")
dashboard.cmd('cd /tmp/sdn_fitness && python3 src/dashboard.py --log-dir /tmp/sdn_fitness/data &')
time.sleep(2)

# Start clients
print("Starting wristband clients...")
client1.cmd('cd /tmp/sdn_fitness && python3 src/wristband_client.py --client-id 1 --priority --dashboard 10.0.0.5 &')
time.sleep(0.5)
client2.cmd('cd /tmp/sdn_fitness && python3 src/wristband_client.py --client-id 2 --dashboard 10.0.0.5 &')
time.sleep(0.5)
client3.cmd('cd /tmp/sdn_fitness && python3 src/wristband_client.py --client-id 3 --dashboard 10.0.0.5 &')
time.sleep(0.5)
client4.cmd('cd /tmp/sdn_fitness && python3 src/wristband_client.py --client-id 4 --dashboard 10.0.0.5 &')

print("All components started!")
EOF

# Copy project to accessible location
sudo rm -rf /tmp/sdn_fitness
sudo cp -r "$PROJECT_DIR" /tmp/sdn_fitness
sudo chmod -R 777 /tmp/sdn_fitness

# Start Mininet with topology
log_info "Launching Mininet network..."
echo ""
echo "========================================================"
echo "IMPORTANT: Mininet CLI will start now"
echo ""
echo "Run these commands in Mininet CLI:"
echo "  1. dashboard python3 /tmp/sdn_fitness/src/dashboard.py --log-dir /tmp/sdn_fitness/data &"
echo "  2. client1 python3 /tmp/sdn_fitness/src/wristband_client.py --client-id 1 --priority --dashboard 10.0.0.5 &"
echo "  3. client2 python3 /tmp/sdn_fitness/src/wristband_client.py --client-id 2 --dashboard 10.0.0.5 &"
echo "  4. client3 python3 /tmp/sdn_fitness/src/wristband_client.py --client-id 3 --dashboard 10.0.0.5 &"
echo "  5. client4 python3 /tmp/sdn_fitness/src/wristband_client.py --client-id 4 --dashboard 10.0.0.5 &"
echo ""
echo "Or for automated running, exit CLI and use the automated script."
echo "========================================================"
echo ""

sudo python3 "$SRC_DIR/topology.py"

# After Mininet exits, copy data back
if [ -d "/tmp/sdn_fitness/data" ]; then
    cp -r /tmp/sdn_fitness/data/* "$DATA_DIR/" 2>/dev/null || true
fi

log_info "Simulation complete"

# Step 4: Run Analysis
log_step "Running data analysis..."

# Find the latest data file
LATEST_DATA=$(ls -t "$DATA_DIR"/fitness_data_*.csv 2>/dev/null | head -1)

if [ -n "$LATEST_DATA" ]; then
    log_info "Analyzing: $LATEST_DATA"
    python3 "$SRC_DIR/analysis.py" "$LATEST_DATA" -o "$RESULTS_DIR"
    log_info "Analysis complete. Results in: $RESULTS_DIR"
else
    log_warn "No data files found for analysis"
fi

echo ""
echo "========================================================"
echo "     SIMULATION COMPLETE"
echo "========================================================"
echo ""
echo "Results:"
echo "  Data files: $DATA_DIR"
echo "  Analysis results: $RESULTS_DIR"
echo "  Logs: $LOG_DIR"
echo ""
