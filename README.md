## Project Files Description

### Directory Structure

```
sdn_fitness_center/
├── src/
│   ├── topology.py          # Mininet network topology
│   ├── controller.py        # Ryu SDN controller
│   ├── wristband_client.py  # Wristband data simulator
│   ├── dashboard.py         # Monitoring dashboard server
│   └── analysis.py          # Data analysis and visualization
├── scripts/
│   ├── run_simulation.sh    # Full simulation automation
│   └── run_demo.py          # Quick demo (no Mininet needed)
├── data/
│   └── fitness_data_*.csv   # Collected simulation data
├── results/
│   ├── analysis_report.txt  # Text report
│   ├── analysis_results.json
│   ├── qos_summary.png
│   ├── latency_comparison.png
│   ├── latency_distribution.png
│   ├── latency_over_time.png
│   ├── heart_rate_analysis.png
│   ├── throughput_analysis.png
│   └── emergency_analysis.png
└── logs/
    └── controller.log
```

## Executive Summary

This project demonstrates the implementation of Software-Defined Networking (SDN) for Quality of Service (QoS) management in a fitness center IoT environment. The system monitors health metrics from wearable devices (wristbands) and prioritizes network traffic based on client health conditions.

**Technologies Used:**
- Mininet (Network Emulator)
- Ryu SDN Controller (OpenFlow 1.3)
- Python 3.9
- Open vSwitch (OVS)

### Objectives

1. Design and implement an SDN-based fitness center network topology
2. Develop a Ryu controller with QoS flow management capabilities
3. Create synthetic wristband data generators simulating real health metrics
4. Implement three-tier traffic prioritization:
   - Emergency traffic (highest priority)
   - Priority client traffic (injury recovery)
   - Normal client traffic (regular members)
5. Develop a real-time monitoring dashboard

## System Architecture

### Network Topology

```
                    ┌─────────────────┐
                    │  Ryu Controller │
                    │   (OpenFlow)    │
                    └────────┬────────┘
                             │
                             │ Control Plane
                    ─────────┼─────────
                             │ Data Plane
                             │
┌──────────┐    ┌────────────┴────────────┐    ┌──────────────┐
│ Client 1 │────│                         │    │              │
│(Priority)│    │   Access Point Switch   │────│  Aggregation │
├──────────┤    │        (s1)             │    │   Switch     │
│ Client 2 │────│                         │    │    (s2)      │
├──────────┤    │   DPID: 0x01            │    │              │
│ Client 3 │────│                         │    │  DPID: 0x02  │
├──────────┤    └─────────────────────────┘    └──────┬───────┘
│ Client 4 │                                          │
└──────────┘                                          │
                                               ┌──────┴───────┐
                                               │  Dashboard   │
                                               │   Server     │
                                               │  10.0.2.1    │
                                               └──────────────┘
```

### IP Addressing Scheme

| Device | IP Address | Subnet | Role |
|--------|------------|--------|------|
| Client 1 | 10.0.1.1 | /24 | Priority (Injury Recovery) |
| Client 2 | 10.0.1.2 | /24 | Normal Member |
| Client 3 | 10.0.1.3 | /24 | Normal Member |
| Client 4 | 10.0.1.4 | /24 | Normal Member |
| Dashboard | 10.0.2.1 | /24 | Monitoring Server |

### Port Assignments

| Port | Data Type | Description |
|------|-----------|-------------|
| 5001 | Heart Rate | Continuous heart rate data |
| 5002 | Steps | Step count updates |
| 5003 | Workout | Workout duration and status |
| 5004 | Emergency | Heart rate spike alerts (>180 BPM) |

### Link Parameters

| Link | Bandwidth | Delay | Purpose |
|------|-----------|-------|---------|
| Client → AP Switch | 10 Mbps | 5 ms | Wireless simulation |
| AP → Aggregation | 100 Mbps | 2 ms | Backbone connection |
| Aggregation → Dashboard | 1000 Mbps | 1 ms | High-speed server link |


## Technical Implementation

### Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    SDN FITNESS CENTER SYSTEM                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Mininet    │  │     Ryu      │  │    Wristband         │  │
│  │   Topology   │  │  Controller  │  │    Clients           │  │
│  │              │  │              │  │                      │  │
│  │ - 4 Clients  │  │ - QoS Rules  │  │ - Heart Rate Gen     │  │
│  │ - 2 Switches │  │ - Flow Mgmt  │  │ - Steps Counter      │  │
│  │ - 1 Server   │  │ - REST API   │  │ - Activity States    │  │
│  │ - OVS Queues │  │ - Alerts     │  │ - Emergency Detect   │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│                                                                  │
│  ┌──────────────┐  ┌──────────────────────────────────────────┐ │
│  │  Dashboard   │  │              Analysis Engine              │ │
│  │   Server     │  │                                          │ │
│  │              │  │ - Statistical Tests (Mann-Whitney U)     │ │
│  │ - UDP Listen │  │ - Latency Analysis                       │ │
│  │ - Data Log   │  │ - Throughput Metrics                     │ │
│  │ - REST API   │  │ - Publication-Quality Visualizations     │ │
│  │ - Web UI     │  │ - Automated Report Generation            │ │
│  └──────────────┘  └──────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Wristband clients** generate synthetic health data
2. **Data packets** are sent via UDP to the dashboard server
3. **AP Switch (s1)** receives packets and queries controller
4. **Ryu Controller** installs flow rules based on:
   - Source IP (priority client check)
   - Destination port (emergency check)
5. **QoS queues** process packets with appropriate priority
6. **Dashboard** receives data, calculates latency, logs to CSV
7. **Analysis engine** processes CSV and generates visualizations

### Activity State Machine

```
                 ┌─────────┐
                 │ RESTING │
                 │ HR: 70  │
                 └────┬────┘
                      │ Start workout
                      ▼
                 ┌─────────┐
                 │ WARMUP  │
                 │ HR: 100 │
                 └────┬────┘
                      │ Intensity increases
                      ▼
                 ┌─────────┐
                 │EXERCISE │
                 │ HR: 145 │
                 └────┬────┘
                      │ Workout ends
                      ▼
                 ┌─────────┐
                 │COOLDOWN │
                 │ HR: 95  │
                 └────┬────┘
                      │ Recovery complete
                      ▼
                 ┌─────────┐
                 │ RESTING │
                 └─────────┘

    Priority Client (Injury Recovery):
                 ┌────────────────┐
                 │INJURY_RECOVERY │
                 │    HR: 90      │
                 │ (Constant)     │
                 └────────────────┘
```

### Installation Steps

#### Step 1: Download/Transfer Project Files

Transfer the `sdn_fitness_center` folder to your Mininet VM desktop.

#### Step 2: Open Terminal

```bash
cd ~/Desktop/sdn_fitness_center
```

#### Step 3: Install Dependencies

```bash
sudo apt update
sudo pip3 install ryu numpy pandas matplotlib seaborn scipy
```

#### Step 4: Set Permissions

```bash
chmod +x scripts/*.sh scripts/*.py
```

#### Step 5: Create Required Directories

```bash
mkdir -p data results logs
```

#### Step 6: Verify Installation

```bash
python3 -c "import ryu; import numpy; import pandas; import matplotlib; print('All packages installed!')"
```

---

## Step-by-Step Execution Guide

### Full Mininet Simulation

This method runs the complete SDN simulation with actual network emulation.

#### Step 1: Open Two Terminal Windows

- Terminal 1: For Mininet topology
- Terminal 2: For Ryu controller

#### Step 2: Start Ryu Controller (Terminal 2)

```bash
cd ~/Desktop/sdn_fitness_center
ryu-manager src/controller.py --observe-links
```

**Expected Output:**
```
loading app src/controller.py
loading app ryu.controller.ofp_handler
instantiating app src/controller.py of FitnessQoSController
instantiating app ryu.controller.ofp_handler of OFPHandler
```

**Keep this terminal running!**

#### Step 3: Start Mininet Topology (Terminal 1)

```bash
cd ~/Desktop/sdn_fitness_center
sudo python3 src/topology.py
```

**Expected Output:**
```
*** Creating network
*** Adding controller
*** Adding hosts: client1 client2 client3 client4 dashboard
*** Adding switches: s1 s2
*** Adding links
*** Configuring hosts
*** Starting controller
*** Starting switches
*** Configuring QoS queues
mininet>
```

#### Step 4: Start Dashboard Server (In Mininet CLI)

```bash
dashboard python3 /home/mininet/Desktop/sdn_fitness_center/src/dashboard.py --log-dir /home/mininet/Desktop/sdn_fitness_center/data &
```

Wait 2-3 seconds for the dashboard to initialize.

#### Step 5: Start Wristband Clients (In Mininet CLI)

```bash
client1 python3 /home/mininet/Desktop/sdn_fitness_center/src/wristband_client.py --client-id 1 --priority --dashboard 10.0.2.1 &
```

```bash
client2 python3 /home/mininet/Desktop/sdn_fitness_center/src/wristband_client.py --client-id 2 --dashboard 10.0.2.1 &
```

```bash
client3 python3 /home/mininet/Desktop/sdn_fitness_center/src/wristband_client.py --client-id 3 --dashboard 10.0.2.1 &
```

```bash
client4 python3 /home/mininet/Desktop/sdn_fitness_center/src/wristband_client.py --client-id 4 --dashboard 10.0.2.1 &
```

#### Step 6: Let Simulation Run

Wait **2-3 minutes** for data collection.

You can verify data is being collected:
```bash
dashboard ls /home/mininet/Desktop/sdn_fitness_center/data/
```

#### Step 7: Stop Simulation

```bash
exit
```

This closes Mininet. Press `Ctrl+C` in Terminal 2 to stop the controller.

#### Step 8: Run Analysis

```bash
cd ~/Desktop/sdn_fitness_center
python3 src/analysis.py data/fitness_data_*.csv -o results/
```

**Note:** If multiple CSV files exist, specify one:
```bash
python3 src/analysis.py data/fitness_data_YYYYMMDD_HHMMSS.csv -o results/
```

#### Step 9: View Results

```bash
cat results/analysis_report.txt
xdg-open results/qos_summary.png
```
              
