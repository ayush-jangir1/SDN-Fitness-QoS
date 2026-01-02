#!/usr/bin/env python3
"""
SDN-Based Fitness Center Network Topology
==========================================
This module implements a Mininet topology simulating a smart fitness center
with wearable devices, access points, and a central dashboard.

Architecture:
    [Client1-Wristband] ----\
    [Client2-Wristband] -----\___[Access Point Switch]---[Aggregation Switch]---[Dashboard Server]
    [Client3-Wristband] -----/          |                        |
    [Client4-Wristband] ----/     [Ryu Controller]         [Controller]

Authors: SDN Fitness Center Research Team
License: MIT
"""

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink
import argparse
import json
import os
import sys
import time


class FitnessCenterTopo(Topo):
    """
    Custom topology for SDN-based Fitness Center simulation.
    
    Network Components:
    - 4 Wristband Clients (IoT devices)
    - 1 Access Point Switch (s1)
    - 1 Aggregation Switch (s2)
    - 1 Dashboard Server
    
    QoS Configuration:
    - Priority Client (recovering from injury): High priority queue
    - Normal Clients: Standard priority queue
    - Emergency traffic: Highest priority (heart rate spikes)
    """
    
    def __init__(self, config=None):
        """Initialize topology with optional configuration."""
        self.config = config or self._default_config()
        super(FitnessCenterTopo, self).__init__()
    
    def _default_config(self):
        """Return default configuration for the topology."""
        return {
            "num_clients": 4,
            "priority_client": 1,  # Client 1 is recovering from injury
            "link_params": {
                "wristband_to_ap": {"bw": 10, "delay": "5ms", "loss": 0},
                "ap_to_agg": {"bw": 100, "delay": "2ms", "loss": 0},
                "agg_to_dashboard": {"bw": 1000, "delay": "1ms", "loss": 0}
            },
            "client_ips": {
                1: "10.0.1.1/24",
                2: "10.0.1.2/24",
                3: "10.0.1.3/24",
                4: "10.0.1.4/24"
            },
            "dashboard_ip": "10.0.2.1/24",
            "gateway_ip": "10.0.0.254"
        }
    
    def build(self):
        """Build the network topology."""
        info("*** Building Fitness Center Topology ***\n")
        
        # Create switches
        # s1: Access Point Switch (connects wristbands)
        # s2: Aggregation Switch (connects to dashboard)
        ap_switch = self.addSwitch('s1', cls=OVSKernelSwitch, 
                                    protocols='OpenFlow13',
                                    dpid='0000000000000001')
        agg_switch = self.addSwitch('s2', cls=OVSKernelSwitch,
                                     protocols='OpenFlow13',
                                     dpid='0000000000000002')
        
        info("*** Adding switches: s1 (Access Point), s2 (Aggregation) ***\n")
        
        # Create wristband clients (IoT devices)
        clients = []
        for i in range(1, self.config["num_clients"] + 1):
            client_name = f'client{i}'
            ip = self.config["client_ips"][i]
            
            # Mark priority client
            is_priority = (i == self.config["priority_client"])
            client = self.addHost(client_name, ip=ip)
            clients.append(client)
            
            # Connect client to access point switch with appropriate bandwidth
            link_params = self.config["link_params"]["wristband_to_ap"]
            self.addLink(client, ap_switch, 
                        bw=link_params["bw"],
                        delay=link_params["delay"],
                        loss=link_params["loss"])
            
            priority_str = " [PRIORITY - Injury Recovery]" if is_priority else ""
            info(f"*** Added {client_name} ({ip}){priority_str} ***\n")
        
        # Create dashboard server
        dashboard = self.addHost('dashboard', ip=self.config["dashboard_ip"])
        info(f"*** Added dashboard server ({self.config['dashboard_ip']}) ***\n")
        
        # Connect switches
        link_params = self.config["link_params"]["ap_to_agg"]
        self.addLink(ap_switch, agg_switch,
                    bw=link_params["bw"],
                    delay=link_params["delay"],
                    loss=link_params["loss"])
        info("*** Connected s1 (AP) to s2 (Aggregation) ***\n")
        
        # Connect dashboard to aggregation switch
        link_params = self.config["link_params"]["agg_to_dashboard"]
        self.addLink(dashboard, agg_switch,
                    bw=link_params["bw"],
                    delay=link_params["delay"],
                    loss=link_params["loss"])
        info("*** Connected dashboard to s2 (Aggregation) ***\n")


def setup_qos_queues(net):
    """
    Configure QoS queues on OVS switches for traffic prioritization.
    
    Queue Configuration:
    - Queue 0: Best Effort (normal traffic)
    - Queue 1: Priority (injury recovery client)
    - Queue 2: Emergency (heart rate spike alerts)
    """
    info("\n*** Configuring QoS Queues ***\n")
    
    switches = ['s1', 's2']
    
    for switch in switches:
        # Get all ports for this switch
        result = net.get(switch).cmd(f'ovs-vsctl list-ports {switch}')
        ports = result.strip().split('\n')
        
        for port in ports:
            if port:
                # Create QoS with three queues
                qos_cmd = f'''
                ovs-vsctl -- set port {port} qos=@newqos \
                -- --id=@newqos create qos type=linux-htb \
                   other-config:max-rate=1000000000 \
                   queues:0=@q0 queues:1=@q1 queues:2=@q2 \
                -- --id=@q0 create queue other-config:min-rate=10000000 other-config:max-rate=100000000 \
                -- --id=@q1 create queue other-config:min-rate=50000000 other-config:max-rate=500000000 \
                -- --id=@q2 create queue other-config:min-rate=100000000 other-config:max-rate=1000000000
                '''
                net.get(switch).cmd(qos_cmd)
        
        info(f"*** QoS configured on {switch} ***\n")


def configure_hosts(net, config):
    """Configure IP routing on hosts."""
    info("\n*** Configuring host networking ***\n")
    
    # Configure clients
    for i in range(1, config["num_clients"] + 1):
        client = net.get(f'client{i}')
        # Add route to dashboard network
        client.cmd('ip route add 10.0.2.0/24 via 10.0.1.254 2>/dev/null || true')
    
    # Configure dashboard
    dashboard = net.get('dashboard')
    dashboard.cmd('ip route add 10.0.1.0/24 via 10.0.2.254 2>/dev/null || true')
    
    info("*** Host networking configured ***\n")


def run_fitness_center_network(config_file=None, cli_mode=True):
    """
    Main function to create and run the fitness center network.
    
    Args:
        config_file: Path to JSON configuration file (optional)
        cli_mode: Whether to start Mininet CLI
    
    Returns:
        Mininet network object
    """
    setLogLevel('info')
    
    # Load configuration
    config = None
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        info(f"*** Loaded configuration from {config_file} ***\n")
    
    # Create topology
    topo = FitnessCenterTopo(config)
    
    # Create network with remote controller (Ryu)
    info("\n*** Creating network with Ryu controller ***\n")
    net = Mininet(
        topo=topo,
        controller=lambda name: RemoteController(name, ip='127.0.0.1', port=6633),
        switch=OVSKernelSwitch,
        link=TCLink,
        autoSetMacs=True
    )
    
    # Start network
    info("\n*** Starting network ***\n")
    net.start()
    
    # Wait for controller connection
    info("*** Waiting for controller connection ***\n")
    time.sleep(2)
    
    # Setup QoS queues
    setup_qos_queues(net)
    
    # Configure host networking
    effective_config = config or topo.config
    configure_hosts(net, effective_config)
    
    # Test connectivity
    info("\n*** Testing connectivity ***\n")
    net.pingAll()
    
    # Print network information
    print_network_info(net)
    
    if cli_mode:
        info("\n*** Starting CLI ***\n")
        info("*** Use 'help' for available commands ***\n")
        info("*** To start the simulation, run the scripts on each host ***\n")
        CLI(net)
        net.stop()
    
    return net


def print_network_info(net):
    """Print detailed network information."""
    info("\n" + "="*60 + "\n")
    info("FITNESS CENTER SDN NETWORK - TOPOLOGY INFORMATION\n")
    info("="*60 + "\n\n")
    
    info("HOSTS:\n")
    info("-"*40 + "\n")
    for host in net.hosts:
        info(f"  {host.name}: {host.IP()}\n")
    
    info("\nSWITCHES:\n")
    info("-"*40 + "\n")
    for switch in net.switches:
        info(f"  {switch.name}: dpid={switch.dpid}\n")
    
    info("\nLINKS:\n")
    info("-"*40 + "\n")
    for link in net.links:
        info(f"  {link.intf1} <--> {link.intf2}\n")
    
    info("\nQoS QUEUES:\n")
    info("-"*40 + "\n")
    info("  Queue 0: Best Effort (10-100 Mbps)\n")
    info("  Queue 1: Priority - Injury Recovery (50-500 Mbps)\n")
    info("  Queue 2: Emergency - Heart Rate Alerts (100-1000 Mbps)\n")
    
    info("\n" + "="*60 + "\n")


def create_default_config():
    """Create and save default configuration file."""
    config = {
        "num_clients": 4,
        "priority_client": 1,
        "link_params": {
            "wristband_to_ap": {"bw": 10, "delay": "5ms", "loss": 0},
            "ap_to_agg": {"bw": 100, "delay": "2ms", "loss": 0},
            "agg_to_dashboard": {"bw": 1000, "delay": "1ms", "loss": 0}
        },
        "client_ips": {
            "1": "10.0.1.1/24",
            "2": "10.0.1.2/24",
            "3": "10.0.1.3/24",
            "4": "10.0.1.4/24"
        },
        "dashboard_ip": "10.0.2.1/24",
        "simulation": {
            "duration_seconds": 300,
            "data_interval_ms": 1000,
            "priority_client_condition": "injury_recovery",
            "heart_rate_spike_threshold": 180
        }
    }
    
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Default configuration saved to {config_path}")
    return config_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SDN-Based Fitness Center Network Topology',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    python3 topology.py                    # Run with default config
    python3 topology.py -c config.json     # Run with custom config
    python3 topology.py --create-config    # Create default config file
        '''
    )
    
    parser.add_argument('-c', '--config', 
                        help='Path to configuration JSON file')
    parser.add_argument('--create-config', action='store_true',
                        help='Create default configuration file')
    parser.add_argument('--no-cli', action='store_true',
                        help='Run without CLI (for automated testing)')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_default_config()
        sys.exit(0)
    
    run_fitness_center_network(
        config_file=args.config,
        cli_mode=not args.no_cli
    )
