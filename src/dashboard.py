#!/usr/bin/env python3
"""
Fitness Center Dashboard Server
===============================
This module implements the central dashboard server that receives fitness data
from all wristband clients, processes it, and provides real-time monitoring
and alerting capabilities.

Features:
- Multi-port UDP listener for different data types
- Real-time data aggregation and analysis
- Emergency alert handling and logging
- QoS metrics collection (latency, packet loss, jitter)
- REST API for data access
- Web-based visualization dashboard

"""

import socket
import json
import threading
import time
import signal
import sys
import argparse
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
import statistics
import csv
import os


@dataclass
class ClientMetrics:
    """Metrics for a single client."""
    client_id: int
    client_name: str
    is_priority: bool = False
    
    # Latest values
    last_heart_rate: int = 0
    last_steps: int = 0
    last_workout_duration: float = 0.0
    last_activity_state: str = 'UNKNOWN'
    last_update: Optional[str] = None
    
    # Statistics
    packets_received: int = 0
    emergency_alerts: int = 0
    bytes_received: int = 0
    
    # QoS metrics
    latencies: List[float] = field(default_factory=list)
    packet_timestamps: List[float] = field(default_factory=list)
    
    # Heart rate history for analysis
    heart_rate_history: List[int] = field(default_factory=list)
    
    def calculate_qos_metrics(self) -> dict:
        """Calculate QoS metrics from collected data."""
        metrics = {
            'avg_latency_ms': 0,
            'max_latency_ms': 0,
            'min_latency_ms': 0,
            'jitter_ms': 0,
            'packet_loss_rate': 0
        }
        
        if self.latencies:
            metrics['avg_latency_ms'] = statistics.mean(self.latencies) * 1000
            metrics['max_latency_ms'] = max(self.latencies) * 1000
            metrics['min_latency_ms'] = min(self.latencies) * 1000
            if len(self.latencies) > 1:
                metrics['jitter_ms'] = statistics.stdev(self.latencies) * 1000
        
        return metrics


class FitnessDashboard:
    """
    Central dashboard for fitness center monitoring.
    
    Listens on multiple UDP ports for different data types:
    - Port 5001: Heart Rate data
    - Port 5002: Steps data
    - Port 5003: Workout duration data
    - Port 5004: Emergency alerts
    """
    
    # Port definitions
    PORT_HEARTRATE = 5001
    PORT_STEPS = 5002
    PORT_WORKOUT = 5003
    PORT_EMERGENCY = 5004
    
    # Alert thresholds
    HEART_RATE_SPIKE_THRESHOLD = 180
    HEART_RATE_LOW_THRESHOLD = 45
    
    def __init__(self, host: str = '0.0.0.0', 
                 http_port: int = 8080,
                 log_dir: str = './data'):
        """
        Initialize the dashboard server.
        
        Args:
            host: IP address to bind to
            http_port: Port for HTTP API
            log_dir: Directory for log files
        """
        self.host = host
        self.http_port = http_port
        self.log_dir = log_dir
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Client metrics storage
        self.clients: Dict[int, ClientMetrics] = {}
        
        # Emergency alerts queue
        self.emergency_alerts: deque = deque(maxlen=1000)
        
        # Global statistics
        self.global_stats = {
            'start_time': datetime.now(),
            'total_packets': 0,
            'total_emergency_alerts': 0,
            'bytes_received': 0
        }
        
        # Thread management
        self.running = False
        self.threads = []
        
        # Sockets
        self.sockets = {}
        
        # Data logging
        self.data_log_file = os.path.join(log_dir, 
            f'fitness_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        self._init_data_log()
        
        print(f"[Dashboard] Initialized")
        print(f"[Dashboard] Log directory: {log_dir}")
    
    def _init_data_log(self):
        """Initialize CSV data log file."""
        with open(self.data_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'client_id', 'client_name', 'is_priority',
                'heart_rate', 'steps', 'workout_duration', 'activity_state',
                'is_emergency', 'latency_ms', 'port'
            ])
    
    def _log_data(self, data: dict, latency: float, port: int):
        """Log received data to CSV file."""
        try:
            with open(self.data_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    data.get('timestamp', ''),
                    data.get('client_id', ''),
                    data.get('client_name', ''),
                    data.get('is_priority', False),
                    data.get('heart_rate', 0),
                    data.get('steps', 0),
                    data.get('workout_duration', 0),
                    data.get('activity_state', ''),
                    data.get('is_emergency', False),
                    latency * 1000,  # Convert to ms
                    port
                ])
        except Exception as e:
            print(f"[Dashboard] Error logging data: {e}")
    
    def _create_socket(self, port: int) -> socket.socket:
        """Create and bind a UDP socket."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, port))
        sock.settimeout(1.0)  # 1 second timeout for graceful shutdown
        return sock
    
    def _process_packet(self, data: bytes, addr: tuple, port: int):
        """Process received fitness data packet."""
        receive_time = time.time()
        
        try:
            # Parse JSON data
            fitness_data = json.loads(data.decode())
            
            client_id = fitness_data.get('client_id')
            if client_id is None:
                return
            
            # Calculate latency (approximate - based on timestamp parsing)
            try:
                send_time = datetime.fromisoformat(fitness_data['timestamp'])
                latency = (datetime.now() - send_time).total_seconds()
            except:
                latency = 0.0
            
            # Initialize client if new
            if client_id not in self.clients:
                self.clients[client_id] = ClientMetrics(
                    client_id=client_id,
                    client_name=fitness_data.get('client_name', f'Client{client_id}'),
                    is_priority=fitness_data.get('is_priority', False)
                )
                print(f"[Dashboard] New client registered: {self.clients[client_id].client_name}")
            
            client = self.clients[client_id]
            
            # Update client metrics
            client.last_heart_rate = fitness_data.get('heart_rate', 0)
            client.last_steps = fitness_data.get('steps', 0)
            client.last_workout_duration = fitness_data.get('workout_duration', 0)
            client.last_activity_state = fitness_data.get('activity_state', 'UNKNOWN')
            client.last_update = fitness_data.get('timestamp')
            client.is_priority = fitness_data.get('is_priority', False)
            
            # Update statistics
            client.packets_received += 1
            client.bytes_received += len(data)
            client.latencies.append(latency)
            client.packet_timestamps.append(receive_time)
            
            # Keep only last 1000 samples
            if len(client.latencies) > 1000:
                client.latencies = client.latencies[-1000:]
            if len(client.packet_timestamps) > 1000:
                client.packet_timestamps = client.packet_timestamps[-1000:]
            
            # Track heart rate history
            client.heart_rate_history.append(client.last_heart_rate)
            if len(client.heart_rate_history) > 1000:
                client.heart_rate_history = client.heart_rate_history[-1000:]
            
            # Global stats
            self.global_stats['total_packets'] += 1
            self.global_stats['bytes_received'] += len(data)
            
            # Handle emergency
            if fitness_data.get('is_emergency') or port == self.PORT_EMERGENCY:
                self._handle_emergency(client_id, fitness_data)
            
            # Check for abnormal heart rate
            self._check_heart_rate(client_id, client.last_heart_rate)
            
            # Log data
            self._log_data(fitness_data, latency, port)
            
        except json.JSONDecodeError:
            print(f"[Dashboard] Invalid JSON from {addr}")
        except Exception as e:
            print(f"[Dashboard] Error processing packet: {e}")
    
    def _handle_emergency(self, client_id: int, data: dict):
        """Handle emergency alert."""
        client = self.clients.get(client_id)
        if client:
            client.emergency_alerts += 1
        
        self.global_stats['total_emergency_alerts'] += 1
        
        alert = {
            'timestamp': datetime.now().isoformat(),
            'client_id': client_id,
            'client_name': data.get('client_name', f'Client{client_id}'),
            'heart_rate': data.get('heart_rate', 0),
            'emergency_type': data.get('emergency_type', 'UNKNOWN'),
            'is_priority': data.get('is_priority', False)
        }
        
        self.emergency_alerts.append(alert)
        
        # Print alert
        priority_str = "[PRIORITY] " if alert['is_priority'] else ""
        print(f"\n{'!'*60}")
        print(f"[Dashboard] 🚨 EMERGENCY ALERT 🚨 {priority_str}")
        print(f"[Dashboard] Client: {alert['client_name']}")
        print(f"[Dashboard] Heart Rate: {alert['heart_rate']} BPM")
        print(f"[Dashboard] Type: {alert['emergency_type']}")
        print(f"{'!'*60}\n")
    
    def _check_heart_rate(self, client_id: int, heart_rate: int):
        """Check for abnormal heart rate values."""
        if heart_rate >= self.HEART_RATE_SPIKE_THRESHOLD:
            client = self.clients.get(client_id)
            if client:
                print(f"[Dashboard] ⚠️  High HR detected for {client.client_name}: "
                      f"{heart_rate} BPM")
        elif heart_rate <= self.HEART_RATE_LOW_THRESHOLD and heart_rate > 0:
            client = self.clients.get(client_id)
            if client:
                print(f"[Dashboard] ⚠️  Low HR detected for {client.client_name}: "
                      f"{heart_rate} BPM")
    
    def _listen_port(self, port: int, port_name: str):
        """Listen on a specific port for data."""
        sock = self._create_socket(port)
        self.sockets[port] = sock
        
        print(f"[Dashboard] Listening on port {port} ({port_name})")
        
        while self.running:
            try:
                data, addr = sock.recvfrom(4096)
                self._process_packet(data, addr, port)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[Dashboard] Error on port {port}: {e}")
    
    def start(self):
        """Start the dashboard server."""
        self.running = True
        
        # Start UDP listeners
        port_configs = [
            (self.PORT_HEARTRATE, 'Heart Rate'),
            (self.PORT_STEPS, 'Steps'),
            (self.PORT_WORKOUT, 'Workout'),
            (self.PORT_EMERGENCY, 'Emergency')
        ]
        
        for port, name in port_configs:
            thread = threading.Thread(
                target=self._listen_port,
                args=(port, name),
                daemon=True
            )
            thread.start()
            self.threads.append(thread)
        
        # Start HTTP server
        http_thread = threading.Thread(
            target=self._run_http_server,
            daemon=True
        )
        http_thread.start()
        self.threads.append(http_thread)
        
        # Start statistics printer
        stats_thread = threading.Thread(
            target=self._print_stats_periodically,
            daemon=True
        )
        stats_thread.start()
        self.threads.append(stats_thread)
        
        print(f"\n[Dashboard] Server started")
        print(f"[Dashboard] HTTP API available at http://{self.host}:{self.http_port}")
    
    def stop(self):
        """Stop the dashboard server."""
        print("\n[Dashboard] Stopping server...")
        self.running = False
        
        # Close sockets
        for sock in self.sockets.values():
            try:
                sock.close()
            except:
                pass
        
        # Wait for threads
        for thread in self.threads:
            thread.join(timeout=2)
        
        self._print_final_summary()
    
    def _run_http_server(self):
        """Run HTTP server for REST API."""
        dashboard = self
        
        class DashboardHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Suppress HTTP logs
            
            def _send_json(self, data):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(data, indent=2, default=str).encode())
            
            def do_GET(self):
                if self.path == '/api/clients':
                    clients_data = {}
                    for cid, client in dashboard.clients.items():
                        clients_data[cid] = {
                            **asdict(client),
                            'qos_metrics': client.calculate_qos_metrics()
                        }
                        # Remove large lists
                        del clients_data[cid]['latencies']
                        del clients_data[cid]['packet_timestamps']
                        del clients_data[cid]['heart_rate_history']
                    self._send_json(clients_data)
                
                elif self.path == '/api/stats':
                    stats = {
                        **dashboard.global_stats,
                        'uptime_seconds': (datetime.now() - 
                            dashboard.global_stats['start_time']).total_seconds()
                    }
                    self._send_json(stats)
                
                elif self.path == '/api/alerts':
                    self._send_json(list(dashboard.emergency_alerts))
                
                elif self.path == '/api/qos':
                    qos_data = {}
                    for cid, client in dashboard.clients.items():
                        qos_data[cid] = {
                            'client_name': client.client_name,
                            'is_priority': client.is_priority,
                            **client.calculate_qos_metrics()
                        }
                    self._send_json(qos_data)
                
                elif self.path == '/' or self.path == '/dashboard':
                    self._serve_dashboard()
                
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def _serve_dashboard(self):
                """Serve the HTML dashboard."""
                html = dashboard._generate_dashboard_html()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                self.wfile.write(html.encode())
        
        server = HTTPServer((self.host, self.http_port), DashboardHandler)
        server.timeout = 1
        
        while self.running:
            server.handle_request()
    
    def _generate_dashboard_html(self) -> str:
        """Generate HTML dashboard page."""
        return '''
<!DOCTYPE html>
<html>
<head>
    <title>SDN Fitness Center Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
        .card { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .priority { border-left: 4px solid #ff9800; }
        .emergency { background: #ffebee; border-left: 4px solid #f44336; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; }
        .metric { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee; }
        .metric-value { font-weight: bold; color: #4CAF50; }
        .heart-rate { font-size: 2em; color: #f44336; }
        .alert { background: #f44336; color: white; padding: 15px; border-radius: 4px; margin: 10px 0; }
        #refresh { padding: 10px 20px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏋️ SDN Fitness Center Dashboard</h1>
        <button id="refresh" onclick="refreshData()">Refresh Data</button>
        <p>Auto-refresh: Every 2 seconds</p>
        
        <h2>📊 Client Status</h2>
        <div id="clients" class="grid"></div>
        
        <h2>🚨 Emergency Alerts</h2>
        <div id="alerts"></div>
        
        <h2>📈 QoS Metrics</h2>
        <div id="qos" class="grid"></div>
    </div>
    
    <script>
        function refreshData() {
            fetch('/api/clients')
                .then(r => r.json())
                .then(data => {
                    let html = '';
                    for (let [id, client] of Object.entries(data)) {
                        let priorityClass = client.is_priority ? 'priority' : '';
                        html += `
                            <div class="card ${priorityClass}">
                                <h3>${client.client_name} ${client.is_priority ? '⭐' : ''}</h3>
                                <div class="heart-rate">❤️ ${client.last_heart_rate} BPM</div>
                                <div class="metric"><span>Steps</span><span class="metric-value">${client.last_steps}</span></div>
                                <div class="metric"><span>Workout</span><span class="metric-value">${(client.last_workout_duration/60).toFixed(1)} min</span></div>
                                <div class="metric"><span>State</span><span class="metric-value">${client.last_activity_state}</span></div>
                                <div class="metric"><span>Packets</span><span class="metric-value">${client.packets_received}</span></div>
                                <div class="metric"><span>Avg Latency</span><span class="metric-value">${client.qos_metrics.avg_latency_ms.toFixed(2)} ms</span></div>
                            </div>
                        `;
                    }
                    document.getElementById('clients').innerHTML = html || '<p>No clients connected</p>';
                });
            
            fetch('/api/alerts')
                .then(r => r.json())
                .then(data => {
                    let html = '';
                    for (let alert of data.slice(-5).reverse()) {
                        html += `
                            <div class="card emergency">
                                <strong>${alert.client_name}</strong> - ${alert.emergency_type}<br>
                                Heart Rate: ${alert.heart_rate} BPM<br>
                                <small>${alert.timestamp}</small>
                            </div>
                        `;
                    }
                    document.getElementById('alerts').innerHTML = html || '<p>No alerts</p>';
                });
            
            fetch('/api/qos')
                .then(r => r.json())
                .then(data => {
                    let html = '';
                    for (let [id, qos] of Object.entries(data)) {
                        html += `
                            <div class="card">
                                <h4>${qos.client_name}</h4>
                                <div class="metric"><span>Avg Latency</span><span class="metric-value">${qos.avg_latency_ms.toFixed(2)} ms</span></div>
                                <div class="metric"><span>Max Latency</span><span class="metric-value">${qos.max_latency_ms.toFixed(2)} ms</span></div>
                                <div class="metric"><span>Jitter</span><span class="metric-value">${qos.jitter_ms.toFixed(2)} ms</span></div>
                            </div>
                        `;
                    }
                    document.getElementById('qos').innerHTML = html || '<p>No QoS data</p>';
                });
        }
        
        refreshData();
        setInterval(refreshData, 2000);
    </script>
</body>
</html>
        '''
    
    def _print_stats_periodically(self):
        """Print statistics every 10 seconds."""
        while self.running:
            time.sleep(10)
            if self.running:
                self._print_current_stats()
    
    def _print_current_stats(self):
        """Print current statistics."""
        print("\n" + "-"*60)
        print("[Dashboard] CURRENT STATUS")
        print("-"*60)
        
        for cid, client in self.clients.items():
            priority_str = "[PRIORITY] " if client.is_priority else ""
            qos = client.calculate_qos_metrics()
            print(f"  {client.client_name} {priority_str}")
            print(f"    HR: {client.last_heart_rate} BPM | "
                  f"Steps: {client.last_steps} | "
                  f"State: {client.last_activity_state}")
            print(f"    Packets: {client.packets_received} | "
                  f"Latency: {qos['avg_latency_ms']:.2f}ms | "
                  f"Alerts: {client.emergency_alerts}")
        
        print(f"\n  Total Packets: {self.global_stats['total_packets']}")
        print(f"  Total Alerts: {self.global_stats['total_emergency_alerts']}")
        print("-"*60 + "\n")
    
    def _print_final_summary(self):
        """Print final summary when stopping."""
        print("\n" + "="*60)
        print("[Dashboard] FINAL SUMMARY")
        print("="*60)
        
        runtime = (datetime.now() - self.global_stats['start_time']).total_seconds()
        print(f"\nRuntime: {runtime:.1f} seconds")
        print(f"Total Packets Received: {self.global_stats['total_packets']}")
        print(f"Total Emergency Alerts: {self.global_stats['total_emergency_alerts']}")
        print(f"Total Bytes Received: {self.global_stats['bytes_received']}")
        
        print("\nPer-Client Summary:")
        print("-"*60)
        
        for cid, client in self.clients.items():
            priority_str = "[PRIORITY] " if client.is_priority else ""
            qos = client.calculate_qos_metrics()
            print(f"\n  {client.client_name} {priority_str}")
            print(f"    Packets: {client.packets_received}")
            print(f"    Bytes: {client.bytes_received}")
            print(f"    Emergency Alerts: {client.emergency_alerts}")
            print(f"    Avg Latency: {qos['avg_latency_ms']:.2f} ms")
            print(f"    Max Latency: {qos['max_latency_ms']:.2f} ms")
            print(f"    Jitter: {qos['jitter_ms']:.2f} ms")
        
        print("\n" + "="*60)
        print(f"Data logged to: {self.data_log_file}")
        print("="*60 + "\n")


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\nReceived interrupt signal. Stopping dashboard...")
    sys.exit(0)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Fitness Center Dashboard Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    # Run dashboard on default settings
    python3 dashboard.py
    
    # Run with custom HTTP port
    python3 dashboard.py --http-port 9000
    
    # Run with custom log directory
    python3 dashboard.py --log-dir ./my_logs
        '''
    )
    
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host address to bind to')
    parser.add_argument('--http-port', type=int, default=8080,
                        help='HTTP API port')
    parser.add_argument('--log-dir', type=str, default='./data',
                        help='Directory for log files')
    
    args = parser.parse_args()
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and start dashboard
    dashboard = FitnessDashboard(
        host=args.host,
        http_port=args.http_port,
        log_dir=args.log_dir
    )
    
    print("\n" + "="*60)
    print("FITNESS CENTER DASHBOARD SERVER")
    print("="*60)
    print(f"Host: {args.host}")
    print(f"HTTP API Port: {args.http_port}")
    print(f"Log Directory: {args.log_dir}")
    print("="*60)
    print("\nListening for wristband data...")
    print("Press Ctrl+C to stop...\n")
    
    dashboard.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.stop()


if __name__ == '__main__':
    main()
