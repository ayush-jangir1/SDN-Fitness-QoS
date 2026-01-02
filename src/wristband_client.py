#!/usr/bin/env python3
"""
Wristband Client Simulator
==========================
This module simulates a fitness wristband device that generates and sends
synthetic health data to the dashboard server.

Data Generated:
- Heart Rate (BPM)
- Steps Count
- Workout Duration (seconds)

Features:
- Realistic data patterns based on activity state
- Heart rate spike simulation for emergency alerts
- Priority client simulation (injury recovery patterns)
- Configurable transmission intervals

Authors: SDN Fitness Center Research Team
License: MIT
"""

import socket
import json
import time
import random
import argparse
import threading
import signal
import sys
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np


@dataclass
class FitnessData:
    """Data structure for fitness metrics."""
    timestamp: str
    client_id: int
    client_name: str
    heart_rate: int
    steps: int
    workout_duration: float
    activity_state: str
    is_emergency: bool = False
    emergency_type: Optional[str] = None


class WristbandSimulator:
    """
    Simulates a fitness wristband device with realistic data patterns.
    
    Activity States:
    - RESTING: Low heart rate, no steps
    - WARMUP: Gradually increasing heart rate
    - EXERCISE: High heart rate, active stepping
    - COOLDOWN: Decreasing heart rate
    - INJURY_RECOVERY: Modified patterns for recovering client
    """
    
    # Port definitions (must match controller)
    PORT_HEARTRATE = 5001
    PORT_STEPS = 5002
    PORT_WORKOUT = 5003
    PORT_EMERGENCY = 5004
    
    # Activity state definitions
    STATES = ['RESTING', 'WARMUP', 'EXERCISE', 'COOLDOWN']
    
    # Heart rate parameters by state
    HR_PARAMS = {
        'RESTING': {'mean': 70, 'std': 5, 'min': 55, 'max': 85},
        'WARMUP': {'mean': 100, 'std': 10, 'min': 85, 'max': 120},
        'EXERCISE': {'mean': 145, 'std': 15, 'min': 120, 'max': 175},
        'COOLDOWN': {'mean': 95, 'std': 8, 'min': 80, 'max': 115},
        'INJURY_RECOVERY': {'mean': 90, 'std': 8, 'min': 70, 'max': 110}
    }
    
    # Steps per minute by state
    STEPS_PARAMS = {
        'RESTING': {'mean': 0, 'std': 0, 'min': 0, 'max': 5},
        'WARMUP': {'mean': 60, 'std': 15, 'min': 30, 'max': 90},
        'EXERCISE': {'mean': 130, 'std': 20, 'min': 100, 'max': 180},
        'COOLDOWN': {'mean': 40, 'std': 10, 'min': 20, 'max': 70},
        'INJURY_RECOVERY': {'mean': 50, 'std': 10, 'min': 30, 'max': 80}
    }
    
    def __init__(self, client_id: int, dashboard_ip: str, 
                 is_priority: bool = False, 
                 data_interval: float = 1.0,
                 enable_spikes: bool = True,
                 spike_probability: float = 0.02):
        """
        Initialize the wristband simulator.
        
        Args:
            client_id: Unique client identifier (1-4)
            dashboard_ip: IP address of the dashboard server
            is_priority: Whether this is a priority (injury recovery) client
            data_interval: Seconds between data transmissions
            enable_spikes: Enable random heart rate spikes
            spike_probability: Probability of spike per interval
        """
        self.client_id = client_id
        self.client_name = f"Client{client_id}"
        self.dashboard_ip = dashboard_ip
        self.is_priority = is_priority
        self.data_interval = data_interval
        self.enable_spikes = enable_spikes
        self.spike_probability = spike_probability
        
        # State management
        self.current_state = 'RESTING'
        self.state_duration = 0
        self.total_steps = 0
        self.workout_start_time = None
        self.last_heart_rate = 70
        
        # Activity cycle timing (seconds)
        self.state_durations = {
            'RESTING': (30, 60),
            'WARMUP': (60, 120),
            'EXERCISE': (300, 600),
            'COOLDOWN': (60, 120)
        }
        
        # If priority client, use injury recovery patterns
        if self.is_priority:
            self.current_state = 'INJURY_RECOVERY'
        
        # Statistics tracking
        self.stats = {
            'packets_sent': 0,
            'emergency_alerts': 0,
            'bytes_sent': 0,
            'errors': 0
        }
        
        # Threading
        self.running = False
        self.send_thread = None
        
        # UDP sockets
        self.sock_heartrate = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_steps = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_workout = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_emergency = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        print(f"[{self.client_name}] Wristband simulator initialized")
        print(f"[{self.client_name}] Priority client: {self.is_priority}")
        print(f"[{self.client_name}] Dashboard: {self.dashboard_ip}")
    
    def _generate_heart_rate(self) -> tuple:
        """
        Generate realistic heart rate with optional spike.
        
        Returns:
            Tuple of (heart_rate, is_emergency)
        """
        state = 'INJURY_RECOVERY' if self.is_priority else self.current_state
        params = self.HR_PARAMS[state]
        
        # Generate base heart rate with temporal smoothing
        target_hr = np.random.normal(params['mean'], params['std'])
        target_hr = np.clip(target_hr, params['min'], params['max'])
        
        # Smooth transition from last heart rate
        smoothing_factor = 0.3
        heart_rate = int(self.last_heart_rate * (1 - smoothing_factor) + 
                        target_hr * smoothing_factor)
        
        is_emergency = False
        
        # Check for random spike
        if self.enable_spikes and random.random() < self.spike_probability:
            # Generate spike - dangerous heart rate
            spike_hr = random.randint(180, 210)
            heart_rate = spike_hr
            is_emergency = True
            print(f"[{self.client_name}] ⚠️  HEART RATE SPIKE: {heart_rate} BPM!")
        
        self.last_heart_rate = heart_rate
        return heart_rate, is_emergency
    
    def _generate_steps(self) -> int:
        """Generate steps based on current activity state."""
        state = 'INJURY_RECOVERY' if self.is_priority else self.current_state
        params = self.STEPS_PARAMS[state]
        
        # Steps per interval (converting from per minute)
        steps_per_second = np.random.normal(params['mean'], params['std']) / 60
        steps = int(max(0, steps_per_second * self.data_interval))
        
        self.total_steps += steps
        return self.total_steps
    
    def _get_workout_duration(self) -> float:
        """Get current workout duration in seconds."""
        if self.workout_start_time is None:
            return 0.0
        return time.time() - self.workout_start_time
    
    def _update_activity_state(self):
        """Update activity state based on timing."""
        if self.is_priority:
            return  # Priority client stays in INJURY_RECOVERY
        
        self.state_duration += self.data_interval
        
        min_duration, max_duration = self.state_durations[self.current_state]
        target_duration = random.uniform(min_duration, max_duration)
        
        if self.state_duration >= target_duration:
            # Transition to next state
            current_idx = self.STATES.index(self.current_state)
            next_idx = (current_idx + 1) % len(self.STATES)
            self.current_state = self.STATES[next_idx]
            self.state_duration = 0
            
            print(f"[{self.client_name}] State changed to: {self.current_state}")
    
    def _send_data(self, data: FitnessData):
        """Send fitness data to dashboard."""
        json_data = json.dumps(asdict(data)).encode()
        
        try:
            # Send heart rate data
            self.sock_heartrate.sendto(
                json_data,
                (self.dashboard_ip, self.PORT_HEARTRATE)
            )
            
            # Send steps data
            self.sock_steps.sendto(
                json_data,
                (self.dashboard_ip, self.PORT_STEPS)
            )
            
            # Send workout duration data
            self.sock_workout.sendto(
                json_data,
                (self.dashboard_ip, self.PORT_WORKOUT)
            )
            
            # If emergency, send on emergency port
            if data.is_emergency:
                self.sock_emergency.sendto(
                    json_data,
                    (self.dashboard_ip, self.PORT_EMERGENCY)
                )
                self.stats['emergency_alerts'] += 1
            
            self.stats['packets_sent'] += 3 + (1 if data.is_emergency else 0)
            self.stats['bytes_sent'] += len(json_data) * (3 + (1 if data.is_emergency else 0))
            
        except Exception as e:
            self.stats['errors'] += 1
            print(f"[{self.client_name}] Error sending data: {e}")
    
    def _run_simulation(self):
        """Main simulation loop."""
        self.workout_start_time = time.time()
        
        print(f"[{self.client_name}] Starting simulation...")
        print(f"[{self.client_name}] Sending data every {self.data_interval}s")
        
        while self.running:
            # Update activity state
            self._update_activity_state()
            
            # Generate data
            heart_rate, is_emergency = self._generate_heart_rate()
            steps = self._generate_steps()
            workout_duration = self._get_workout_duration()
            
            # Create fitness data object
            data = FitnessData(
                timestamp=datetime.now().isoformat(),
                client_id=self.client_id,
                client_name=self.client_name,
                heart_rate=heart_rate,
                steps=steps,
                workout_duration=workout_duration,
                activity_state='INJURY_RECOVERY' if self.is_priority else self.current_state,
                is_emergency=is_emergency,
                emergency_type='HEART_RATE_SPIKE' if is_emergency else None
            )
            
            # Send data
            self._send_data(data)
            
            # Log periodically
            if self.stats['packets_sent'] % 30 == 0:
                self._log_status(data)
            
            # Wait for next interval
            time.sleep(self.data_interval)
    
    def _log_status(self, data: FitnessData):
        """Log current status."""
        priority_str = "[PRIORITY] " if self.is_priority else ""
        print(f"[{self.client_name}] {priority_str}"
              f"HR: {data.heart_rate} BPM | "
              f"Steps: {data.steps} | "
              f"State: {data.activity_state} | "
              f"Packets: {self.stats['packets_sent']}")
    
    def start(self):
        """Start the simulation."""
        self.running = True
        self.send_thread = threading.Thread(target=self._run_simulation)
        self.send_thread.daemon = True
        self.send_thread.start()
    
    def stop(self):
        """Stop the simulation."""
        self.running = False
        if self.send_thread:
            self.send_thread.join(timeout=2)
        
        # Close sockets
        self.sock_heartrate.close()
        self.sock_steps.close()
        self.sock_workout.close()
        self.sock_emergency.close()
        
        self._print_summary()
    
    def _print_summary(self):
        """Print simulation summary."""
        print("\n" + "="*50)
        print(f"[{self.client_name}] SIMULATION SUMMARY")
        print("="*50)
        print(f"  Total Packets Sent: {self.stats['packets_sent']}")
        print(f"  Emergency Alerts: {self.stats['emergency_alerts']}")
        print(f"  Total Bytes Sent: {self.stats['bytes_sent']}")
        print(f"  Errors: {self.stats['errors']}")
        print(f"  Total Steps: {self.total_steps}")
        duration = self._get_workout_duration()
        print(f"  Workout Duration: {duration:.1f} seconds")
        print("="*50 + "\n")
    
    def get_stats(self) -> dict:
        """Get current statistics."""
        return {
            'client_id': self.client_id,
            'client_name': self.client_name,
            'is_priority': self.is_priority,
            'current_state': self.current_state,
            'stats': self.stats.copy(),
            'total_steps': self.total_steps,
            'workout_duration': self._get_workout_duration()
        }


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\nReceived interrupt signal. Stopping simulation...")
    sys.exit(0)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Fitness Wristband Simulator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    # Run as Client 1 (priority - injury recovery)
    python3 wristband_client.py --client-id 1 --priority --dashboard 10.0.2.1
    
    # Run as Client 2 (normal)
    python3 wristband_client.py --client-id 2 --dashboard 10.0.2.1
    
    # Run with custom interval and spike probability
    python3 wristband_client.py --client-id 3 --dashboard 10.0.2.1 \\
        --interval 0.5 --spike-prob 0.05
        '''
    )
    
    parser.add_argument('--client-id', type=int, required=True,
                        choices=[1, 2, 3, 4],
                        help='Client ID (1-4)')
    parser.add_argument('--dashboard', type=str, default='10.0.2.1',
                        help='Dashboard server IP address')
    parser.add_argument('--priority', action='store_true',
                        help='Mark as priority client (injury recovery)')
    parser.add_argument('--interval', type=float, default=1.0,
                        help='Data transmission interval in seconds')
    parser.add_argument('--spike-prob', type=float, default=0.02,
                        help='Probability of heart rate spike per interval')
    parser.add_argument('--no-spikes', action='store_true',
                        help='Disable random heart rate spikes')
    parser.add_argument('--duration', type=int, default=0,
                        help='Simulation duration in seconds (0 = infinite)')
    
    args = parser.parse_args()
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and start simulator
    simulator = WristbandSimulator(
        client_id=args.client_id,
        dashboard_ip=args.dashboard,
        is_priority=args.priority,
        data_interval=args.interval,
        enable_spikes=not args.no_spikes,
        spike_probability=args.spike_prob
    )
    
    print("\n" + "="*50)
    print("FITNESS WRISTBAND SIMULATOR")
    print("="*50)
    print(f"Client ID: {args.client_id}")
    print(f"Priority: {'Yes (Injury Recovery)' if args.priority else 'No'}")
    print(f"Dashboard: {args.dashboard}")
    print(f"Interval: {args.interval}s")
    print(f"Spike Probability: {args.spike_prob}")
    print("="*50)
    print("\nPress Ctrl+C to stop...\n")
    
    simulator.start()
    
    try:
        if args.duration > 0:
            time.sleep(args.duration)
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        simulator.stop()


if __name__ == '__main__':
    main()
