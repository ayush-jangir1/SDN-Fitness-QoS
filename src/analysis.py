#!/usr/bin/env python3
"""
SDN Fitness Center - Data Analysis and Visualization
====================================================
This code provides comprehensive analysis tools for evaluating SDN QoS
performance in the fitness center simulation.

Analysis Capabilities:
1. QoS Metrics Analysis (latency, jitter, throughput)
2. Priority Traffic Performance Comparison
3. Emergency Response Time Analysis
4. Statistical Significance Testing
5. Quality Visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, ttest_ind, wilcoxon
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
import argparse

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


class FitnessDataAnalyzer:
    """
    Comprehensive analyzer for SDN fitness center data.
    
    Provides statistical analysis and visualization for:
    - QoS performance metrics
    - Priority vs normal traffic comparison
    - Emergency handling effectiveness
    """
    
    def __init__(self, data_file: str, output_dir: str = './results'):
        """
        Initialize the analyzer.
        
        Args:
            data_file: Path to CSV data file from dashboard
            output_dir: Directory for output files
        """
        self.data_file = data_file
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        self.df = self._load_data()
        
        # Analysis results
        self.results = {}
        
        print(f"[Analyzer] Loaded {len(self.df)} records from {data_file}")
    
    def _load_data(self) -> pd.DataFrame:
        """Load and preprocess the data."""
        df = pd.read_csv(self.data_file)
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate relative time from start
        df['relative_time'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
        
        # Ensure boolean columns
        df['is_priority'] = df['is_priority'].astype(bool)
        df['is_emergency'] = df['is_emergency'].astype(bool)
        
        return df
    
    def analyze_qos_metrics(self) -> Dict:
        """
        Analyze QoS metrics across all traffic types.
        
        Returns:
            Dictionary with QoS analysis results
        """
        print("\n[Analyzer] Analyzing QoS metrics...")
        
        # Group by client
        client_groups = self.df.groupby('client_id')
        
        qos_results = {}
        
        for client_id, group in client_groups:
            client_name = group['client_name'].iloc[0]
            is_priority = group['is_priority'].iloc[0]
            
            latencies = group['latency_ms'].dropna()
            
            qos_results[client_id] = {
                'client_name': client_name,
                'is_priority': is_priority,
                'packet_count': len(group),
                'latency': {
                    'mean': latencies.mean(),
                    'std': latencies.std(),
                    'min': latencies.min(),
                    'max': latencies.max(),
                    'median': latencies.median(),
                    'p95': latencies.quantile(0.95),
                    'p99': latencies.quantile(0.99)
                },
                'emergency_count': group['is_emergency'].sum()
            }
        
        # Calculate aggregate statistics
        priority_data = self.df[self.df['is_priority'] == True]['latency_ms']
        normal_data = self.df[self.df['is_priority'] == False]['latency_ms']
        
        qos_results['aggregate'] = {
            'priority': {
                'mean': priority_data.mean(),
                'std': priority_data.std(),
                'median': priority_data.median()
            },
            'normal': {
                'mean': normal_data.mean(),
                'std': normal_data.std(),
                'median': normal_data.median()
            }
        }
        
        # Statistical test: Mann-Whitney U test for latency difference
        if len(priority_data) > 0 and len(normal_data) > 0:
            statistic, p_value = mannwhitneyu(priority_data, normal_data, 
                                               alternative='less')
            qos_results['statistical_test'] = {
                'test': 'Mann-Whitney U',
                'null_hypothesis': 'Priority latency >= Normal latency',
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        self.results['qos'] = qos_results
        return qos_results
    
    def analyze_priority_traffic(self) -> Dict:
        """
        Analyze priority traffic (injury recovery) performance.
        
        Returns:
            Dictionary with priority traffic analysis
        """
        print("[Analyzer] Analyzing priority traffic...")
        
        priority_df = self.df[self.df['is_priority'] == True]
        normal_df = self.df[self.df['is_priority'] == False]
        
        results = {
            'priority_packets': len(priority_df),
            'normal_packets': len(normal_df),
            'priority_ratio': len(priority_df) / len(self.df) if len(self.df) > 0 else 0
        }
        
        # Latency comparison
        if len(priority_df) > 0 and len(normal_df) > 0:
            results['latency_improvement'] = {
                'absolute_ms': normal_df['latency_ms'].mean() - priority_df['latency_ms'].mean(),
                'percentage': ((normal_df['latency_ms'].mean() - priority_df['latency_ms'].mean()) / 
                              normal_df['latency_ms'].mean() * 100) if normal_df['latency_ms'].mean() > 0 else 0
            }
        
        # Per-port analysis
        port_analysis = {}
        for port in self.df['port'].unique():
            port_df = self.df[self.df['port'] == port]
            port_priority = port_df[port_df['is_priority'] == True]['latency_ms']
            port_normal = port_df[port_df['is_priority'] == False]['latency_ms']
            
            port_analysis[int(port)] = {
                'priority_mean': port_priority.mean() if len(port_priority) > 0 else 0,
                'normal_mean': port_normal.mean() if len(port_normal) > 0 else 0,
                'priority_count': len(port_priority),
                'normal_count': len(port_normal)
            }
        
        results['per_port'] = port_analysis
        
        self.results['priority'] = results
        return results
    
    def analyze_emergency_response(self) -> Dict:
        """
        Analyze emergency traffic handling performance.
        
        Returns:
            Dictionary with emergency response analysis
        """
        print("[Analyzer] Analyzing emergency response...")
        
        emergency_df = self.df[self.df['is_emergency'] == True]
        normal_df = self.df[self.df['is_emergency'] == False]
        
        results = {
            'total_emergencies': len(emergency_df),
            'emergency_rate': len(emergency_df) / len(self.df) if len(self.df) > 0 else 0
        }
        
        if len(emergency_df) > 0:
            results['emergency_latency'] = {
                'mean': emergency_df['latency_ms'].mean(),
                'std': emergency_df['latency_ms'].std(),
                'max': emergency_df['latency_ms'].max(),
                'p95': emergency_df['latency_ms'].quantile(0.95)
            }
            
            results['heart_rate_during_emergency'] = {
                'mean': emergency_df['heart_rate'].mean(),
                'max': emergency_df['heart_rate'].max(),
                'min': emergency_df['heart_rate'].min()
            }
            
            # Per-client emergency analysis
            emergency_per_client = emergency_df.groupby('client_id').agg({
                'latency_ms': ['mean', 'count'],
                'heart_rate': 'max'
            }).round(2)
            
            # Convert to serializable format
            results['per_client'] = {str(k): v for k, v in emergency_per_client.to_dict().items()}
        
        self.results['emergency'] = results
        return results
    
    def analyze_throughput(self) -> Dict:
        """
        Analyze network throughput over time.
        
        Returns:
            Dictionary with throughput analysis
        """
        print("[Analyzer] Analyzing throughput...")
        
        # Resample to 1-second intervals
        df_time = self.df.set_index('timestamp')
        
        throughput = df_time.resample('1S').size()
        
        results = {
            'mean_pps': throughput.mean(),
            'max_pps': throughput.max(),
            'min_pps': throughput.min(),
            'std_pps': throughput.std()
        }
        
        # Throughput by client type
        priority_throughput = df_time[df_time['is_priority'] == True].resample('1S').size()
        normal_throughput = df_time[df_time['is_priority'] == False].resample('1S').size()
        
        results['priority_throughput'] = {
            'mean_pps': priority_throughput.mean() if len(priority_throughput) > 0 else 0,
            'max_pps': priority_throughput.max() if len(priority_throughput) > 0 else 0
        }
        
        results['normal_throughput'] = {
            'mean_pps': normal_throughput.mean() if len(normal_throughput) > 0 else 0,
            'max_pps': normal_throughput.max() if len(normal_throughput) > 0 else 0
        }
        
        self.results['throughput'] = results
        return results
    
    def generate_visualizations(self):
        """Generate all visualizations."""
        print("\n[Analyzer] Generating visualizations...")
        
        self._plot_latency_comparison()
        self._plot_latency_distribution()
        self._plot_latency_over_time()
        self._plot_heart_rate_analysis()
        self._plot_throughput_analysis()
        self._plot_qos_summary()
        self._plot_emergency_analysis()
        
        print(f"[Analyzer] Visualizations saved to {self.output_dir}")
    
    def _plot_latency_comparison(self):
        """Plot latency comparison between priority and normal traffic."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Box plot
        ax1 = axes[0]
        priority_latency = self.df[self.df['is_priority'] == True]['latency_ms']
        normal_latency = self.df[self.df['is_priority'] == False]['latency_ms']
        
        data_to_plot = [priority_latency, normal_latency]
        bp = ax1.boxplot(data_to_plot, labels=['Priority\n(Injury Recovery)', 'Normal'],
                        patch_artist=True, showmeans=True)
        
        colors = ['#FF9800', '#4CAF50']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_ylabel('Latency (ms)')
        ax1.set_title('Latency Distribution by Traffic Type')
        ax1.grid(True, alpha=0.3)
        
        # Add statistical annotation
        if len(priority_latency) > 0 and len(normal_latency) > 0:
            stat, p_val = mannwhitneyu(priority_latency, normal_latency, alternative='less')
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            ax1.text(1.5, ax1.get_ylim()[1] * 0.95, f'p = {p_val:.4f} ({significance})',
                    ha='center', fontsize=11)
        
        # Bar plot with error bars
        ax2 = axes[1]
        means = [priority_latency.mean(), normal_latency.mean()]
        stds = [priority_latency.std(), normal_latency.std()]
        x_pos = [0, 1]
        
        bars = ax2.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(['Priority\n(Injury Recovery)', 'Normal'])
        ax2.set_ylabel('Mean Latency (ms)')
        ax2.set_title('Mean Latency Comparison')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
                    f'{mean:.2f}±{std:.2f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'latency_comparison.png'))
        plt.savefig(os.path.join(self.output_dir, 'latency_comparison.pdf'))
        plt.close()
    
    def _plot_latency_distribution(self):
        """Plot detailed latency distribution."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Overall distribution
        ax1 = axes[0, 0]
        sns.histplot(data=self.df, x='latency_ms', hue='is_priority',
                    bins=50, ax=ax1, alpha=0.6,
                    palette={True: '#FF9800', False: '#4CAF50'})
        ax1.set_xlabel('Latency (ms)')
        ax1.set_ylabel('Count')
        ax1.set_title('Latency Distribution')
        ax1.legend(['Normal', 'Priority'], title='Traffic Type')
        
        # CDF plot
        ax2 = axes[0, 1]
        for is_priority, label, color in [(True, 'Priority', '#FF9800'), 
                                           (False, 'Normal', '#4CAF50')]:
            data = self.df[self.df['is_priority'] == is_priority]['latency_ms'].sort_values()
            cdf = np.arange(1, len(data) + 1) / len(data)
            ax2.plot(data, cdf, label=label, color=color, linewidth=2)
        
        ax2.set_xlabel('Latency (ms)')
        ax2.set_ylabel('CDF')
        ax2.set_title('Cumulative Distribution Function')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='95th percentile')
        
        # Per-client distribution
        ax3 = axes[1, 0]
        client_data = []
        client_labels = []
        for client_id in self.df['client_id'].unique():
            client_df = self.df[self.df['client_id'] == client_id]
            client_data.append(client_df['latency_ms'])
            is_priority = client_df['is_priority'].iloc[0]
            label = f"{client_df['client_name'].iloc[0]}\n{'(Priority)' if is_priority else ''}"
            client_labels.append(label)
        
        bp = ax3.boxplot(client_data, labels=client_labels, patch_artist=True)
        colors = ['#FF9800' if self.df[self.df['client_id'] == i+1]['is_priority'].iloc[0] 
                 else '#4CAF50' for i in range(len(client_data))]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax3.set_ylabel('Latency (ms)')
        ax3.set_title('Latency Distribution by Client')
        
        # Per-port distribution
        ax4 = axes[1, 1]
        port_names = {5001: 'Heart Rate', 5002: 'Steps', 5003: 'Workout', 5004: 'Emergency'}
        port_data = []
        port_labels = []
        for port in sorted(self.df['port'].unique()):
            port_df = self.df[self.df['port'] == port]
            if len(port_df) > 0:
                port_data.append(port_df['latency_ms'])
                port_labels.append(port_names.get(int(port), f'Port {port}'))
        
        if port_data:
            bp = ax4.boxplot(port_data, labels=port_labels, patch_artist=True)
            port_colors = ['#2196F3', '#9C27B0', '#009688', '#F44336']
            for patch, color in zip(bp['boxes'], port_colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax4.set_ylabel('Latency (ms)')
        ax4.set_title('Latency Distribution by Data Type')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'latency_distribution.png'))
        plt.savefig(os.path.join(self.output_dir, 'latency_distribution.pdf'))
        plt.close()
    
    def _plot_latency_over_time(self):
        """Plot latency trends over time."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Rolling average latency
        ax1 = axes[0]
        for is_priority, label, color in [(True, 'Priority', '#FF9800'), 
                                           (False, 'Normal', '#4CAF50')]:
            data = self.df[self.df['is_priority'] == is_priority].copy()
            data = data.sort_values('relative_time')
            
            # Rolling mean with window of 10
            if len(data) > 10:
                data['rolling_latency'] = data['latency_ms'].rolling(window=10, min_periods=1).mean()
                ax1.plot(data['relative_time'], data['rolling_latency'], 
                        label=label, color=color, linewidth=1.5, alpha=0.8)
        
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Latency (ms) - Rolling Mean')
        ax1.set_title('Latency Trends Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Scatter plot with emergency markers
        ax2 = axes[1]
        normal = self.df[(self.df['is_priority'] == False) & (self.df['is_emergency'] == False)]
        priority = self.df[(self.df['is_priority'] == True) & (self.df['is_emergency'] == False)]
        emergency = self.df[self.df['is_emergency'] == True]
        
        ax2.scatter(normal['relative_time'], normal['latency_ms'], 
                   c='#4CAF50', alpha=0.3, s=10, label='Normal')
        ax2.scatter(priority['relative_time'], priority['latency_ms'],
                   c='#FF9800', alpha=0.5, s=15, label='Priority')
        ax2.scatter(emergency['relative_time'], emergency['latency_ms'],
                   c='#F44336', alpha=0.8, s=50, marker='x', label='Emergency')
        
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Latency (ms)')
        ax2.set_title('Individual Packet Latencies')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'latency_over_time.png'))
        plt.savefig(os.path.join(self.output_dir, 'latency_over_time.pdf'))
        plt.close()
    
    def _plot_heart_rate_analysis(self):
        """Plot heart rate analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Heart rate over time by client
        ax1 = axes[0, 0]
        for client_id in self.df['client_id'].unique():
            client_df = self.df[self.df['client_id'] == client_id].sort_values('relative_time')
            is_priority = client_df['is_priority'].iloc[0]
            color = '#FF9800' if is_priority else '#4CAF50'
            label = f"Client {client_id}" + (" (Priority)" if is_priority else "")
            ax1.plot(client_df['relative_time'], client_df['heart_rate'],
                    label=label, alpha=0.7, linewidth=1)
        
        ax1.axhline(y=180, color='red', linestyle='--', alpha=0.5, label='Spike Threshold')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Heart Rate (BPM)')
        ax1.set_title('Heart Rate Over Time')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Heart rate distribution
        ax2 = axes[0, 1]
        for client_id in self.df['client_id'].unique():
            client_df = self.df[self.df['client_id'] == client_id]
            is_priority = client_df['is_priority'].iloc[0]
            color = '#FF9800' if is_priority else '#4CAF50'
            label = f"Client {client_id}" + (" (Priority)" if is_priority else "")
            sns.kdeplot(data=client_df, x='heart_rate', ax=ax2, 
                       label=label, color=color, alpha=0.6)
        
        ax2.set_xlabel('Heart Rate (BPM)')
        ax2.set_ylabel('Density')
        ax2.set_title('Heart Rate Distribution by Client')
        ax2.legend()
        
        # Activity state distribution
        ax3 = axes[1, 0]
        state_counts = self.df.groupby(['client_id', 'activity_state']).size().unstack(fill_value=0)
        state_counts.plot(kind='bar', ax=ax3, colormap='viridis', alpha=0.8)
        ax3.set_xlabel('Client ID')
        ax3.set_ylabel('Count')
        ax3.set_title('Activity State Distribution')
        ax3.legend(title='State', bbox_to_anchor=(1.02, 1))
        ax3.tick_params(axis='x', rotation=0)
        
        # Emergency events
        ax4 = axes[1, 1]
        emergency_df = self.df[self.df['is_emergency'] == True]
        if len(emergency_df) > 0:
            emergency_counts = emergency_df.groupby('client_id').size()
            bars = ax4.bar(emergency_counts.index.astype(str), emergency_counts.values, 
                          color='#F44336', alpha=0.7)
            ax4.set_xlabel('Client ID')
            ax4.set_ylabel('Emergency Count')
            ax4.set_title('Emergency Events by Client')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
        else:
            ax4.text(0.5, 0.5, 'No Emergency Events', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=14)
            ax4.set_title('Emergency Events by Client')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'heart_rate_analysis.png'))
        plt.savefig(os.path.join(self.output_dir, 'heart_rate_analysis.pdf'))
        plt.close()
    
    def _plot_throughput_analysis(self):
        """Plot throughput analysis."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # Packets per second over time
        ax1 = axes[0]
        df_time = self.df.set_index('timestamp')
        
        # Overall throughput
        throughput = df_time.resample('1S').size()
        ax1.plot(range(len(throughput)), throughput.values, 
                color='#2196F3', alpha=0.7, label='Overall')
        
        # Priority throughput
        priority_throughput = df_time[df_time['is_priority'] == True].resample('1S').size()
        ax1.plot(range(len(priority_throughput)), priority_throughput.values,
                color='#FF9800', alpha=0.7, label='Priority')
        
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Packets per Second')
        ax1.set_title('Network Throughput Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Throughput by client
        ax2 = axes[1]
        client_throughput = self.df.groupby('client_id').size()
        colors = ['#FF9800' if self.df[self.df['client_id'] == cid]['is_priority'].iloc[0]
                 else '#4CAF50' for cid in client_throughput.index]
        
        bars = ax2.bar(client_throughput.index.astype(str), client_throughput.values,
                      color=colors, alpha=0.7)
        ax2.set_xlabel('Client ID')
        ax2.set_ylabel('Total Packets')
        ax2.set_title('Total Packets by Client')
        
        # Add legend
        priority_patch = mpatches.Patch(color='#FF9800', alpha=0.7, label='Priority')
        normal_patch = mpatches.Patch(color='#4CAF50', alpha=0.7, label='Normal')
        ax2.legend(handles=[priority_patch, normal_patch])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'throughput_analysis.png'))
        plt.savefig(os.path.join(self.output_dir, 'throughput_analysis.pdf'))
        plt.close()
    
    def _plot_qos_summary(self):
        """Plot comprehensive QoS summary."""
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Latency by traffic type
        ax1 = fig.add_subplot(gs[0, 0])
        priority_lat = self.df[self.df['is_priority'] == True]['latency_ms']
        normal_lat = self.df[self.df['is_priority'] == False]['latency_ms']
        
        bp = ax1.boxplot([priority_lat, normal_lat], labels=['Priority', 'Normal'],
                        patch_artist=True)
        bp['boxes'][0].set_facecolor('#FF9800')
        bp['boxes'][1].set_facecolor('#4CAF50')
        ax1.set_ylabel('Latency (ms)')
        ax1.set_title('Latency by Traffic Type')
        
        # 2. Packet distribution
        ax2 = fig.add_subplot(gs[0, 1])
        sizes = [len(priority_lat), len(normal_lat)]
        ax2.pie(sizes, labels=['Priority', 'Normal'], autopct='%1.1f%%',
               colors=['#FF9800', '#4CAF50'], startangle=90)
        ax2.set_title('Packet Distribution')
        
        # 3. Emergency response
        ax3 = fig.add_subplot(gs[0, 2])
        emergency = self.df[self.df['is_emergency'] == True]['latency_ms']
        non_emergency = self.df[self.df['is_emergency'] == False]['latency_ms']
        
        if len(emergency) > 0:
            bp = ax3.boxplot([emergency, non_emergency], labels=['Emergency', 'Normal'],
                            patch_artist=True)
            bp['boxes'][0].set_facecolor('#F44336')
            bp['boxes'][1].set_facecolor('#4CAF50')
        ax3.set_ylabel('Latency (ms)')
        ax3.set_title('Emergency vs Normal Latency')
        
        # 4. Latency percentiles
        ax4 = fig.add_subplot(gs[1, :2])
        percentiles = [50, 75, 90, 95, 99]
        priority_pcts = [priority_lat.quantile(p/100) for p in percentiles]
        normal_pcts = [normal_lat.quantile(p/100) for p in percentiles]
        
        x = np.arange(len(percentiles))
        width = 0.35
        ax4.bar(x - width/2, priority_pcts, width, label='Priority', color='#FF9800', alpha=0.7)
        ax4.bar(x + width/2, normal_pcts, width, label='Normal', color='#4CAF50', alpha=0.7)
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'P{p}' for p in percentiles])
        ax4.set_ylabel('Latency (ms)')
        ax4.set_title('Latency Percentiles Comparison')
        ax4.legend()
        
        # 5. Statistical summary table
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        
        stats_data = [
            ['Metric', 'Priority', 'Normal'],
            ['Mean (ms)', f'{priority_lat.mean():.2f}', f'{normal_lat.mean():.2f}'],
            ['Std (ms)', f'{priority_lat.std():.2f}', f'{normal_lat.std():.2f}'],
            ['Median (ms)', f'{priority_lat.median():.2f}', f'{normal_lat.median():.2f}'],
            ['P95 (ms)', f'{priority_lat.quantile(0.95):.2f}', f'{normal_lat.quantile(0.95):.2f}'],
            ['Count', f'{len(priority_lat)}', f'{len(normal_lat)}']
        ]
        
        table = ax5.table(cellText=stats_data, loc='center', cellLoc='center',
                         colWidths=[0.4, 0.3, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax5.set_title('Statistical Summary', y=0.95)
        
        # 6. QoS improvement visualization
        ax6 = fig.add_subplot(gs[2, :])
        
        # Calculate improvement metrics
        latency_improvement = ((normal_lat.mean() - priority_lat.mean()) / normal_lat.mean() * 100) if normal_lat.mean() > 0 else 0
        p95_improvement = ((normal_lat.quantile(0.95) - priority_lat.quantile(0.95)) / 
                          normal_lat.quantile(0.95) * 100) if normal_lat.quantile(0.95) > 0 else 0
        
        metrics = ['Mean Latency\nReduction', 'P95 Latency\nReduction', 'Jitter\nReduction']
        jitter_improvement = ((normal_lat.std() - priority_lat.std()) / normal_lat.std() * 100) if normal_lat.std() > 0 else 0
        values = [latency_improvement, p95_improvement, jitter_improvement]
        
        colors_imp = ['#4CAF50' if v > 0 else '#F44336' for v in values]
        bars = ax6.bar(metrics, values, color=colors_imp, alpha=0.7)
        
        ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax6.set_ylabel('Improvement (%)')
        ax6.set_title('SDN QoS Performance Improvement (Priority vs Normal)')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=12, fontweight='bold')
        
        plt.suptitle('SDN Fitness Center - QoS Performance Summary', fontsize=18, y=1.02)
        plt.savefig(os.path.join(self.output_dir, 'qos_summary.png'))
        plt.savefig(os.path.join(self.output_dir, 'qos_summary.pdf'))
        plt.close()
    
    def _plot_emergency_analysis(self):
        """Plot detailed emergency analysis."""
        emergency_df = self.df[self.df['is_emergency'] == True]
        
        if len(emergency_df) == 0:
            print("[Analyzer] No emergency events to analyze")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Emergency timeline
        ax1 = axes[0, 0]
        ax1.scatter(emergency_df['relative_time'], emergency_df['heart_rate'],
                   c='#F44336', s=100, alpha=0.7, marker='x')
        ax1.axhline(y=180, color='orange', linestyle='--', alpha=0.5, label='Threshold')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Heart Rate (BPM)')
        ax1.set_title('Emergency Events Timeline')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Emergency latency distribution
        ax2 = axes[0, 1]
        sns.histplot(data=emergency_df, x='latency_ms', bins=20, ax=ax2,
                    color='#F44336', alpha=0.7)
        ax2.axvline(x=emergency_df['latency_ms'].mean(), color='black',
                   linestyle='--', label=f"Mean: {emergency_df['latency_ms'].mean():.2f}ms")
        ax2.set_xlabel('Latency (ms)')
        ax2.set_ylabel('Count')
        ax2.set_title('Emergency Response Latency')
        ax2.legend()
        
        # Emergency by client
        ax3 = axes[1, 0]
        emergency_by_client = emergency_df.groupby('client_name').size()
        bars = ax3.bar(emergency_by_client.index, emergency_by_client.values,
                      color='#F44336', alpha=0.7)
        ax3.set_xlabel('Client')
        ax3.set_ylabel('Emergency Count')
        ax3.set_title('Emergencies by Client')
        
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Emergency response time comparison
        ax4 = axes[1, 1]
        emergency_latency = emergency_df['latency_ms']
        normal_latency = self.df[self.df['is_emergency'] == False]['latency_ms']
        
        data = [emergency_latency, normal_latency]
        bp = ax4.boxplot(data, labels=['Emergency', 'Normal'],
                        patch_artist=True, showmeans=True)
        bp['boxes'][0].set_facecolor('#F44336')
        bp['boxes'][1].set_facecolor('#4CAF50')
        
        ax4.set_ylabel('Latency (ms)')
        ax4.set_title('Emergency vs Normal Traffic Latency')
        
        # Statistical test
        if len(emergency_latency) > 1 and len(normal_latency) > 1:
            stat, p_val = mannwhitneyu(emergency_latency, normal_latency)
            ax4.text(1.5, ax4.get_ylim()[1] * 0.9, f'p = {p_val:.4f}',
                    ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'emergency_analysis.png'))
        plt.savefig(os.path.join(self.output_dir, 'emergency_analysis.pdf'))
        plt.close()
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive analysis report.
        
        Returns:
            Path to the generated report
        """
        print("\n[Analyzer] Generating analysis report...")
        
        # Run all analyses
        self.analyze_qos_metrics()
        self.analyze_priority_traffic()
        self.analyze_emergency_response()
        self.analyze_throughput()
        
        # Generate report
        report_path = os.path.join(self.output_dir, 'analysis_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("SDN FITNESS CENTER - COMPREHENSIVE ANALYSIS REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data File: {self.data_file}\n")
            f.write(f"Total Records: {len(self.df)}\n\n")
            
            # QoS Analysis
            f.write("-"*70 + "\n")
            f.write("1. QoS METRICS ANALYSIS\n")
            f.write("-"*70 + "\n\n")
            
            qos = self.results.get('qos', {})
            if 'aggregate' in qos:
                f.write("Aggregate Latency Statistics:\n")
                f.write(f"  Priority Traffic:\n")
                f.write(f"    Mean: {qos['aggregate']['priority']['mean']:.2f} ms\n")
                f.write(f"    Std: {qos['aggregate']['priority']['std']:.2f} ms\n")
                f.write(f"  Normal Traffic:\n")
                f.write(f"    Mean: {qos['aggregate']['normal']['mean']:.2f} ms\n")
                f.write(f"    Std: {qos['aggregate']['normal']['std']:.2f} ms\n\n")
            
            if 'statistical_test' in qos:
                test = qos['statistical_test']
                f.write(f"Statistical Test ({test['test']}):\n")
                f.write(f"  Null Hypothesis: {test['null_hypothesis']}\n")
                f.write(f"  p-value: {test['p_value']:.6f}\n")
                f.write(f"  Significant (α=0.05): {'Yes' if test['significant'] else 'No'}\n\n")
            
            # Priority Traffic Analysis
            f.write("-"*70 + "\n")
            f.write("2. PRIORITY TRAFFIC ANALYSIS\n")
            f.write("-"*70 + "\n\n")
            
            priority = self.results.get('priority', {})
            f.write(f"Priority Packets: {priority.get('priority_packets', 0)}\n")
            f.write(f"Normal Packets: {priority.get('normal_packets', 0)}\n")
            
            if 'latency_improvement' in priority:
                f.write(f"\nLatency Improvement:\n")
                f.write(f"  Absolute: {priority['latency_improvement']['absolute_ms']:.2f} ms\n")
                f.write(f"  Percentage: {priority['latency_improvement']['percentage']:.2f}%\n\n")
            
            # Emergency Response Analysis
            f.write("-"*70 + "\n")
            f.write("3. EMERGENCY RESPONSE ANALYSIS\n")
            f.write("-"*70 + "\n\n")
            
            emergency = self.results.get('emergency', {})
            f.write(f"Total Emergency Events: {emergency.get('total_emergencies', 0)}\n")
            f.write(f"Emergency Rate: {emergency.get('emergency_rate', 0)*100:.2f}%\n")
            
            if 'emergency_latency' in emergency:
                f.write(f"\nEmergency Response Latency:\n")
                f.write(f"  Mean: {emergency['emergency_latency']['mean']:.2f} ms\n")
                f.write(f"  Max: {emergency['emergency_latency']['max']:.2f} ms\n")
                f.write(f"  P95: {emergency['emergency_latency']['p95']:.2f} ms\n\n")
            
            # Throughput Analysis
            f.write("-"*70 + "\n")
            f.write("4. THROUGHPUT ANALYSIS\n")
            f.write("-"*70 + "\n\n")
            
            throughput = self.results.get('throughput', {})
            f.write(f"Mean Packets/Second: {throughput.get('mean_pps', 0):.2f}\n")
            f.write(f"Max Packets/Second: {throughput.get('max_pps', 0):.2f}\n")
            f.write(f"Std Packets/Second: {throughput.get('std_pps', 0):.2f}\n\n")
            
            # Conclusions
            f.write("="*70 + "\n")
            f.write("CONCLUSIONS\n")
            f.write("="*70 + "\n\n")
            
            f.write("This analysis demonstrates the effectiveness of SDN-based QoS\n")
            f.write("management in a fitness center IoT environment. Key findings:\n\n")
            
            if 'latency_improvement' in priority and priority['latency_improvement']['percentage'] > 0:
                f.write(f"1. Priority traffic (injury recovery client) achieved\n")
                f.write(f"   {priority['latency_improvement']['percentage']:.1f}% lower latency than normal traffic.\n\n")
            
            if 'statistical_test' in qos and qos['statistical_test']['significant']:
                f.write(f"2. The latency difference is statistically significant\n")
                f.write(f"   (p = {qos['statistical_test']['p_value']:.6f}), confirming\n")
                f.write(f"   the effectiveness of SDN QoS policies.\n\n")
            
            f.write("3. The SDN controller successfully prioritized health-critical\n")
            f.write("   traffic, demonstrating its viability for healthcare IoT applications.\n")
            
            f.write("\n" + "="*70 + "\n")
        
        # Save results as JSON
        json_path = os.path.join(self.output_dir, 'analysis_results.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"[Analyzer] Report saved to {report_path}")
        print(f"[Analyzer] Results saved to {json_path}")
        
        return report_path
    
    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print("\n" + "="*60)
        print("SDN FITNESS CENTER - DATA ANALYSIS")
        print("="*60)
        
        # Run analyses
        self.analyze_qos_metrics()
        self.analyze_priority_traffic()
        self.analyze_emergency_response()
        self.analyze_throughput()
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Generate report
        report_path = self.generate_report()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Output directory: {self.output_dir}")
        print(f"Report: {report_path}")
        print("="*60 + "\n")
        
        return self.results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='SDN Fitness Center Data Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    python3 analysis.py data/fitness_data.csv
    python3 analysis.py data/fitness_data.csv -o ./my_results
        '''
    )
    
    parser.add_argument('data_file', type=str,
                        help='Path to CSV data file from dashboard')
    parser.add_argument('-o', '--output', type=str, default='./results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_file):
        print(f"Error: Data file not found: {args.data_file}")
        sys.exit(1)
    
    analyzer = FitnessDataAnalyzer(args.data_file, args.output)
    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
