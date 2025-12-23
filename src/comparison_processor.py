"""
Complete Comparison Processor Module
File: src/comparison_processor.py

Processes video in edge, cloud, and hybrid modes and generates unified comparison plots
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from datetime import datetime
from typing import Dict, List
import json
import sys

# Ensure src directory is in path
sys.path.append('src')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from metrics import MetricsCollector


class ComparisonProcessor:
    """Process video in multiple modes and compare results"""
    
    def __init__(self, system, output_dir='data/results/comparison'):
        self.system = system
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = {}
        
    def process_all_modes(self, video_path, bandwidth=300):
        """Process video in edge, cloud, and hybrid modes"""
        
        modes = ['edge', 'cloud', 'hybrid']
        
        print("\n" + "="*70)
        print(f"MULTI-MODE COMPARISON: {os.path.basename(video_path)}")
        print("="*70 + "\n")
        
        for mode in modes:
            print(f"\n{'='*70}")
            print(f"Processing Mode: {mode.upper()}")
            print(f"{'='*70}\n")
            
            # Create subfolder for this mode
            mode_output = os.path.join(self.output_dir, mode)
            os.makedirs(mode_output, exist_ok=True)
            
            # Reset metrics collector for this mode
            self.system.metrics = MetricsCollector(mode_output)
            
            # Reset resource monitor if available
            if hasattr(self.system, 'resource_monitor'):
                self.system.resource_monitor = ResourceMonitor(interval=0.1)
            
            # Process video
            start_time = datetime.now()
            try:
                self.system.process_video(
                    video_path=video_path,
                    output_path=mode_output,
                    bandwidth=bandwidth,
                    mode=mode
                )
            except Exception as e:
                print(f"❌ Error processing {mode} mode: {e}")
                continue
            
            end_time = datetime.now()
            
            # Store results
            duration = (end_time - start_time).total_seconds()
            metrics_df = pd.DataFrame(self.system.metrics.metrics)
            
            self.results[mode] = {
                'duration': duration,
                'metrics': metrics_df,
                'summary': self.system.metrics.generate_summary(metrics_df)
            }
            
            print(f"\n✓ {mode.upper()} mode completed in {duration:.2f} seconds\n")
        
        if not self.results:
            print("❌ No modes were successfully processed")
            return
        
        # Generate unified comparison plots
        self._generate_unified_plots()
        self._generate_comparison_report()
        
        print("\n" + "="*70)
        print("MULTI-MODE COMPARISON COMPLETE")
        print("="*70 + "\n")
    
    def _generate_unified_plots(self):
        """Generate comprehensive comparison plots"""
        
        # Create large figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Row 1: Processing Time Comparison
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_processing_time_comparison(ax1)
        
        # Row 2: Vehicle Count, Speed, Congestion
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_vehicle_count_comparison(ax2)
        
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_speed_comparison(ax3)
        
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_congestion_comparison(ax4)
        
        # Row 3: Performance Metrics
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_avg_processing_time_bars(ax5)
        
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_accuracy_comparison(ax6)
        
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_total_duration(ax7)
        
        # Row 4: Node Distribution and Summary
        ax8 = fig.add_subplot(gs[3, 0])
        self._plot_node_distribution(ax8)
        
        ax9 = fig.add_subplot(gs[3, 1])
        self._plot_efficiency_metrics(ax9)
        
        ax10 = fig.add_subplot(gs[3, 2])
        self._plot_summary_table(ax10)
        
        # Add title
        fig.suptitle('Multi-Mode Comparison Analysis', fontsize=18, fontweight='bold', y=0.995)
        
        # Save figure
        filepath = os.path.join(self.output_dir, 'unified_comparison.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✓ Unified comparison plot saved to: {filepath}")
        plt.close()
        
        # Also generate individual comparison plots
        self._generate_individual_comparisons()
    
    def _plot_processing_time_comparison(self, ax):
        """Plot processing time over frames for all modes"""
        colors = {'edge': 'blue', 'cloud': 'red', 'hybrid': 'green'}
        
        for mode, data in self.results.items():
            df = data['metrics']
            # Use rolling average to smooth the plot
            rolling = df['processing_time_ms'].rolling(window=30, min_periods=1).mean()
            ax.plot(df['frame'], rolling, label=mode.capitalize(), 
                   color=colors[mode], linewidth=2, alpha=0.8)
        
        ax.set_title('Processing Time Comparison (30-frame rolling average)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Frame Number', fontsize=11)
        ax.set_ylabel('Processing Time (ms)', fontsize=11)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    
    def _plot_vehicle_count_comparison(self, ax):
        """Plot vehicle count over time"""
        colors = {'edge': 'blue', 'cloud': 'red', 'hybrid': 'green'}
        
        for mode, data in self.results.items():
            df = data['metrics']
            rolling = df['vehicle_count'].rolling(window=20, min_periods=1).mean()
            ax.plot(df['frame'], rolling, label=mode.capitalize(), 
                   color=colors[mode], linewidth=2, alpha=0.7)
        
        ax.set_title('Vehicle Count Comparison', fontsize=12, fontweight='bold')
        ax.set_xlabel('Frame Number', fontsize=10)
        ax.set_ylabel('Vehicle Count', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    def _plot_speed_comparison(self, ax):
        """Plot average speed comparison"""
        colors = {'edge': 'blue', 'cloud': 'red', 'hybrid': 'green'}
        
        for mode, data in self.results.items():
            df = data['metrics']
            # Filter out zero speeds for cleaner plot
            speeds = df['avg_speed'].replace(0, np.nan)
            rolling = speeds.rolling(window=20, min_periods=1).mean()
            ax.plot(df['frame'], rolling, label=mode.capitalize(), 
                   color=colors[mode], linewidth=2, alpha=0.7)
        
        ax.set_title('Average Speed Comparison', fontsize=12, fontweight='bold')
        ax.set_xlabel('Frame Number', fontsize=10)
        ax.set_ylabel('Speed (mph)', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    def _plot_congestion_comparison(self, ax):
        """Plot congestion detection comparison"""
        colors = {'edge': 'blue', 'cloud': 'red', 'hybrid': 'green'}
        
        for mode, data in self.results.items():
            df = data['metrics']
            # Calculate rolling percentage
            rolling = df['congestion'].rolling(window=50, min_periods=1).mean() * 100
            ax.plot(df['frame'], rolling, label=mode.capitalize(), 
                   color=colors[mode], linewidth=2, alpha=0.7)
        
        ax.set_title('Congestion Detection Rate', fontsize=12, fontweight='bold')
        ax.set_xlabel('Frame Number', fontsize=10)
        ax.set_ylabel('Congestion (%)', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    def _plot_avg_processing_time_bars(self, ax):
        """Bar chart of average processing times"""
        modes = list(self.results.keys())
        avg_times = [data['summary']['avg_processing_time_ms'] 
                    for data in self.results.values()]
        
        colors_map = {'edge': 'blue', 'cloud': 'red', 'hybrid': 'green'}
        colors = [colors_map.get(m, 'gray') for m in modes]
        
        bars = ax.bar(modes, avg_times, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, avg_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{val:.1f}ms', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_title('Average Processing Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Time (ms)', fontsize=10)
        ax.set_xlabel('Mode', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add speedup annotations
        if 'edge' in self.results and 'cloud' in self.results:
            edge_time = self.results['edge']['summary']['avg_processing_time_ms']
            cloud_time = self.results['cloud']['summary']['avg_processing_time_ms']
            speedup = cloud_time / edge_time if edge_time > 0 else 0
            ax.text(0.5, 0.95, f'Edge is {speedup:.2f}x faster than Cloud', 
                   transform=ax.transAxes, ha='center', va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    def _plot_accuracy_comparison(self, ax):
        """Compare detection accuracy (using cloud as ground truth)"""
        if 'cloud' not in self.results:
            ax.text(0.5, 0.5, 'Cloud data not available\nfor accuracy comparison', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=11)
            ax.set_title('Detection Accuracy', fontsize=12, fontweight='bold')
            return
        
        cloud_avg = self.results['cloud']['summary']['avg_vehicle_count']
        
        modes = []
        errors = []
        
        for mode in ['edge', 'hybrid']:
            if mode in self.results:
                mode_avg = self.results[mode]['summary']['avg_vehicle_count']
                error = abs(mode_avg - cloud_avg)
                error_pct = (error / cloud_avg * 100) if cloud_avg > 0 else 0
                
                modes.append(mode.capitalize())
                errors.append(error_pct)
        
        if modes:
            colors = ['blue' if m == 'Edge' else 'green' for m in modes]
            bars = ax.bar(modes, errors, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            
            for bar, val in zip(bars, errors):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_title('Detection Error vs Cloud (Ground Truth)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Absolute Error (%)', fontsize=10)
            ax.set_xlabel('Mode', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(bottom=0)
    
    def _plot_total_duration(self, ax):
        """Bar chart of total processing duration"""
        modes = list(self.results.keys())
        durations = [data['duration'] for data in self.results.values()]
        
        colors_map = {'edge': 'blue', 'cloud': 'red', 'hybrid': 'green'}
        colors = [colors_map.get(m, 'gray') for m in modes]
        
        bars = ax.bar(modes, durations, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        for bar, val in zip(bars, durations):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{val:.1f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_title('Total Processing Duration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Time (seconds)', fontsize=10)
        ax.set_xlabel('Mode', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(bottom=0)
    
    def _plot_node_distribution(self, ax):
        """Plot pie chart showing edge/cloud frame distribution for hybrid mode"""
        if 'hybrid' not in self.results:
            ax.text(0.5, 0.5, 'Hybrid mode\ndata not available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=11)
            ax.set_title('Node Distribution (Hybrid)', fontsize=12, fontweight='bold')
            return
        
        summary = self.results['hybrid']['summary']
        edge_frames = summary.get('edge_frames', 0)
        cloud_frames = summary.get('cloud_frames', 0)
        
        if edge_frames == 0 and cloud_frames == 0:
            # Try to calculate from metrics
            df = self.results['hybrid']['metrics']
            edge_frames = (df['node'] == 'edge').sum()
            cloud_frames = (df['node'] == 'cloud').sum()
        
        if edge_frames + cloud_frames > 0:
            sizes = [edge_frames, cloud_frames]
            labels = [f'Edge\n({edge_frames} frames)', f'Cloud\n({cloud_frames} frames)']
            colors = ['skyblue', 'lightcoral']
            explode = (0.05, 0.05)
            
            ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                  autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 10})
            ax.set_title('Node Distribution (Hybrid Mode)', fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No node distribution\ndata available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=11)
            ax.set_title('Node Distribution (Hybrid)', fontsize=12, fontweight='bold')
    
    def _plot_efficiency_metrics(self, ax):
        """Plot efficiency comparison metrics"""
        if len(self.results) < 2:
            ax.text(0.5, 0.5, 'Need at least 2 modes\nfor efficiency comparison', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=11)
            ax.set_title('Efficiency Metrics', fontsize=12, fontweight='bold')
            return
        
        modes = list(self.results.keys())
        
        # Calculate frames per second
        fps_values = []
        for mode in modes:
            total_frames = self.results[mode]['summary']['total_frames']
            duration = self.results[mode]['duration']
            fps = total_frames / duration if duration > 0 else 0
            fps_values.append(fps)
        
        colors_map = {'edge': 'blue', 'cloud': 'red', 'hybrid': 'green'}
        colors = [colors_map.get(m, 'gray') for m in modes]
        
        bars = ax.bar(modes, fps_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        for bar, val in zip(bars, fps_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_title('Processing Throughput', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frames per Second', fontsize=10)
        ax.set_xlabel('Mode', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(bottom=0)
    
    def _plot_summary_table(self, ax):
        """Display summary statistics table"""
        ax.axis('off')
        
        # Prepare table data
        table_data = [['Metric', 'Edge', 'Cloud', 'Hybrid']]
        
        metrics = [
            ('Avg Time (ms)', 'avg_processing_time_ms', '{:.1f}'),
            ('Avg Vehicles', 'avg_vehicle_count', '{:.2f}'),
            ('Avg Speed (mph)', 'avg_speed_mph', '{:.1f}'),
            ('Congestion (%)', 'congestion_percentage', '{:.1f}'),
            ('Total Duration (s)', 'duration', '{:.1f}'),
        ]
        
        for label, key, fmt in metrics:
            row = [label]
            for mode in ['edge', 'cloud', 'hybrid']:
                if mode in self.results:
                    if key == 'duration':
                        val = self.results[mode]['duration']
                    else:
                        val = self.results[mode]['summary'].get(key, 0)
                    row.append(fmt.format(val))
                else:
                    row.append('N/A')
            table_data.append(row)
        
        # Create table
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.4, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.2)
        
        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data)):
            for j in range(4):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        ax.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    def _generate_individual_comparisons(self):
        """Generate focused comparison plots"""
        
        # Processing time comparison
        fig, ax = plt.subplots(figsize=(14, 7))
        colors = {'edge': 'blue', 'cloud': 'red', 'hybrid': 'green'}
        
        for mode, data in self.results.items():
            df = data['metrics']
            ax.plot(df['frame'], df['processing_time_ms'], 
                   label=mode.capitalize(), color=colors[mode], 
                   linewidth=1.5, alpha=0.7)
        
        ax.set_title('Processing Time Comparison (All Modes)', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Frame Number', fontsize=13)
        ax.set_ylabel('Processing Time (ms)', fontsize=13)
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3)
        
        filepath = os.path.join(self.output_dir, 'processing_time_comparison.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Processing time comparison saved to: {filepath}")
        
        # Vehicle detection comparison
        fig, ax = plt.subplots(figsize=(14, 7))
        
        for mode, data in self.results.items():
            df = data['metrics']
            ax.plot(df['frame'], df['vehicle_count'], 
                   label=mode.capitalize(), color=colors[mode], 
                   linewidth=1.5, alpha=0.7)
        
        ax.set_title('Vehicle Detection Comparison (All Modes)', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Frame Number', fontsize=13)
        ax.set_ylabel('Vehicle Count', fontsize=13)
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3)
        
        filepath = os.path.join(self.output_dir, 'vehicle_count_comparison.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Vehicle count comparison saved to: {filepath}")
    
    def _generate_comparison_report(self):
        """Generate text report comparing all modes"""
        
        report_path = os.path.join(self.output_dir, 'comparison_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("MULTI-MODE COMPARISON REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Processing duration
            f.write("-"*70 + "\n")
            f.write("PROCESSING DURATION\n")
            f.write("-"*70 + "\n")
            for mode, data in self.results.items():
                f.write(f"{mode.capitalize():12s}: {data['duration']:.2f} seconds\n")
            f.write("\n")
            
            # Performance metrics
            f.write("-"*70 + "\n")
            f.write("PERFORMANCE METRICS\n")
            f.write("-"*70 + "\n\n")
            
            for mode, data in self.results.items():
                summary = data['summary']
                f.write(f"{mode.upper()} MODE:\n")
                f.write(f"  Average processing time: {summary['avg_processing_time_ms']:.2f} ms\n")
                f.write(f"  Average vehicle count: {summary['avg_vehicle_count']:.2f}\n")
                f.write(f"  Average speed: {summary['avg_speed_mph']:.2f} mph\n")
                f.write(f"  Congestion percentage: {summary['congestion_percentage']:.2f}%\n")
                f.write(f"  Total frames: {summary['total_frames']}\n\n")
            
            # Comparison analysis
            if 'edge' in self.results and 'cloud' in self.results:
                f.write("-"*70 + "\n")
                f.write("COMPARISON ANALYSIS (Edge vs Cloud)\n")
                f.write("-"*70 + "\n")
                
                edge_time = self.results['edge']['summary']['avg_processing_time_ms']
                cloud_time = self.results['cloud']['summary']['avg_processing_time_ms']
                speedup = cloud_time / edge_time if edge_time > 0 else 0
                
                f.write(f"Speed improvement (Edge): {speedup:.2f}x faster\n")
                
                edge_vehicles = self.results['edge']['summary']['avg_vehicle_count']
                cloud_vehicles = self.results['cloud']['summary']['avg_vehicle_count']
                accuracy = (1 - abs(edge_vehicles - cloud_vehicles) / cloud_vehicles) * 100 if cloud_vehicles > 0 else 0
                
                f.write(f"Edge detection accuracy: {accuracy:.1f}%\n")
                f.write(f"Vehicle count difference: {abs(edge_vehicles - cloud_vehicles):.2f}\n\n")
            
            if 'hybrid' in self.results:
                f.write("-"*70 + "\n")
                f.write("HYBRID MODE ANALYSIS\n")
                f.write("-"*70 + "\n")
                hybrid_summary = self.results['hybrid']['summary']
                edge_frames = hybrid_summary.get('edge_frames', 0)
                cloud_frames = hybrid_summary.get('cloud_frames', 0)
                
                # Try to get from metrics if not in summary
                if edge_frames == 0 and cloud_frames == 0:
                    df = self.results['hybrid']['metrics']
                    edge_frames = (df['node'] == 'edge').sum()
                    cloud_frames = (df['node'] == 'cloud').sum()
                
                f.write(f"Edge frames: {edge_frames}\n")
                f.write(f"Cloud frames: {cloud_frames}\n")
                total = edge_frames + cloud_frames
                edge_pct = (edge_frames / total * 100) if total > 0 else 0
                f.write(f"Edge usage: {edge_pct:.1f}%\n\n")
            
            f.write("="*70 + "\n")
        
        print(f"✓ Comparison report saved to: {report_path}")
        
        # Save as JSON
        json_path = os.path.join(self.output_dir, 'comparison_data.json')
        comparison_data = {
            mode: {
                'duration': data['duration'],
                'summary': {k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else str(v) 
                           for k, v in data['summary'].items()}
            }
            for mode, data in self.results.items()
        }
        
        with open(json_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        print(f"✓ Comparison data saved to: {json_path}")


# Import for resource monitoring (optional)
try:
    from resource_monitor import ResourceMonitor
except ImportError:
    ResourceMonitor = None