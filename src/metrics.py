"""
Metrics Collection and Analysis Module
Collects and analyzes performance metrics for traffic monitoring
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List
import os


class MetricsCollector:
    """Collect and analyze performance metrics"""
    
    def __init__(self, output_dir=None):
        """
        Initialize MetricsCollector
        
        Args:
            output_dir: Optional output directory for saving results
        """
        self.output_dir = output_dir or 'data/results'
        self.data = []
        self.processing_times = []
        self.vehicle_counts = []
        self.speeds = []
        self.congestion_states = []
        self.nodes = []
    
    def add_frame_metrics(self, processing_time: float, vehicle_count: int,
                         avg_speed: float, congestion: bool, node_type: str):
        """
        Add metrics for a single frame
        
        Args:
            processing_time: Processing time in milliseconds
            vehicle_count: Number of vehicles detected
            avg_speed: Average speed in mph
            congestion: Whether congestion was detected
            node_type: 'edge' or 'cloud'
        """
        self.data.append({
            'processing_time_ms': processing_time,
            'vehicle_count': vehicle_count,
            'avg_speed': avg_speed,
            'congestion': congestion,
            'node': node_type
        })
        
        self.processing_times.append(processing_time)
        self.vehicle_counts.append(vehicle_count)
        self.speeds.append(avg_speed)
        self.congestion_states.append(1 if congestion else 0)
        self.nodes.append(node_type)
    
    def save_to_csv(self, filepath: str):
        """Save metrics to CSV file"""
        if not self.data:
            print("No metrics to save")
            return None
        
        df = pd.DataFrame(self.data)
        df['frame_number'] = range(len(df))
        
        # Reorder columns
        cols = ['frame_number', 'processing_time_ms', 'vehicle_count', 
                'avg_speed', 'congestion', 'node']
        df = df[cols]
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        df.to_csv(filepath, index=False)
        print(f"✅ Metrics saved to: {filepath}")
        return df
    
    def calculate_errors(self) -> Dict:
        """Calculate aggregate error metrics"""
        if not self.data:
            return {
                'avg_processing_time_ms': 0,
                'avg_vehicle_count': 0,
                'avg_speed': 0,
                'avg_speed_mph': 0,
                'total_frames': 0,
                'congestion_error_rate': 0,
                'congestion_frames': 0,
                'congestion_percentage': 0,
                'speed_error_rate': 0,
                'rms_error': 0
            }
        
        # Calculate statistics
        processing_times = np.array(self.processing_times)
        vehicle_counts = np.array(self.vehicle_counts)
        speeds = np.array(self.speeds)
        congestion_states = np.array(self.congestion_states)
        
        # RMS error for speeds
        speed_mean = np.mean(speeds)
        rms_error = np.sqrt(np.mean(np.square(speeds - speed_mean)))
        
        # Speed error rate (coefficient of variation)
        speed_error_rate = (np.std(speeds) / speed_mean) if speed_mean > 0 else 0
        
        return {
            'avg_processing_time_ms': float(np.mean(processing_times)),
            'avg_vehicle_count': float(np.mean(vehicle_counts)),
            'avg_speed': float(speed_mean),
            'avg_speed_mph': float(speed_mean),  # Alias for compatibility
            'total_frames': len(self.data),
            'congestion_error_rate': float(np.var(congestion_states)),
            'congestion_frames': int(np.sum(congestion_states)),
            'congestion_percentage': float(np.mean(congestion_states) * 100),
            'speed_error_rate': float(speed_error_rate),
            'rms_error': float(rms_error)
        }
    
    def generate_summary(self, df=None):
        """
        Generate summary statistics from DataFrame
        
        Args:
            df: Optional DataFrame, if None uses self.data
            
        Returns:
            Dictionary with summary statistics
        """
        if df is None:
            if not self.data:
                return {
                    'total_frames': 0,
                    'avg_processing_time_ms': 0,
                    'avg_vehicle_count': 0,
                    'avg_speed_mph': 0,
                    'congestion_percentage': 0,
                }
            df = pd.DataFrame(self.data)
        
        return {
            'total_frames': len(df),
            'avg_processing_time_ms': float(df['processing_time_ms'].mean()) if len(df) > 0 else 0,
            'avg_vehicle_count': float(df['vehicle_count'].mean()) if len(df) > 0 else 0,
            'avg_speed_mph': float(df['avg_speed'].mean()) if len(df) > 0 else 0,
            'congestion_percentage': float((df['congestion'].sum() / len(df)) * 100) if len(df) > 0 else 0,
        }
    
    def plot_metrics(self, output_path: str):
        """Generate comprehensive metrics visualization"""
        if not self.data:
            print("No metrics to plot")
            return
        
        df = pd.DataFrame(self.data)
        df['frame_number'] = range(len(df))
        
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Plot 1: Processing Time
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(df['frame_number'], df['processing_time_ms'], 
                label='Processing Time', color='blue', alpha=0.7, linewidth=0.5)
        ax1.axhline(df['processing_time_ms'].mean(), color='red', 
                   linestyle='--', label=f"Mean: {df['processing_time_ms'].mean():.2f}ms")
        ax1.set_xlabel('Frame Number', fontweight='bold')
        ax1.set_ylabel('Processing Time (ms)', fontweight='bold')
        ax1.set_title('Processing Time per Frame', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Vehicle Count
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(df['frame_number'], df['vehicle_count'], 
                label='Vehicle Count', color='green', alpha=0.7, linewidth=0.5)
        ax2.axhline(df['vehicle_count'].mean(), color='red', 
                   linestyle='--', label=f"Mean: {df['vehicle_count'].mean():.2f}")
        ax2.set_xlabel('Frame Number', fontweight='bold')
        ax2.set_ylabel('Vehicle Count', fontweight='bold')
        ax2.set_title('Vehicle Detection Count', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Average Speed
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(df['frame_number'], df['avg_speed'], 
                label='Avg Speed', color='orange', alpha=0.7, linewidth=0.5)
        ax3.axhline(df['avg_speed'].mean(), color='red', 
                   linestyle='--', label=f"Mean: {df['avg_speed'].mean():.2f} mph")
        ax3.set_xlabel('Frame Number', fontweight='bold')
        ax3.set_ylabel('Speed (mph)', fontweight='bold')
        ax3.set_title('Average Vehicle Speed', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Congestion Detection
        ax4 = fig.add_subplot(gs[1, 1])
        congestion_numeric = df['congestion'].astype(int)
        ax4.fill_between(df['frame_number'], congestion_numeric, 
                        alpha=0.5, color='red', label='Congestion Detected')
        congestion_pct = (congestion_numeric.sum() / len(df)) * 100
        ax4.text(0.5, 0.5, f'Congestion: {congestion_pct:.1f}% of frames',
                transform=ax4.transAxes, ha='center', va='center',
                fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        ax4.set_xlabel('Frame Number', fontweight='bold')
        ax4.set_ylabel('Congestion Status', fontweight='bold')
        ax4.set_title('Congestion Detection', fontsize=12, fontweight='bold')
        ax4.set_ylim(-0.1, 1.1)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Node Distribution
        ax5 = fig.add_subplot(gs[2, 0])
        if 'node' in df.columns:
            node_counts = df['node'].value_counts()
            colors = {'edge': '#2ecc71', 'cloud': '#e74c3c', 
                     'edge_fallback': '#f39c12'}
            bars = ax5.bar(node_counts.index, node_counts.values,
                          color=[colors.get(n, '#95a5a6') for n in node_counts.index],
                          edgecolor='black', linewidth=1.5)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}\n({height/len(df)*100:.1f}%)',
                        ha='center', va='bottom', fontweight='bold')
            
            ax5.set_xlabel('Processing Node', fontweight='bold')
            ax5.set_ylabel('Frame Count', fontweight='bold')
            ax5.set_title('Processing Node Distribution', fontsize=12, fontweight='bold')
            ax5.grid(axis='y', alpha=0.3)
        
        # Plot 6: Summary Statistics
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')
        
        stats = self.calculate_errors()
        summary_text = f"""
        SUMMARY STATISTICS
        {'='*40}
        
        Total Frames: {stats['total_frames']:,}
        
        Processing Time:
          • Average: {stats['avg_processing_time_ms']:.2f} ms
          • Min: {min(self.processing_times):.2f} ms
          • Max: {max(self.processing_times):.2f} ms
        
        Vehicle Detection:
          • Avg Count: {stats['avg_vehicle_count']:.2f}
          • Max Count: {max(self.vehicle_counts)}
        
        Speed Analysis:
          • Avg Speed: {stats['avg_speed']:.2f} mph
          • Speed Error Rate: {stats['speed_error_rate']:.4f}
          • RMS Error: {stats['rms_error']:.2f} mph
        
        Congestion:
          • Congested Frames: {stats['congestion_frames']}
          • Congestion %: {stats['congestion_percentage']:.2f}%
        """
        
        ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Main title
        fig.suptitle('Traffic Monitoring Performance Metrics', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Save plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Metrics plot saved to: {output_path}")
    
    def reset(self):
        """Reset all metrics"""
        self.data = []
        self.processing_times = []
        self.vehicle_counts = []
        self.speeds = []
        self.congestion_states = []
        self.nodes = []
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get metrics as pandas DataFrame"""
        if not self.data:
            return pd.DataFrame()
        return pd.DataFrame(self.data)