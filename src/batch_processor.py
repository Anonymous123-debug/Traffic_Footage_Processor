"""
Complete Batch Processor Module
Process multiple videos and generate aggregate analysis
"""

import os
import glob
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
from typing import List, Dict
from tqdm import tqdm

# Ensure src directory is in path
sys.path.append('src')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from metrics import MetricsCollector


class BatchProcessor:
    """Process multiple videos and aggregate metrics"""
    
    def __init__(self, system, output_dir='data/results/batch'):
        self.system = system
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.batch_results = []
        self.all_metrics = []
        
    def process_folder(self, folder_path, file_pattern='*.mp4', mode='hybrid', 
                      bandwidth=None, max_videos=None):
        """
        Process all videos in a folder
        
        Args:
            folder_path: Path to folder containing videos
            file_pattern: Pattern to match video files (default: '*.mp4')
            mode: Processing mode ('edge', 'cloud', 'hybrid')
            bandwidth: Network bandwidth in kbps
            max_videos: Maximum number of videos to process (None = all)
        """
        
        # Find all video files
        video_files = self._find_video_files(folder_path, file_pattern)
        
        if not video_files:
            print(f"❌ No video files found in {folder_path}")
            return
        
        if max_videos:
            video_files = video_files[:max_videos]
        
        print(f"\n{'='*70}")
        print(f"BATCH PROCESSING: {len(video_files)} videos found")
        print(f"Mode: {mode.upper()} | Bandwidth: {bandwidth or 'default'} kbps")
        print(f"{'='*70}\n")
        
        # Process each video
        for idx, video_path in enumerate(video_files, 1):
            video_name = os.path.basename(video_path)
            print(f"\n{'='*70}")
            print(f"[{idx}/{len(video_files)}] Processing: {video_name}")
            print(f"{'='*70}")
            
            try:
                result = self._process_single_video(
                    video_path=video_path,
                    mode=mode,
                    bandwidth=bandwidth,
                    video_index=idx
                )
                self.batch_results.append(result)
                print(f"✓ Completed in {result['processing_duration_sec']:.2f} seconds")
                
            except Exception as e:
                print(f"❌ Error processing {video_name}: {str(e)}")
                self.batch_results.append({
                    'video_name': video_name,
                    'video_path': video_path,
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Generate batch analysis
        print(f"\n{'='*70}")
        print("GENERATING BATCH ANALYSIS...")
        print(f"{'='*70}\n")
        
        self._generate_batch_summary()
        self._generate_batch_plots()
        self._generate_comparison_matrix()
        
        print(f"\n{'='*70}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'='*70}\n")
        
        self._print_batch_summary()
    
    def _find_video_files(self, folder_path, pattern):
        """Find all video files matching pattern"""
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.MP4', '*.AVI']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(folder_path, ext)))
        
        # Remove duplicates and sort
        video_files = sorted(list(set(video_files)))
        return video_files
    
    def _process_single_video(self, video_path, mode, bandwidth, video_index):
        """Process a single video and return results"""
        
        video_name = os.path.basename(video_path)
        video_stem = Path(video_name).stem
        
        # Create output subfolder
        video_output = os.path.join(self.output_dir, f"{video_index:03d}_{video_stem}")
        os.makedirs(video_output, exist_ok=True)
        
        # Reset metrics collector
        self.system.metrics = MetricsCollector(video_output)
        
        # Process video
        start_time = datetime.now()
        self.system.process_video(
            video_path=video_path,
            output_dir=video_output,
            network_bandwidth=bandwidth,
            mode=mode
        )
        end_time = datetime.now()
        processing_duration = (end_time - start_time).total_seconds()
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        # Collect summary metrics
        metrics_df = pd.DataFrame(self.system.metrics.metrics)
        summary = self.system.metrics.generate_summary(metrics_df)
        
        # Calculate throughput
        fps_processed = total_frames / processing_duration if processing_duration > 0 else 0
        
        # Store metrics for aggregate analysis
        self.all_metrics.append({
            'video_name': video_name,
            'metrics': metrics_df
        })
        
        # Build result dictionary
        result = {
            'video_index': video_index,
            'video_name': video_name,
            'video_path': video_path,
            'output_path': video_output,
            'mode': mode,
            'bandwidth': bandwidth,
            'status': 'success',
            'processing_duration_sec': processing_duration,
            'video_duration_sec': video_duration,
            'video_width': width,
            'video_height': height,
            'video_fps': fps,
            'processing_fps': fps_processed,
            'realtime_factor': fps_processed / fps if fps > 0 else 0,
            **summary
        }
        
        return result
    
    def _generate_batch_summary(self):
        """Generate summary CSV and JSON files"""
        
        # Save batch results to CSV
        df = pd.DataFrame(self.batch_results)
        csv_path = os.path.join(self.output_dir, 'batch_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"✓ Batch summary CSV saved to: {csv_path}")
        
        # Save as JSON
        json_path = os.path.join(self.output_dir, 'batch_summary.json')
        with open(json_path, 'w') as f:
            json.dump(self.batch_results, f, indent=2, default=str)
        print(f"✓ Batch summary JSON saved to: {json_path}")
        
        # Generate detailed text report
        self._generate_text_report()
    
    def _generate_text_report(self):
        """Generate comprehensive text report"""
        
        report_path = os.path.join(self.output_dir, 'batch_report.txt')
        
        successful = [r for r in self.batch_results if r.get('status') == 'success']
        failed = [r for r in self.batch_results if r.get('status') == 'failed']
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("BATCH PROCESSING REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total videos: {len(self.batch_results)}\n")
            f.write(f"Successful: {len(successful)}\n")
            f.write(f"Failed: {len(failed)}\n\n")
            
            if successful:
                f.write("-"*70 + "\n")
                f.write("AGGREGATE STATISTICS (Successful Videos)\n")
                f.write("-"*70 + "\n\n")
                
                # Calculate aggregate stats
                total_frames = sum(r.get('total_frames', 0) for r in successful)
                total_duration = sum(r.get('processing_duration_sec', 0) for r in successful)
                avg_processing_time = np.mean([r.get('avg_processing_time_ms', 0) for r in successful])
                avg_vehicle_count = np.mean([r.get('avg_vehicle_count', 0) for r in successful])
                avg_speed = np.mean([r.get('avg_speed_mph', 0) for r in successful])
                avg_congestion = np.mean([r.get('congestion_percentage', 0) for r in successful])
                avg_fps = np.mean([r.get('processing_fps', 0) for r in successful])
                
                f.write(f"Total frames processed: {total_frames:,}\n")
                f.write(f"Total processing time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)\n")
                f.write(f"Average processing speed: {avg_fps:.2f} fps\n")
                f.write(f"Average processing time per frame: {avg_processing_time:.2f} ms\n")
                f.write(f"Average vehicle count: {avg_vehicle_count:.2f}\n")
                f.write(f"Average speed: {avg_speed:.2f} mph\n")
                f.write(f"Average congestion: {avg_congestion:.2f}%\n\n")
                
                # Performance ranking
                f.write("-"*70 + "\n")
                f.write("PERFORMANCE RANKING (by processing time)\n")
                f.write("-"*70 + "\n\n")
                
                sorted_by_time = sorted(successful, key=lambda x: x['avg_processing_time_ms'])
                for i, result in enumerate(sorted_by_time[:10], 1):  # Top 10
                    f.write(f"{i:2d}. {result['video_name']:40s} {result['avg_processing_time_ms']:6.2f} ms\n")
                f.write("\n")
                
                # Traffic statistics
                f.write("-"*70 + "\n")
                f.write("TRAFFIC STATISTICS (by vehicle count)\n")
                f.write("-"*70 + "\n\n")
                
                sorted_by_vehicles = sorted(successful, key=lambda x: x['avg_vehicle_count'], reverse=True)
                for i, result in enumerate(sorted_by_vehicles[:10], 1):  # Top 10
                    f.write(f"{i:2d}. {result['video_name']:40s} {result['avg_vehicle_count']:6.2f} vehicles\n")
                f.write("\n")
            
            # Individual results
            f.write("-"*70 + "\n")
            f.write("INDIVIDUAL VIDEO RESULTS\n")
            f.write("-"*70 + "\n\n")
            
            for result in self.batch_results:
                vid_idx = result.get('video_index', '?')
                vid_idx_str = f"{vid_idx:03d}" if isinstance(vid_idx, int) else str(vid_idx)
                f.write(f"[{vid_idx_str}] {result['video_name']}\n")
                f.write(f"     Status: {result['status']}\n")
                
                if result['status'] == 'success':
                    f.write(f"     Frames: {result.get('total_frames', 'N/A')}\n")
                    f.write(f"     Processing time: {result.get('processing_duration_sec', 0):.2f} s\n")
                    f.write(f"     Avg vehicles: {result.get('avg_vehicle_count', 0):.2f}\n")
                    f.write(f"     Avg speed: {result.get('avg_speed_mph', 0):.2f} mph\n")
                    f.write(f"     Congestion: {result.get('congestion_percentage', 0):.2f}%\n")
                else:
                    f.write(f"     Error: {result.get('error', 'Unknown')}\n")
                
                f.write("\n")
            
            if failed:
                f.write("-"*70 + "\n")
                f.write("FAILED VIDEOS\n")
                f.write("-"*70 + "\n\n")
                for result in failed:
                    f.write(f"• {result['video_name']}\n")
                    f.write(f"  Error: {result.get('error', 'Unknown error')}\n\n")
        
        print(f"✓ Batch report saved to: {report_path}")
    
    def _generate_batch_plots(self):
        """Generate comparison plots across all videos"""
        
        successful = [r for r in self.batch_results if r.get('status') == 'success']
        
        if len(successful) < 1:
            print("⚠ No successful videos for plotting")
            return
        
        # Create comprehensive plot
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Prepare data
        video_names = [r['video_name'][:25] for r in successful]
        indices = range(len(video_names))
        
        # Plot 1: Processing time comparison
        ax1 = fig.add_subplot(gs[0, :])
        processing_times = [r['avg_processing_time_ms'] for r in successful]
        bars = ax1.bar(indices, processing_times, color='skyblue', edgecolor='black', linewidth=1)
        ax1.set_title('Average Processing Time per Video', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Time (ms)', fontsize=11)
        ax1.set_xticks(indices)
        ax1.set_xticklabels(video_names, rotation=45, ha='right', fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Vehicle count comparison
        ax2 = fig.add_subplot(gs[1, 0])
        vehicle_counts = [r['avg_vehicle_count'] for r in successful]
        bars = ax2.bar(indices, vehicle_counts, color='lightgreen', edgecolor='black')
        ax2.set_title('Average Vehicle Count', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Vehicles', fontsize=10)
        ax2.set_xticks(indices)
        ax2.set_xticklabels(video_names, rotation=45, ha='right', fontsize=8)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Average speed comparison
        ax3 = fig.add_subplot(gs[1, 1])
        avg_speeds = [r['avg_speed_mph'] for r in successful]
        bars = ax3.bar(indices, avg_speeds, color='orange', edgecolor='black')
        ax3.set_title('Average Speed', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Speed (mph)', fontsize=10)
        ax3.set_xticks(indices)
        ax3.set_xticklabels(video_names, rotation=45, ha='right', fontsize=8)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Congestion percentage
        ax4 = fig.add_subplot(gs[1, 2])
        congestion_pcts = [r['congestion_percentage'] for r in successful]
        bars = ax4.bar(indices, congestion_pcts, color='coral', edgecolor='black')
        ax4.set_title('Congestion Percentage', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Congestion (%)', fontsize=10)
        ax4.set_xticks(indices)
        ax4.set_xticklabels(video_names, rotation=45, ha='right', fontsize=8)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Plot 5: Processing FPS
        ax5 = fig.add_subplot(gs[2, 0])
        fps_values = [r['processing_fps'] for r in successful]
        bars = ax5.bar(indices, fps_values, color='purple', alpha=0.7, edgecolor='black')
        ax5.set_title('Processing Throughput', fontsize=12, fontweight='bold')
        ax5.set_ylabel('FPS', fontsize=10)
        ax5.set_xticks(indices)
        ax5.set_xticklabels(video_names, rotation=45, ha='right', fontsize=8)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Plot 6: Total duration
        ax6 = fig.add_subplot(gs[2, 1])
        durations = [r['processing_duration_sec'] for r in successful]
        bars = ax6.bar(indices, durations, color='teal', alpha=0.7, edgecolor='black')
        ax6.set_title('Total Processing Duration', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Time (seconds)', fontsize=10)
        ax6.set_xticks(indices)
        ax6.set_xticklabels(video_names, rotation=45, ha='right', fontsize=8)
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Plot 7: Statistics summary
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        
        stats_text = "BATCH STATISTICS\n" + "="*30 + "\n\n"
        stats_text += f"Total Videos: {len(successful)}\n"
        stats_text += f"Total Frames: {sum(r['total_frames'] for r in successful):,}\n"
        stats_text += f"Total Time: {sum(durations):.2f}s\n\n"
        stats_text += f"Avg Processing: {np.mean(processing_times):.2f}ms\n"
        stats_text += f"Avg Vehicles: {np.mean(vehicle_counts):.2f}\n"
        stats_text += f"Avg Speed: {np.mean(avg_speeds):.2f}mph\n"
        stats_text += f"Avg Congestion: {np.mean(congestion_pcts):.2f}%\n"
        
        ax7.text(0.1, 0.9, stats_text, transform=ax7.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Save figure
        fig.suptitle('Batch Processing Comparison', fontsize=16, fontweight='bold', y=0.995)
        filepath = os.path.join(self.output_dir, 'batch_comparison.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Batch comparison plot saved to: {filepath}")
    
    def _generate_comparison_matrix(self):
        """Generate heatmap comparison matrix"""
        
        successful = [r for r in self.batch_results if r.get('status') == 'success']
        
        if len(successful) < 2:
            print("⚠ Need at least 2 videos for comparison matrix")
            return
        
        # Create heatmap data
        metrics = ['avg_processing_time_ms', 'avg_vehicle_count', 'avg_speed_mph', 
                  'congestion_percentage', 'processing_fps']
        metric_labels = ['Processing Time\n(ms)', 'Vehicles', 'Speed\n(mph)', 
                        'Congestion\n(%)', 'FPS']
        
        data = []
        for metric in metrics:
            row = [r.get(metric, 0) for r in successful]
            # Normalize to 0-100 scale
            if row:
                min_val, max_val = min(row), max(row)
                if max_val > min_val:
                    row = [(x - min_val) / (max_val - min_val) * 100 for x in row]
            data.append(row)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(max(12, len(successful)), 8))
        
        video_names = [r['video_name'][:20] for r in successful]
        
        im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
        
        # Set ticks
        ax.set_xticks(range(len(video_names)))
        ax.set_yticks(range(len(metric_labels)))
        ax.set_xticklabels(video_names, rotation=45, ha='right')
        ax.set_yticklabels(metric_labels)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Value (0-100)', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(metrics)):
            for j in range(len(successful)):
                value = successful[j].get(metrics[i], 0)
                text = ax.text(j, i, f'{value:.1f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('Performance Comparison Matrix', fontsize=14, fontweight='bold', pad=20)
        
        filepath = os.path.join(self.output_dir, 'comparison_matrix.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Comparison matrix saved to: {filepath}")
    
    def _print_batch_summary(self):
        """Print summary to console"""
        
        successful = [r for r in self.batch_results if r.get('status') == 'success']
        failed = [r for r in self.batch_results if r.get('status') == 'failed']
        
        print(f"\n{'='*70}")
        print("BATCH PROCESSING SUMMARY")
        print(f"{'='*70}")
        print(f"Total videos: {len(self.batch_results)}")
        print(f"Successful: {len(successful)} ✓")
        print(f"Failed: {len(failed)} ✗")
        
        if successful:
            print(f"\nAggregate Statistics:")
            print(f"  Total frames: {sum(r['total_frames'] for r in successful):,}")
            print(f"  Total time: {sum(r['processing_duration_sec'] for r in successful):.2f}s")
            print(f"  Avg processing: {np.mean([r['avg_processing_time_ms'] for r in successful]):.2f}ms")
            print(f"  Avg vehicles: {np.mean([r['avg_vehicle_count'] for r in successful]):.2f}")
            print(f"  Avg speed: {np.mean([r['avg_speed_mph'] for r in successful]):.2f}mph")
        
        print(f"\nResults saved to: {self.output_dir}")
        print(f"  📊 batch_comparison.png")
        print(f"  📈 comparison_matrix.png")
        print(f"  📄 batch_report.txt")
        print(f"  💾 batch_summary.csv")
        print(f"  💾 batch_summary.json")
        print(f"{'='*70}\n")