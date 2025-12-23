"""
Traffic Monitoring System - Main Entry Point
Complete implementation with Edge-Cloud processing
"""

import cv2
import time
import argparse
from pathlib import Path
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

# Add src to path
sys.path.append('src')

# Core imports
from edge_node import EdgeNode
from cloud_node import CloudNode
from network_simulator import NetworkSimulator
from metrics import MetricsCollector
from utils import ConfigLoader

# Asymmetric processor (with hybrid comparison)
from asymmetric_processor import run_asymmetric_processing


class TrafficMonitoringSystem:
    """Main traffic monitoring system with hybrid edge-cloud processing"""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        self.config_path = config_path
        self.config = ConfigLoader.load_config(config_path)
        
        print("Initializing Traffic Monitoring System...")
        self.edge_node = EdgeNode(self.config)
        self.cloud_node = CloudNode(self.config)
        
        self.network = NetworkSimulator(
            self.config['system']['network_threshold'])
        
        self.metrics = MetricsCollector()
        
    def process_video(self, video_path: str, output_dir: str,
                     network_bandwidth: int = 1000, mode: str = 'hybrid'):
        """Process video with specified mode"""
        self.network.set_condition(network_bandwidth)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\n{'='*60}")
        print(f"Processing: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}")
        print(f"Total frames: {total_frames}")
        print(f"Mode: {mode}, Bandwidth: {network_bandwidth} KB/s")
        print(f"{'='*60}\n")
        
        # Output video
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'output_{mode}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start = time.time()
            
            # Determine processing node
            if mode == 'hybrid':
                use_cloud = self.network.should_use_cloud()
                node_name = 'cloud' if use_cloud else 'edge'
            elif mode == 'cloud':
                use_cloud = True
                node_name = 'cloud'
            else:  # edge
                use_cloud = False
                node_name = 'edge'
            
            # Process frame
            if use_cloud:
                transferred_frame = self.network.simulate_frame_transfer(frame)
                if transferred_frame is not None:
                    result = self.cloud_node.process_frame(transferred_frame)
                else:
                    result = self.edge_node.process_frame(frame)
                    node_name = 'edge_fallback'
            else:
                result = self.edge_node.process_frame(frame)
            
            processing_time = (time.time() - frame_start) * 1000  # Convert to ms
            
            # Visualize
            vis_frame = self._visualize_results(frame.copy(), result, node_name)
            out.write(vis_frame)
            
            # Collect metrics
            self.metrics.add_frame_metrics(
                processing_time=processing_time,
                vehicle_count=result.get('vehicle_count', 0),
                avg_speed=result.get('avg_speed', 0),
                congestion=result.get('is_congested', False),
                node_type=node_name
            )
            
            frame_count += 1
            
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps_actual = frame_count / elapsed
                print(f"Processed {frame_count}/{total_frames} frames "
                      f"({fps_actual:.1f} fps, {processing_time:.1f}ms/frame)")
        
        cap.release()
        out.release()
        
        print(f"\nVideo saved to: {output_path}")
        
        # Save metrics
        metrics_path = os.path.join(output_dir, f'metrics_{mode}.csv')
        self.metrics.save_to_csv(metrics_path)
        
        # Generate plots
        plots_path = os.path.join(output_dir, f'metrics_plot_{mode}.png')
        self.metrics.plot_metrics(plots_path)
        
        # Print summary
        errors = self.metrics.calculate_errors()
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        print(f"Mode: {mode}")
        print(f"Total frames: {frame_count}")
        print(f"Avg processing time: {errors['avg_processing_time_ms']:.2f} ms")
        print(f"Avg vehicle count: {errors['avg_vehicle_count']:.2f}")
        print(f"Congestion error rate: {errors['congestion_error_rate']:.4f}")
        print(f"Speed error rate: {errors['speed_error_rate']:.4f}")
        print(f"RMS error: {errors['rms_error']:.2f} mph")
        print("="*60 + "\n")
        
        return errors
    
    def _visualize_results(self, frame, result: dict, node_name: str):
        """Add visualization overlays"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Text info
        y_offset = 30
        texts = [
            f"Node: {node_name.upper()}",
            f"Vehicles: {result['vehicle_count']}",
            f"Avg Speed: {result['avg_speed']:.1f} mph",
        ]
        
        for text in texts:
            cv2.putText(frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
        
        # Congestion warning
        if result.get('is_congested', False):
            cv2.putText(frame, "CONGESTION!", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        
        # Draw bounding boxes
        if 'detections' in result:
            for det in result['detections']:
                x, y, w_box, h_box = det['bbox']
                color = (0, 255, 0) if det.get('confidence', 0) > 0.5 else (0, 255, 255)
                cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), color, 2)
                
                # Label
                if 'class_name' in det:
                    label = f"{det['class_name']} {det['confidence']:.2f}"
                    cv2.putText(frame, label, (x, y-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def compare_modes(self, video_path: str, output_dir: str,
                     bandwidth_levels: List[int] = None):
        """Compare all processing modes"""
        if bandwidth_levels is None:
            bandwidth_levels = [1000, 500, 300, 200]
        
        results = []
        
        for bandwidth in bandwidth_levels:
            print(f"\n{'='*60}")
            print(f"Testing with {bandwidth} KB/s bandwidth")
            print(f"{'='*60}")
            
            for mode in ['edge', 'cloud', 'hybrid']:
                mode_dir = os.path.join(output_dir, f'{mode}_{bandwidth}kbps')
                os.makedirs(mode_dir, exist_ok=True)
                
                self.metrics = MetricsCollector()
                
                errors = self.process_video(video_path, mode_dir,
                                           bandwidth, mode)
                
                results.append({
                    'mode': mode,
                    'bandwidth_kbps': bandwidth,
                    **errors
                })
        
        # Save comparison
        comparison_df = pd.DataFrame(results)
        comparison_path = os.path.join(output_dir, 'comparison_results.csv')
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\nComparison saved to: {comparison_path}")
        
        # Generate comparison plots
        self._plot_comparison(comparison_df, output_dir)
        
        return comparison_df
    
    def _plot_comparison(self, df: pd.DataFrame, output_dir: str):
        """Generate comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Processing time
        pivot = df.pivot(index='bandwidth_kbps', columns='mode',
                        values='avg_processing_time_ms')
        pivot.plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Processing Time Comparison')
        axes[0, 0].set_ylabel('Time (ms)')
        axes[0, 0].set_xlabel('Bandwidth (KB/s)')
        axes[0, 0].legend(title='Mode')
        
        # Vehicle count
        pivot = df.pivot(index='bandwidth_kbps', columns='mode',
                        values='avg_vehicle_count')
        pivot.plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Average Vehicle Count')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_xlabel('Bandwidth (KB/s)')
        
        # Congestion error
        pivot = df.pivot(index='bandwidth_kbps', columns='mode',
                        values='congestion_error_rate')
        pivot.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Congestion Detection Error Rate')
        axes[1, 0].set_ylabel('Error Rate')
        axes[1, 0].set_xlabel('Bandwidth (KB/s)')
        
        # Speed error
        pivot = df.pivot(index='bandwidth_kbps', columns='mode',
                        values='speed_error_rate')
        pivot.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Speed Detection Error Rate')
        axes[1, 1].set_ylabel('Error Rate')
        axes[1, 1].set_xlabel('Bandwidth (KB/s)')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'comparison_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Comparison plots saved to: {plot_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Traffic Monitoring System - Edge-Cloud Processing')
    
    # Input arguments
    parser.add_argument('--video', type=str,
                       help='Path to input video file')
    parser.add_argument('--folder', type=str,
                       help='Path to folder containing videos')
    parser.add_argument('--output', type=str, default='data/results',
                       help='Output directory')
    
    # Processing mode
    parser.add_argument('--mode', type=str, default='hybrid',
                       choices=['edge', 'cloud', 'hybrid', 'compare'],
                       help='Processing mode')
    parser.add_argument('--bandwidth', type=int, default=1000,
                       help='Network bandwidth in KB/s')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Configuration file path')
    
    # Asymmetric processing
    parser.add_argument('--asymmetric', action='store_true',
                       help='Use asymmetric processing (5 edge || 1 cloud)')
    parser.add_argument('--workers', type=int, default=5,
                       help='Number of parallel edge workers (default: 5)')
    
    args = parser.parse_args()
    
    # Create directories
    ConfigLoader.create_directories()
    os.makedirs(args.output, exist_ok=True)
    
    # ========== ASYMMETRIC MODE ==========
    if args.asymmetric and args.folder:
        if not os.path.exists(args.folder):
            print(f"ERROR: Folder not found: {args.folder}")
            return
        
        run_asymmetric_processing(
            config_path=args.config,
            folder_path=args.folder,
            num_edge_workers=args.workers,
            output_dir=args.output
        )
        return
    
    # ========== BATCH FOLDER MODE ==========
    if args.folder:
        if not os.path.exists(args.folder):
            print(f"ERROR: Folder not found: {args.folder}")
            return
        
        print("\n📁 Processing folder sequentially...")
        system = TrafficMonitoringSystem(args.config)
        video_files = list(Path(args.folder).glob('*.mp4'))
        
        if not video_files:
            print(f"ERROR: No .mp4 files found in {args.folder}")
            return
        
        for i, video_file in enumerate(video_files):
            print(f"\n{'='*60}")
            print(f"Video {i+1}/{len(video_files)}: {video_file.name}")
            print(f"{'='*60}")
            
            video_output = os.path.join(args.output, video_file.stem)
            system.metrics = MetricsCollector()  # Reset metrics
            
            try:
                system.process_video(str(video_file), video_output, 
                                   args.bandwidth, args.mode)
            except Exception as e:
                print(f"ERROR processing {video_file.name}: {e}")
                continue
        
        print(f"\n✅ Batch processing complete! Results in: {args.output}")
        return
    
    # ========== SINGLE VIDEO MODE ==========
    if not args.video:
        print("ERROR: Either --video or --folder must be specified")
        parser.print_help()
        return
    
    if not os.path.exists(args.video):
        print(f"ERROR: Video file not found: {args.video}")
        print("\nPlace your video in: data/input_videos/")
        return
    
    # Initialize system
    print("="*60)
    print("TRAFFIC MONITORING SYSTEM")
    print("Smart Traffic Monitoring Using CV and Edge Computing")
    print("="*60 + "\n")
    
    system = TrafficMonitoringSystem(args.config)
    
    # Process video
    if args.mode == 'compare':
        print("Running comparison across all modes and bandwidths...")
        results = system.compare_modes(args.video, args.output)
        print("\n=== COMPARISON COMPLETE ===")
        print(results.to_string())
    else:
        system.process_video(args.video, args.output,
                           args.bandwidth, args.mode)
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print(f"Results saved to: {args.output}")
    print("="*60)


if __name__ == '__main__':
    main()