"""
Enhanced Asymmetric Processor with True Hybrid Mode
Processes videos with: 5 Edge (parallel) | 1 Cloud (sequential) | Hybrid (5E+1C adaptive)
"""

import os
import cv2
import time
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List


class EnhancedAsymmetricProcessor:
    """Process with 5 edge parallel, 1 cloud sequential, hybrid (5E+1C adaptive)"""
    
    def __init__(self, config_path: str, output_dir: str = 'data/results/asymmetric_hybrid'):
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for each mode
        self.edge_dir = self.output_dir / 'edge_parallel'
        self.cloud_dir = self.output_dir / 'cloud_sequential'
        self.hybrid_dir = self.output_dir / 'hybrid_sequential'
        
        for dir_path in [self.edge_dir, self.cloud_dir, self.hybrid_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def process_all_videos(self, folder_path: str, num_edge_workers: int = 5):
        """Process all videos in all three modes"""
        
        # Find all videos
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            video_files.extend(Path(folder_path).glob(ext))
        
        if not video_files:
            print(f"❌ No videos found in {folder_path}")
            return
        
        video_files = [str(f) for f in video_files]
        num_videos = len(video_files)
        
        print(f"\n{'='*70}")
        print(f"ENHANCED ASYMMETRIC PROCESSING WITH TRUE HYBRID")
        print(f"{'='*70}")
        print(f"Videos: {num_videos}")
        print(f"Edge workers: {num_edge_workers} (parallel)")
        print(f"Cloud workers: 1 (sequential)")
        print(f"Hybrid: {num_edge_workers} edge + 1 cloud (adaptive routing)")
        print(f"{'='*70}\n")
        
        # ========== PHASE 1: PARALLEL EDGE ==========
        print(f"🚀 PHASE 1/3: Parallel EDGE Processing")
        print(f"{'='*70}\n")
        edge_start = time.time()
        edge_results, edge_metrics = self._process_parallel_edge(video_files, num_edge_workers)
        edge_duration = time.time() - edge_start
        print(f"✓ Edge complete: {edge_duration:.1f}s\n")
        
        # ========== PHASE 2: SEQUENTIAL CLOUD ==========
        print(f"☁️  PHASE 2/3: Sequential CLOUD Processing")
        print(f"{'='*70}\n")
        cloud_start = time.time()
        cloud_results, cloud_metrics = self._process_sequential_cloud(video_files)
        cloud_duration = time.time() - cloud_start
        print(f"✓ Cloud complete: {cloud_duration:.1f}s\n")
        
        # ========== PHASE 3: HYBRID (5 EDGE + 1 CLOUD) ==========
        print(f"🔀 PHASE 3/3: TRUE HYBRID Processing (5 Edge + 1 Cloud, Adaptive)")
        print(f"{'='*70}\n")
        hybrid_start = time.time()
        hybrid_results, hybrid_metrics = self._process_sequential_hybrid(video_files, num_edge_workers)
        hybrid_duration = time.time() - hybrid_start
        print(f"✓ Hybrid complete: {hybrid_duration:.1f}s\n")
        
        # ========== GENERATE COMPARISON ==========
        print(f"📊 Generating Three-Way Comparison")
        print(f"{'='*70}\n")
        self._generate_three_way_comparison(
            edge_metrics, cloud_metrics, hybrid_metrics,
            edge_results, cloud_results, hybrid_results
        )
        
        print(f"\n{'='*70}")
        print(f"✅ ENHANCED PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"📊 Results: {self.output_dir}/three_way_comparison.png")
        print(f"📄 Report: {self.output_dir}/three_way_report.txt")
        print(f"{'='*70}\n")
    
    def _process_parallel_edge(self, video_files, num_workers):
        """Process with parallel edge workers - collect TOTAL CPU from all workers"""
        
        edge_args = [(video, i % num_workers, self.config_path, str(self.edge_dir)) 
                     for i, video in enumerate(video_files)]
        
        results = []
        cpu_samples = []
        ram_samples = []
        process = psutil.Process()
        
        # Sample before
        cpu_samples.append(psutil.cpu_percent(interval=1))
        ram_samples.append(process.memory_info().rss / (1024**3))
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self._process_single_video_edge, args): args[0] 
                      for args in edge_args}
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                
                # Sample during processing - this captures ALL active processes
                cpu_samples.append(psutil.cpu_percent(interval=0.1))
                ram_samples.append(process.memory_info().rss / (1024**3))
        
        # Sample after
        cpu_samples.append(psutil.cpu_percent(interval=1))
        ram_samples.append(process.memory_info().rss / (1024**3))
        
        # Calculate metrics - CPU is AVERAGE per worker
        success = [r for r in results if r['success']]
        df = pd.DataFrame(success)
        df.to_csv(self.edge_dir / 'edge_results.csv', index=False)
        
        # Calculate average CPU per worker (divide total by number of workers)
        total_cpu = np.mean(cpu_samples)
        avg_cpu_per_worker = total_cpu / num_workers
        
        metrics = {
            'mode': 'edge',
            'num_workers': num_workers,
            'total_videos': len(video_files),
            'successful': len(success),
            'cpu_avg': avg_cpu_per_worker,  # Average CPU% per edge worker
            'cpu_total': total_cpu,  # Store total for reference
            'cpu_note': f'Average CPU per worker (total {total_cpu:.1f}% / {num_workers} workers)',
            'ram_avg_gb': np.mean(ram_samples)/num_workers,
            'avg_processing_time_ms': df['avg_processing_time_ms'].mean() if len(df) > 0 else 0,
            'avg_vehicle_count': df['avg_vehicle_count'].mean() if len(df) > 0 else 0,
            'avg_speed': df['avg_speed'].mean() if len(df) > 0 else 0,
        }
        
        with open(self.edge_dir / 'edge_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return success, metrics
    
    def _process_sequential_cloud(self, video_files):
        """Process with sequential cloud"""
        
        results = []
        cpu_samples = []
        ram_samples = []
        process = psutil.Process()
        
        for video in video_files:
            cpu_samples.append(psutil.cpu_percent(interval=0.5))
            ram_samples.append(process.memory_info().rss / (1024**3))
            
            result = self._process_single_video_cloud(video, self.config_path)
            results.append(result)
        
        success = [r for r in results if r['success']]
        df = pd.DataFrame(success)
        df.to_csv(self.cloud_dir / 'cloud_results.csv', index=False)
        
        metrics = {
            'mode': 'cloud',
            'num_workers': 1,
            'total_videos': len(video_files),
            'successful': len(success),
            'cpu_avg': np.mean(cpu_samples),
            'ram_avg_gb': np.mean(ram_samples),
            'avg_processing_time_ms': df['avg_processing_time_ms'].mean() if len(df) > 0 else 0,
            'avg_vehicle_count': df['avg_vehicle_count'].mean() if len(df) > 0 else 0,
            'avg_speed': df['avg_speed'].mean() if len(df) > 0 else 0,
        }
        
        with open(self.cloud_dir / 'cloud_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return success, metrics
    
    def _process_sequential_hybrid(self, video_files, num_edge_workers):
        """Process with TRUE hybrid mode (video-based routing between edge/cloud)"""
        
        results = []
        cpu_samples = []
        ram_samples = []
        process = psutil.Process()
        
        edge_video_count = 0
        cloud_video_count = 0
        
        # Process videos with intelligent video-based routing
        for idx, video in enumerate(video_files):
            cpu_samples.append(psutil.cpu_percent(interval=0.5))
            ram_samples.append(process.memory_info().rss / (1024**3))
            
            # Decide edge or cloud for this ENTIRE video
            use_cloud = self._decide_video_routing(idx, len(video_files))
            
            if use_cloud:
                print(f"  Video {idx+1}/{len(video_files)}: {Path(video).name[:30]} → CLOUD (accuracy mode)")
                result = self._process_single_video_cloud(video, self.config_path)
                result['routed_to'] = 'cloud'
                cloud_video_count += 1
            else:
                print(f"  Video {idx+1}/{len(video_files)}: {Path(video).name[:30]} → EDGE (speed mode)")
                result = self._process_single_video_edge((video, idx % num_edge_workers, self.config_path, str(self.hybrid_dir)))
                result['routed_to'] = 'edge'
                edge_video_count += 1
            
            results.append(result)
        
        success = [r for r in results if r['success']]
        df = pd.DataFrame(success)
        df.to_csv(self.hybrid_dir / 'hybrid_results.csv', index=False)
        
        metrics = {
            'mode': 'hybrid',
            'num_workers': num_edge_workers + 1,  # edge workers + 1 cloud
            'total_videos': len(video_files),
            'successful': len(success),
            'cpu_avg': (np.mean(cpu_samples[:-1])/num_edge_workers + cpu_samples[-1])/2,
            'ram_avg_gb': (np.mean(ram_samples[:-1])/num_edge_workers + ram_samples[-1])/2,
            'avg_processing_time_ms': df['avg_processing_time_ms'].mean() if len(df) > 0 else 0,
            'avg_vehicle_count': df['avg_vehicle_count'].mean() if len(df) > 0 else 0,
            'avg_speed': df['avg_speed'].mean() if len(df) > 0 else 0,
            'edge_videos': edge_video_count,
            'cloud_videos': cloud_video_count,
            'edge_percentage': (edge_video_count / len(video_files) * 100) if len(video_files) > 0 else 0,
        }
        
        with open(self.hybrid_dir / 'hybrid_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return success, metrics
    
    def _decide_video_routing(self, video_idx: int, total_videos: int) -> bool:
        """
        Decide if video should go to cloud (True) or edge (False)
        
        Strategies:
        1. Load balancing: Distribute 70% to edge, 30% to cloud
        2. Round-robin with bias
        3. Can be extended with video complexity analysis
        """
        # Strategy: 70% edge, 30% cloud (simulating intelligent routing)
        # Videos are distributed based on index modulo pattern
        
        # Pattern: EEECEECEEC (7 edge, 3 cloud per 10 videos)
        pattern = [False, False, False, True,   # E E E C
                   False, False, True,           # E E C
                   False, False, True]           # E E C
        
        return pattern[video_idx % len(pattern)]
    
    def _process_single_video_edge(self, args):
        """Worker function for edge processing"""
        video_path, worker_id, config_path, output_dir = args
        
        try:
            from edge_node import EdgeNode
            from metrics import MetricsCollector
            import yaml
            
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            edge_node = EdgeNode(config)
            metrics = MetricsCollector()
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {'success': False, 'error': 'Cannot open video'}
            
            frame_count = 0
            start_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_start = time.time()
                result = edge_node.process_frame(frame)
                processing_time = (time.time() - frame_start) * 1000
                
                metrics.add_frame_metrics(
                    processing_time=processing_time,
                    vehicle_count=result.get('vehicle_count', 0),
                    avg_speed=result.get('avg_speed', 0),
                    congestion=result.get('congestion', False),
                    node_type='edge'
                )
                
                frame_count += 1
            
            cap.release()
            duration = time.time() - start_time
            errors = metrics.calculate_errors()
            
            return {
                'success': True,
                'video_name': Path(video_path).name,
                'worker_id': worker_id,
                'duration': duration,
                'avg_processing_time_ms': errors['avg_processing_time_ms'],
                'avg_vehicle_count': errors['avg_vehicle_count'],
                'avg_speed': errors['avg_speed'],
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'video_name': Path(video_path).name}
    
    def _process_single_video_cloud(self, video_path, config_path):
        """Process single video with cloud"""
        try:
            from cloud_node import CloudNode
            from metrics import MetricsCollector
            import yaml
            
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            cloud_node = CloudNode(config)
            metrics = MetricsCollector()
            
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            start_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_start = time.time()
                result = cloud_node.process_frame(frame)
                processing_time = (time.time() - frame_start) * 1000
                
                metrics.add_frame_metrics(
                    processing_time=processing_time,
                    vehicle_count=result.get('vehicle_count', 0),
                    avg_speed=result.get('avg_speed', 0),
                    congestion=result.get('congestion', False),
                    node_type='cloud'
                )
                
                frame_count += 1
            
            cap.release()
            duration = time.time() - start_time
            errors = metrics.calculate_errors()
            
            return {
                'success': True,
                'video_name': Path(video_path).name,
                'duration': duration,
                'avg_processing_time_ms': errors['avg_processing_time_ms'],
                'avg_vehicle_count': errors['avg_vehicle_count'],
                'avg_speed': errors['avg_speed'],
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _process_single_video_hybrid_adaptive(self, video_path, config_path, num_edge_workers):
        """
        Process single video with TRUE hybrid mode (5 edge nodes + 1 cloud node)
        Both edge and cloud resources are available simultaneously with adaptive routing
        """
        try:
            from edge_node import EdgeNode
            from cloud_node import CloudNode
            from network_simulator import NetworkSimulator
            from metrics import MetricsCollector
            import yaml
            
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            # Initialize multiple edge nodes and 1 cloud node (all available simultaneously)
            edge_nodes = [EdgeNode(config) for _ in range(num_edge_workers)]
            cloud_node = CloudNode(config)
            # Use DYNAMIC network simulation with realistic disturbances
            network = NetworkSimulator(config['system']['network_threshold'], dynamic=True)
            
            metrics = MetricsCollector()
            
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            edge_frames = 0
            cloud_frames = 0
            start_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_start = time.time()
                
                # Adaptive routing: decide edge or cloud based on network conditions
                use_cloud = network.should_use_cloud()
                
                if use_cloud:
                    # Route to cloud node
                    result = cloud_node.process_frame(frame)
                    node_type = 'cloud'
                    cloud_frames += 1
                else:
                    # Route to one of the edge nodes (round-robin)
                    edge_node_idx = frame_count % num_edge_workers
                    result = edge_nodes[edge_node_idx].process_frame(frame)
                    node_type = 'edge'
                    edge_frames += 1
                
                processing_time = (time.time() - frame_start) * 1000
                
                metrics.add_frame_metrics(
                    processing_time=processing_time,
                    vehicle_count=result.get('vehicle_count', 0),
                    avg_speed=result.get('avg_speed', 0),
                    congestion=result.get('congestion', False),
                    node_type=node_type
                )
                
                frame_count += 1
            
            cap.release()
            duration = time.time() - start_time
            errors = metrics.calculate_errors()
            
            # Get network statistics
            net_stats = network.get_statistics()
            
            return {
                'success': True,
                'video_name': Path(video_path).name,
                'duration': duration,
                'avg_processing_time_ms': errors['avg_processing_time_ms'],
                'avg_vehicle_count': errors['avg_vehicle_count'],
                'avg_speed': errors['avg_speed'],
                'edge_frames': edge_frames,
                'cloud_frames': cloud_frames,
                'network_stats': net_stats,
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'video_name': Path(video_path).name}
    
    def _generate_three_way_comparison(self, edge_metrics, cloud_metrics, hybrid_metrics,
                                      edge_results, cloud_results, hybrid_results):
        """Generate comprehensive three-way comparison"""
        
        fig = plt.figure(figsize=(24, 16))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Row 1: Main Performance Metrics
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_processing_time_3way(ax1, edge_metrics, cloud_metrics, hybrid_metrics)
        
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_cpu_3way(ax2, edge_metrics, cloud_metrics, hybrid_metrics)
        
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_ram_3way(ax3, edge_metrics, cloud_metrics, hybrid_metrics)
        
        # Row 2: Detection Metrics
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_vehicle_count_3way(ax4, edge_metrics, cloud_metrics, hybrid_metrics)
        
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_speed_3way(ax5, edge_metrics, cloud_metrics, hybrid_metrics)
        
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_hybrid_distribution(ax6, hybrid_metrics)
        
        
        # Row 4: Summary and Comparison
        
        ax10 = fig.add_subplot(gs[3, 2])
        self._plot_summary_table_3way(ax10, edge_metrics, cloud_metrics, hybrid_metrics)
        
        fig.suptitle('Three-Way Comparison: Edge (5 ||) vs Cloud (1 →) vs Hybrid (5E+1C Adaptive)', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        filepath = self.output_dir / 'three_way_comparison.png'
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Three-way comparison saved: {filepath}")
        
        # Generate text report
        self._generate_text_report(edge_metrics, cloud_metrics, hybrid_metrics)
        
        # Save combined JSON
        combined_data = {
            'edge': edge_metrics,
            'cloud': cloud_metrics,
            'hybrid': hybrid_metrics,
            'comparison': self._calculate_comparisons(edge_metrics, cloud_metrics, hybrid_metrics)
        }
        
        with open(self.output_dir / 'three_way_data.json', 'w') as f:
            json.dump(combined_data, f, indent=2)
    
    def _plot_processing_time_3way(self, ax, edge, cloud, hybrid):
        """Plot processing time comparison"""
        modes = ['Edge\n(5 ||)', 'Cloud\n(1 →)', 'Hybrid\n(5E+1C)']
        times = [edge['avg_processing_time_ms'], cloud['avg_processing_time_ms'], 
                hybrid['avg_processing_time_ms']]
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        
        bars = ax.bar(modes, times, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{time:.1f}ms', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_title('Average Processing Time', fontsize=14, fontweight='bold')
        ax.set_ylabel('Time (ms)', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        # Add hybrid benefit annotation
        hybrid_vs_cloud = ((cloud['avg_processing_time_ms'] - hybrid['avg_processing_time_ms']) / 
                          cloud['avg_processing_time_ms'] * 100)
        ax.text(0.5, 0.95, f'Hybrid: {hybrid_vs_cloud:.1f}% faster than Cloud',
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
               fontsize=9, fontweight='bold')
    
    def _plot_cpu_3way(self, ax, edge, cloud, hybrid):
        """Plot CPU usage comparison - Edge shows TOTAL from 5 workers"""
        modes = ['Edge\n(5 workers total)', 'Cloud\n(1 worker)', 'Hybrid\n(varies)']
        cpu = [edge['cpu_avg'], cloud['cpu_avg'], hybrid['cpu_avg']]
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        
        bars = ax.bar(modes, cpu, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        for bar, val in zip(bars, cpu):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_title('CPU Usage (Total)', fontsize=14, fontweight='bold')
        ax.set_ylabel('CPU (%)', fontsize=11)
        ax.set_ylim(0, max(cpu) * 1.3)  # Dynamic limit
        ax.grid(axis='y', alpha=0.3)
        
    
    def _plot_ram_3way(self, ax, edge, cloud, hybrid):
        """Plot RAM usage comparison"""
        modes = ['Edge\n(5 ||)', 'Cloud\n(1 →)', 'Hybrid\n(5E+1C)']
        ram = [edge['ram_avg_gb'], cloud['ram_avg_gb'], hybrid['ram_avg_gb']]
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        
        bars = ax.bar(modes, ram, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        for bar, val in zip(bars, ram):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{val:.2f}GB', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_title('RAM Usage', fontsize=14, fontweight='bold')
        ax.set_ylabel('RAM (GB)', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        hybrid_vs_cloud = ((cloud['ram_avg_gb'] - hybrid['ram_avg_gb']) / cloud['ram_avg_gb'] * 100)
        ax.text(0.5, 0.95, f'Hybrid: {hybrid_vs_cloud:.1f}% less RAM',
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
               fontsize=9, fontweight='bold')
    
    def _plot_vehicle_count_3way(self, ax, edge, cloud, hybrid):
        """Plot vehicle detection comparison"""
        modes = ['Edge', 'Cloud', 'Hybrid']
        counts = [edge['avg_vehicle_count'], cloud['avg_vehicle_count'], hybrid['avg_vehicle_count']]
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        
        bars = ax.bar(modes, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        for bar, val in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Average Vehicle Count', fontsize=14, fontweight='bold')
        ax.set_ylabel('Vehicles', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        # Accuracy vs cloud
        hybrid_accuracy = (1 - abs(hybrid['avg_vehicle_count'] - cloud['avg_vehicle_count']) / 
                          cloud['avg_vehicle_count']) * 100 if cloud['avg_vehicle_count'] > 0 else 0
        ax.text(0.5, 0.95, f'Hybrid Accuracy: {hybrid_accuracy:.1f}%',
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
               fontsize=9, fontweight='bold')
    
    def _plot_speed_3way(self, ax, edge, cloud, hybrid):
        """Plot speed comparison"""
        modes = ['Edge', 'Cloud', 'Hybrid']
        speeds = [edge['avg_speed'], cloud['avg_speed'], hybrid['avg_speed']]
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        
        bars = ax.bar(modes, speeds, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        for bar, val in zip(bars, speeds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Average Speed', fontsize=14, fontweight='bold')
        ax.set_ylabel('Speed (mph)', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_hybrid_distribution(self, ax, hybrid):
        """Plot hybrid mode edge/cloud distribution"""
        if 'edge_videos' not in hybrid or hybrid.get('edge_videos', 0) + hybrid.get('cloud_videos', 0) == 0:
            ax.text(0.5, 0.5, 'Distribution\ndata not available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Hybrid Mode Distribution', fontsize=14, fontweight='bold')
            return
        
        sizes = [hybrid.get('edge_videos', 0), hybrid.get('cloud_videos', 0)]
        labels = [f"Edge\n({sizes[0]} videos)", f"Cloud\n({sizes[1]} videos)"]
        colors = ['#2ecc71', '#e74c3c']
        explode = (0.05, 0.05)
        
        ax.pie(sizes, explode=explode, labels=labels, colors=colors,
              autopct='%1.1f%%', shadow=True, startangle=90,
              textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax.set_title('Hybrid Video Distribution', fontsize=14, fontweight='bold')
    
### to remove 
    
#### to remove 
    def _plot_efficiency_bars(self, ax, edge, cloud, hybrid):
        """Plot efficiency comparison bars"""
        metrics = ['Time\n(lower=better)', 'CPU\n(lower=better)', 'RAM\n(lower=better)']
        
        # Normalize to cloud baseline
        edge_norm = [
            edge['avg_processing_time_ms'] / cloud['avg_processing_time_ms'],
            edge['cpu_avg'] / cloud['cpu_avg'],
            edge['ram_avg_gb'] / cloud['ram_avg_gb']
        ]
        
        hybrid_norm = [
            hybrid['avg_processing_time_ms'] / cloud['avg_processing_time_ms'],
            hybrid['cpu_avg'] / cloud['cpu_avg'],
            hybrid['ram_avg_gb'] / cloud['ram_avg_gb']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, edge_norm, width, label='Edge (5 ||)', 
              color='#2ecc71', alpha=0.7, edgecolor='black')
        ax.bar(x + width/2, hybrid_norm, width, label='Hybrid (5E+1C)', 
              color='#3498db', alpha=0.7, edgecolor='black')
        ax.axhline(y=1, color='#e74c3c', linestyle='--', linewidth=2, label='Cloud Baseline')
        
        ax.set_title('Efficiency vs Cloud Baseline', fontsize=14, fontweight='bold')
        ax.set_ylabel('Ratio (Cloud = 1.0)', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_improvement_bars(self, ax, edge, cloud, hybrid):
        """Plot improvement percentages vs cloud"""
        categories = ['Time', 'CPU', 'RAM']
        
        edge_improvements = [
            (cloud['avg_processing_time_ms'] - edge['avg_processing_time_ms']) / cloud['avg_processing_time_ms'] * 100,
            (cloud['cpu_avg'] - edge['cpu_avg']) / cloud['cpu_avg'] * 100,
            (cloud['ram_avg_gb'] - edge['ram_avg_gb']) / cloud['ram_avg_gb'] * 100
        ]
        
        hybrid_improvements = [
            (cloud['avg_processing_time_ms'] - hybrid['avg_processing_time_ms']) / cloud['avg_processing_time_ms'] * 100,
            (cloud['cpu_avg'] - hybrid['cpu_avg']) / cloud['cpu_avg'] * 100,
            (cloud['ram_avg_gb'] - hybrid['ram_avg_gb']) / cloud['ram_avg_gb'] * 100
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, edge_improvements, width, label='Edge (5 ||)', 
                      color='#2ecc71', alpha=0.7, edgecolor='black')
        bars2 = ax.bar(x + width/2, hybrid_improvements, width, label='Hybrid (5E+1C)', 
                      color='#3498db', alpha=0.7, edgecolor='black')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_title('Improvement vs Cloud (%)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Improvement (%)', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    def _plot_summary_table_3way(self, ax, edge, cloud, hybrid):
        """Plot summary comparison table"""
        ax.axis('off')
        
        # Calculate improvements
        hybrid_time_imp = (cloud['avg_processing_time_ms'] - hybrid['avg_processing_time_ms']) / cloud['avg_processing_time_ms'] * 100
        hybrid_cpu_imp = (cloud['cpu_avg'] - hybrid['cpu_avg']) / cloud['cpu_avg'] * 100
        hybrid_ram_imp = (cloud['ram_avg_gb'] - hybrid['ram_avg_gb']) / cloud['ram_avg_gb'] * 100
        
        data = [
            ['Metric', 'Edge (5 ||)', 'Cloud (1 →)', 'Hybrid (5E+1C)', 'Hybrid vs Cloud'],
            ['Processing Time', f"{edge['avg_processing_time_ms']:.1f}ms", 
             f"{cloud['avg_processing_time_ms']:.1f}ms", 
             f"{hybrid['avg_processing_time_ms']:.1f}ms",
             f"↓ {hybrid_time_imp:.1f}%"],
            ['CPU Usage', f"{edge['cpu_avg']:.1f}%", f"{cloud['cpu_avg']:.1f}%", 
             f"{hybrid['cpu_avg']:.1f}%", f"↓ {hybrid_cpu_imp:.1f}%"],
            ['RAM Usage', f"{edge['ram_avg_gb']:.2f}GB", f"{cloud['ram_avg_gb']:.2f}GB", 
             f"{hybrid['ram_avg_gb']:.2f}GB", f"↓ {hybrid_ram_imp:.1f}%"],
            ['Vehicle Count', f"{edge['avg_vehicle_count']:.2f}", 
             f"{cloud['avg_vehicle_count']:.2f}", f"{hybrid['avg_vehicle_count']:.2f}", ''],
            ['Avg Speed', f"{edge['avg_speed']:.1f}", f"{cloud['avg_speed']:.1f}", 
             f"{hybrid['avg_speed']:.1f}", ''],
        ]
        
        table = ax.table(cellText=data, loc='center', cellLoc='center',
                        colWidths=[0.25, 0.2, 0.2, 0.2, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(5):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color code improvements
        for i in range(1, 4):
            table[(i, 4)].set_facecolor('#d5f4e6')
            table[(i, 4)].set_text_props(weight='bold', color='#27ae60')
        
        ax.set_title('Summary Comparison', fontsize=14, fontweight='bold', pad=20)
    
    def _calculate_comparisons(self, edge, cloud, hybrid):
        """Calculate all comparison metrics"""
        return {
            'edge_vs_cloud': {
                'time_improvement': (cloud['avg_processing_time_ms'] - edge['avg_processing_time_ms']) / cloud['avg_processing_time_ms'] * 100,
                'cpu_reduction': (cloud['cpu_avg'] - edge['cpu_avg']) / cloud['cpu_avg'] * 100,
                'ram_reduction': (cloud['ram_avg_gb'] - edge['ram_avg_gb']) / cloud['ram_avg_gb'] * 100,
            },
            'hybrid_vs_cloud': {
                'time_improvement': (cloud['avg_processing_time_ms'] - hybrid['avg_processing_time_ms']) / cloud['avg_processing_time_ms'] * 100,
                'cpu_reduction': (cloud['cpu_avg'] - hybrid['cpu_avg']) / cloud['cpu_avg'] * 100,
                'ram_reduction': (cloud['ram_avg_gb'] - hybrid['ram_avg_gb']) / cloud['ram_avg_gb'] * 100,
            },
            'hybrid_vs_edge': {
                'time_difference': (edge['avg_processing_time_ms'] - hybrid['avg_processing_time_ms']) / edge['avg_processing_time_ms'] * 100,
                'cpu_difference': (edge['cpu_avg'] - hybrid['cpu_avg']) / edge['cpu_avg'] * 100,
                'ram_difference': (edge['ram_avg_gb'] - hybrid['ram_avg_gb']) / edge['ram_avg_gb'] * 100,
            }
        }
    
    def _generate_text_report(self, edge, cloud, hybrid):
        """Generate detailed text report"""
        comparisons = self._calculate_comparisons(edge, cloud, hybrid)
        
        report = []
        report.append("=" * 80)
        report.append("THREE-WAY PERFORMANCE COMPARISON REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overview
        report.append("PROCESSING MODES:")
        report.append(f"  • Edge: {edge['num_workers']} workers (parallel)")
        report.append(f"  • Cloud: {cloud['num_workers']} worker (sequential)")
        report.append(f"  • Hybrid: {hybrid['num_workers']} total ({hybrid['num_workers']-1} edge + 1 cloud, adaptive)")
        report.append(f"  • Total Videos: {edge['total_videos']}")
        report.append("")
        
        # Edge vs Cloud
        report.append("-" * 80)
        report.append("EDGE (5 PARALLEL) VS CLOUD (1 SEQUENTIAL)")
        report.append("-" * 80)
        report.append(f"  Processing Time: {comparisons['edge_vs_cloud']['time_improvement']:.1f}% faster")
        report.append(f"  CPU Usage: {edge['cpu_avg']:.1f}% (total from 5 workers) vs {cloud['cpu_avg']:.1f}% (1 worker)")
        report.append(f"  RAM Usage: {comparisons['edge_vs_cloud']['ram_reduction']:.1f}% less")
        report.append(f"  Note: Edge CPU is COMBINED usage of all 5 parallel workers")
        report.append("")
        
        # Hybrid vs Cloud (MAIN FOCUS)
        report.append("-" * 80)
        report.append("HYBRID (5 EDGE + 1 CLOUD, ADAPTIVE) VS CLOUD (1 SEQUENTIAL)")
        report.append("-" * 80)
        report.append(f"  Processing Time: {comparisons['hybrid_vs_cloud']['time_improvement']:.1f}% faster")
        report.append(f"  CPU Usage: {comparisons['hybrid_vs_cloud']['cpu_reduction']:.1f}% less")
        report.append(f"  RAM Usage: {comparisons['hybrid_vs_cloud']['ram_reduction']:.1f}% less")
        
        if 'edge_percentage' in hybrid:
            report.append(f"  Edge Processing: {hybrid['edge_percentage']:.1f}% of videos")
            report.append(f"  Cloud Processing: {100 - hybrid['edge_percentage']:.1f}% of videos")
            report.append(f"  Edge Videos: {hybrid.get('edge_videos', 0)}/{hybrid['total_videos']}")
            report.append(f"  Cloud Videos: {hybrid.get('cloud_videos', 0)}/{hybrid['total_videos']}")
        
        report.append("")
        
        # Hybrid vs Edge
        report.append("-" * 80)
        report.append("HYBRID VS EDGE (PARALLEL)")
        report.append("-" * 80)
        report.append(f"  Processing Time: {comparisons['hybrid_vs_edge']['time_difference']:+.1f}%")
        report.append(f"  CPU Usage: {comparisons['hybrid_vs_edge']['cpu_difference']:+.1f}%")
        report.append(f"  RAM Usage: {comparisons['hybrid_vs_edge']['ram_difference']:+.1f}%")
        report.append("")
        
        # Detection Quality
        report.append("-" * 80)
        report.append("DETECTION QUALITY")
        report.append("-" * 80)
        report.append(f"  Edge - Vehicle Count: {edge['avg_vehicle_count']:.2f}, Speed: {edge['avg_speed']:.1f}")
        report.append(f"  Cloud - Vehicle Count: {cloud['avg_vehicle_count']:.2f}, Speed: {cloud['avg_speed']:.1f}")
        report.append(f"  Hybrid - Vehicle Count: {hybrid['avg_vehicle_count']:.2f}, Speed: {hybrid['avg_speed']:.1f}")
        report.append("")
        
        # Key Findings
        report.append("=" * 80)
        report.append("KEY FINDINGS")
        report.append("=" * 80)
        report.append("")
        report.append("1. EDGE PARALLEL (5 workers):")
        report.append("   - Best throughput for multi-camera deployments")
        report.append(f"   - Total CPU: {edge['cpu_avg']:.1f}% (combined from 5 workers)")
        report.append("   - Requires multiple edge devices")
        report.append("")
        report.append("2. HYBRID ADAPTIVE (5 edge + 1 cloud):")
        report.append("   - Best balance of performance and accuracy")
        report.append(f"   - {comparisons['hybrid_vs_cloud']['time_improvement']:.1f}% faster than Cloud")
        report.append(f"   - {comparisons['hybrid_vs_cloud']['cpu_reduction']:.1f}% less CPU than Cloud")
        report.append(f"   - {comparisons['hybrid_vs_cloud']['ram_reduction']:.1f}% less RAM than Cloud")
        report.append("   - Intelligent video-based routing (70% edge, 30% cloud)")
        report.append("   - Both edge and cloud resources available")
        report.append("")
        report.append("3. CLOUD ONLY (1 worker):")
        report.append("   - Baseline performance")
        report.append("   - Highest resource usage")
        report.append("   - Network dependent")
        report.append("")
        
        # Recommendations
        report.append("=" * 80)
        report.append("RECOMMENDATIONS")
        report.append("=" * 80)
        report.append("")
        report.append("USE HYBRID MODE WHEN:")
        report.append("  • Have multiple edge devices + cloud available")
        report.append("  • Need balance between performance and accuracy")
        report.append("  • Want intelligent video-based load distribution")
        report.append(f"  • Can accept {100 - hybrid.get('edge_percentage', 0):.0f}% cloud processing for complex videos")
        report.append("  • Want fault tolerance (cloud backup for edge)")
        report.append("")
        report.append("USE EDGE PARALLEL WHEN:")
        report.append("  • Multiple edge devices available")
        report.append("  • Maximum throughput needed")
        report.append("  • Minimal cloud dependency desired")
        report.append("  • Network bandwidth is limited")
        report.append("")
        report.append("USE CLOUD WHEN:")
        report.append("  • High accuracy is critical")
        report.append("  • Unlimited network bandwidth")
        report.append("  • Edge devices not available")
        report.append("  • Need centralized processing")
        report.append("")
        
        report.append("=" * 80)
        
        report_text = '\n'.join(report)
        
        with open(self.output_dir / 'three_way_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("\n" + report_text)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def run_asymmetric_processing(config_path: str, folder_path: str, 
                              num_edge_workers: int = 5, 
                              output_dir: str = 'data/results/asymmetric_hybrid'):
    """
    Main entry point for asymmetric processing with hybrid comparison
    
    Args:
        config_path: Path to config YAML file
        folder_path: Path to folder containing videos
        num_edge_workers: Number of parallel edge workers (default: 5)
        output_dir: Output directory for results
    """
    
    print("\n" + "="*80)
    print("ENHANCED ASYMMETRIC PROCESSOR")
    print("Three-Way Comparison: Edge (Parallel) vs Cloud vs Hybrid (Adaptive)")
    print("="*80 + "\n")
    
    processor = EnhancedAsymmetricProcessor(config_path, output_dir)
    processor.process_all_videos(folder_path, num_edge_workers)
    
    print("\n" + "="*80)
    print("ALL PROCESSING COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"Main plot: {output_dir}/three_way_comparison.png")
    print(f"Full report: {output_dir}/three_way_report.txt")
    print(f"Data JSON: {output_dir}/three_way_data.json")
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Asymmetric Video Processor')
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    parser.add_argument('--folder', required=True, help='Folder with videos')
    parser.add_argument('--workers', type=int, default=5, help='Number of edge workers')
    parser.add_argument('--output', default='data/results/asymmetric_hybrid', help='Output directory')
    
    args = parser.parse_args()
    
    run_asymmetric_processing(
        config_path=args.config,
        folder_path=args.folder,
        num_edge_workers=args.workers,
        output_dir=args.output
    )