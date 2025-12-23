# Traffic Monitoring System - Complete Setup Guide

## 📁 Complete Folder Structure

```
traffic_monitoring/
├── main.py                          ⭐ Main entry point
├── README.md                        📄 This file
├── requirements.txt                 📦 Python dependencies
│
├── config/
│   └── config.yaml                  ⚙️ Configuration file
│
├── src/
│   ├── __init__.py                  (empty file)
│   ├── edge_node.py                 🔵 Edge processing node
│   ├── cloud_node.py                ☁️ Cloud processing node
│   ├── network_simulator.py         🌐 Network simulation
│   ├── metrics.py                   ⭐ Metrics collection (UPDATED)
│   ├── utils.py                     ⭐ Utilities (UPDATED)
│   └── asymmetric_processor.py      ⭐ Parallel processor (NEW)
│
├── data/
│   ├── input_videos/                📹 Place your videos here
│   │   ├── video1.mp4
│   │   ├── video2.mp4
│   │   └── ...
│   │
│   └── results/                     📊 Processing results
│       ├── asymmetric/              (5 edge || 1 cloud results)
│       ├── edge/                    (Edge-only results)
│       ├── cloud/                   (Cloud-only results)
│       └── hybrid/                  (Hybrid results)
│
└── models/
    └── yolov8n.pt                   🤖 YOLO model (download)
```

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install opencv-python numpy pandas matplotlib pyyaml ultralytics psutil
```

Or use requirements.txt:

```bash
pip install -r requirements.txt
```

### Step 2: Download YOLO Model

```bash
# The model will auto-download on first run, or manually:
mkdir models
cd models
# Download from: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

### Step 3: Add Your Videos

```bash
# Place your videos in:
data/input_videos/
```

### Step 4: Run the System

```bash
# Single video - Edge mode
python main.py --video data/input_videos/video1.mp4 --mode edge

# Single video - Hybrid mode
python main.py --video data/input_videos/video1.mp4 --mode hybrid --bandwidth 300

# Asymmetric parallel (5 edge || 1 cloud)
python main.py --folder data/input_videos --asymmetric --workers 5
```

---

## 📊 Usage Examples

### 1. Single Video Processing

```bash
# Edge only
python main.py --video data/input_videos/test.mp4 --mode edge --output data/results/test_edge

# Cloud only
python main.py --video data/input_videos/test.mp4 --mode cloud --output data/results/test_cloud

# Hybrid with 300 kbps bandwidth
python main.py --video data/input_videos/test.mp4 --mode hybrid --bandwidth 300
```

### 2. Compare All Modes

```bash
python main.py --video data/input_videos/test.mp4 --mode compare
```

This will test the video with:
- Edge mode
- Cloud mode  
- Hybrid mode at different bandwidths (1000, 500, 300, 200 kbps)

### 3. Batch Processing (Sequential)

```bash
# Process all videos in folder sequentially
python main.py --folder data/input_videos --mode hybrid --bandwidth 300
```

### 4. Asymmetric Parallel Processing ⭐ RECOMMENDED

```bash
# 5 edge workers (parallel) vs 1 cloud worker (sequential)
python main.py --folder data/input_videos --asymmetric --workers 5

# Custom number of workers
python main.py --folder data/input_videos --asymmetric --workers 3

# Custom output directory
python main.py --folder data/input_videos --asymmetric --workers 5 --output data/results/test1
```

---

## 📈 Understanding the Results

### Asymmetric Processing Output

```
data/results/asymmetric/
├── edge_parallel/
│   ├── edge_results.csv              # Per-video metrics
│   └── edge_metrics.json             # Aggregate statistics
│
├── cloud_sequential/
│   ├── cloud_results.csv             # Per-video metrics
│   └── cloud_metrics.json            # Aggregate statistics
│
├── asymmetric_comparison.png         # ⭐ 8-plot comparison
├── asymmetric_report.txt             # ⭐ Detailed report
└── asymmetric_data.json              # ⭐ Complete data
```

### Key Metrics

**Processing Time:**
- Edge: ~28ms per frame
- Cloud: ~80ms per frame
- Speedup: 2-3x faster with edge

**Resource Usage:**
- Edge CPU: ~35%
- Cloud CPU: ~68%
- Edge RAM: ~2.3 GB
- Cloud RAM: ~4.1 GB

**Detection Accuracy:**
- Edge: ~95-97% (vs cloud baseline)
- Cloud: 100% (reference)

---

## ⚙️ Configuration

Edit `config/config.yaml` to adjust:

### Network Bandwidth Threshold
```yaml
system:
  network_threshold: 300  # Switch to cloud above this (kbps)
```

### Edge Resolution (faster but less accurate)
```yaml
edge:
  resolution: [320, 180]  # Lower = faster
```

### Cloud Resolution (slower but more accurate)
```yaml
cloud:
  resolution: [1280, 720]  # Higher = better accuracy
```

### Detection Thresholds
```yaml
detection:
  congestion_threshold: 10      # Vehicles for congestion
  min_speed: 1                  # Minimum valid speed (mph)
  max_speed: 150                # Maximum valid speed (mph)
```




## 📊 Analyzing Results

### View Results in Python

```python
import pandas as pd
import json

# Load edge results
edge_df = pd.read_csv('data/results/asymmetric/edge_parallel/edge_results.csv')
print(edge_df.head())

# Load comparison data
with open('data/results/asymmetric/asymmetric_data.json') as f:
    data = json.load(f)

print(f"Speedup: {data['comparison']['speedup']:.2f}x")
print(f"CPU Reduction: {data['comparison']['cpu_reduction_percent']:.1f}%")
```

### View Plots

```bash
# Open comparison plot
start data/results/asymmetric/asymmetric_comparison.png

# Read detailed report
notepad data/results/asymmetric/asymmetric_report.txt
```

---

## 🎯 Performance Expectations

### For 20 Videos (8-core CPU, 16GB RAM):

| Mode | Workers | Time | CPU | RAM | Throughput |
|------|---------|------|-----|-----|------------|
| **Edge Parallel** | 5 | ~45s | 35% | 2.3GB | 0.44 video/s |
| **Cloud Sequential** | 1 | ~130s | 68% | 4.1GB | 0.15 video/s |
| **Speedup** | - | **2.9x** | **-48%** | **-44%** | **2.9x** |

---

## 📝 File Descriptions

### Core Files (Required)

| File | Purpose | Status |
|------|---------|--------|
| `main.py` | Entry point, video processing | ⭐ Updated |
| `src/metrics.py` | Metrics collection | ⭐ Updated |
| `src/utils.py` | Configuration, file management | ⭐ Updated |
| `src/asymmetric_processor.py` | Parallel processing | ⭐ New |
| `config/config.yaml` | System configuration | ⭐ Updated |


### Statistics to Report:
Example : 
- Speedup: 2.9x
- CPU Reduction: 48.6%
- RAM Reduction: 43.9%
- Edge Accuracy: 97.4% (vs cloud)
- Throughput: 0.442 videos/sec (edge) vs 0.152 (cloud)

---

Example Run command:

```bash
python main.py --folder data/input_videos --asymmetric --workers 5
```


