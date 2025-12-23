import cv2
import numpy as np
import time
import os
from typing import Dict, List, Tuple
from ultralytics import YOLO
import torch

class CloudNode:
    """Cloud computing node with YOLO detection"""
    
    def __init__(self, config: dict):
        self.config = config['cloud']
        self.detection_config = config['detection']
        self.resolution = tuple(self.config['resolution'])
        
        # Load YOLO model
        model_path = 'models/yolov10n.pt'
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            print("Downloading YOLOv8n model...")
            self.model = YOLO('yolov8n.pt')
            os.makedirs('models', exist_ok=True)
            self.model.save(model_path)
        
        # GPU support
        self.device = 'cuda' if torch.cuda.is_available() and self.config['use_gpu'] else 'cpu'
        print(f"Cloud node using device: {self.device}")
        self.model.to(self.device)
        
        # Vehicle classes
        self.vehicle_names = ['car', 'motorcycle', 'bus', 'truck', 'bicycle']
        self.vehicle_classes = [
            k for k, v in self.model.names.items() 
            if v.lower() in self.vehicle_names
        ]
        
        # Tracking
        self.vehicle_tracks = {}
        self.next_id = 0
        self.congestion_frames = 0
        self.fps = config['system']['fps']
        self.pixels_per_meter = config['calibration']['pixels_per_meter']
        self.frame_count = 0
        
        # Background subtractor for congestion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.config['gmm_history'],
            varThreshold=self.config['gmm_var_threshold'],
            detectShadows=True
        )
        
        # Global Foreground Model
        self.gfm_model = None
        self.gfm_frame_count = 0
        
    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process frame with YOLO"""
        self.frame_count += 1
        frame_resized = cv2.resize(frame, self.resolution)
        
        return self._process_with_yolo(frame_resized)
    
    def _process_with_yolo(self, frame: np.ndarray) -> Dict:
        """Process frame using YOLO detection"""
        # YOLO detection
        results = self.model(frame, conf=0.3, verbose=False)[0]
        
        vehicles = []
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls in self.vehicle_classes:
                conf = float(box.conf[0])
                if conf > 0.3:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    vehicles.append({
                        'bbox': (int(x1), int(y1), int(x2-x1), int(y2-y1)),
                        'centroid': (int((x1+x2)/2), int((y1+y2)/2)),
                        'confidence': conf,
                        'class_name': results.names[cls]
                    })
        
        # Fallback to background subtraction if YOLO fails
        if len(vehicles) == 0:
            vehicles = self._gmm_fallback(frame)
        
        # Track and estimate speeds
        speeds = self._track_vehicles(vehicles)
        
        # Congestion detection using background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Update GFM
        if self.gfm_model is None:
            self.gfm_model = fg_mask.astype(np.float32)
        else:
            alpha = 0.01
            self.gfm_model = cv2.addWeighted(
                self.gfm_model, 1-alpha, 
                fg_mask.astype(np.float32), alpha, 0
            )
        self.gfm_frame_count += 1
        
        # Detect congestion
        congestion = self._detect_congestion(fg_mask, vehicles)
        
        if self.frame_count % 30 == 0:
            print(f"Cloud frame {self.frame_count}: {len(vehicles)} vehicles, "
                  f"{np.mean(speeds) if speeds else 0:.1f} mph avg")
        
        return {
            'congestion': congestion,
            'vehicle_count': len(vehicles),
            'speeds': speeds,
            'avg_speed': np.mean(speeds) if speeds else 0,
            'detections': vehicles,
            'processing_node': 'cloud'
        }
    
    def _gmm_fallback(self, frame: np.ndarray) -> List[Dict]:
        """Fallback detection using GMM background subtraction"""
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        vehicles = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if (self.detection_config['min_vehicle_area'] < area < 
                self.detection_config['max_vehicle_area']):
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                if 0.3 < aspect_ratio < 5.0:
                    vehicles.append({
                        'bbox': (x, y, w, h),
                        'centroid': (x + w//2, y + h//2),
                        'confidence': 0.6,
                        'class_name': 'vehicle'
                    })
        
        return vehicles
    
    def _track_vehicles(self, vehicles: List[Dict]) -> List[float]:
        """Track vehicles and estimate speeds"""
        speeds = []
        matched = set()
        
        for vehicle in vehicles:
            best_match = None
            min_dist = float('inf')
            
            for tid, track in self.vehicle_tracks.items():
                if tid in matched:
                    continue
                dist = np.linalg.norm(
                    np.array(vehicle['centroid']) - np.array(track['positions'][-1])
                )
                if dist < 150 and dist < min_dist:
                    min_dist = dist
                    best_match = tid
            
            if best_match:
                self.vehicle_tracks[best_match]['positions'].append(vehicle['centroid'])
                matched.add(best_match)
                
                # Calculate speed
                if len(self.vehicle_tracks[best_match]['positions']) >= 5:
                    pos = self.vehicle_tracks[best_match]['positions'][-5:]
                    dist = sum(np.linalg.norm(np.array(pos[i+1]) - np.array(pos[i])) 
                              for i in range(len(pos)-1))
                    speed_pps = (dist / 4) * self.fps
                    speed_mph = (speed_pps / self.pixels_per_meter) * 2.23694
                    if 1 < speed_mph < 150:
                        speeds.append(speed_mph)
            else:
                self.vehicle_tracks[self.next_id] = {
                    'positions': [vehicle['centroid']],
                    'start_time': time.time()
                }
                self.next_id += 1
        
        # Clean old tracks
        current_time = time.time()
        self.vehicle_tracks = {
            tid: track for tid, track in self.vehicle_tracks.items()
            if current_time - track['start_time'] < 10
        }
        
        return speeds
    
    def _detect_congestion(self, fg_mask: np.ndarray, 
                          vehicles: List[Dict]) -> bool:
        """Detect traffic congestion"""
        stopped_count = 0
        
        # Check if vehicles are in stopped areas
        for vehicle in vehicles:
            x, y, w, h = vehicle['bbox']
            y_end = min(y+h, fg_mask.shape[0])
            x_end = min(x+w, fg_mask.shape[1])
            
            if y < fg_mask.shape[0] and x < fg_mask.shape[1]:
                roi = fg_mask[y:y_end, x:x_end]
                if roi.size > 0 and np.sum(roi > 0) > (w * h * 0.3):
                    stopped_count += 1
        
        # Update congestion counter
        if stopped_count >= self.detection_config['congestion_vehicle_threshold']:
            self.congestion_frames += 1
        else:
            self.congestion_frames = max(0, self.congestion_frames - 1)
        
        # Check if congestion persists
        threshold = self.detection_config['congestion_time_threshold'] * self.fps
        return self.congestion_frames > threshold