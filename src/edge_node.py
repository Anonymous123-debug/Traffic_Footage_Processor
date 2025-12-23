import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional

class EdgeNode:
    """Edge computing node with lightweight processing"""
    
    def __init__(self, config: dict):
        self.config = config['edge']
        self.detection_config = config['detection']
        self.calibration = config['calibration']
        self.resolution = tuple(self.config['resolution'])
        
        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=25,
            detectShadows=True
        )
        
        # Vehicle tracking
        self.vehicle_tracks = {}
        self.next_id = 0
        self.congestion_frames = 0
        self.fps = config['system']['fps']
        self.frame_count = 0
        
        print(f"Edge Node initialized: {self.resolution}, "
              f"{self.config['color_space']} color space")
        
    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process frame with edge configuration (lightweight)"""
        self.frame_count += 1
        
        # Resize
        frame_resized = cv2.resize(frame, self.resolution)
        
        # Convert color space
        if self.config['color_space'] == 'gray':
            processed = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        else:
            processed = frame_resized
        
        # Background subtraction
        if len(processed.shape) == 2:
            processed_3ch = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        else:
            processed_3ch = processed
            
        fg_mask = self.bg_subtractor.apply(processed_3ch, 
                                          learningRate=self.config['learning_rate'])
        
        # Remove shadows
        fg_mask[fg_mask == 127] = 0
        
        # Optional morphology
        if self.config['morphological_ops']:
            fg_mask = self._apply_morphology(fg_mask)
        
        # Detect vehicles
        vehicles = self._detect_vehicles(fg_mask)
        
        # Track and estimate speeds
        speeds = self._track_and_estimate_speeds(vehicles)
        
        # Detect congestion
        congestion = self._detect_congestion(fg_mask, vehicles)
        
        return {
            'congestion': congestion,
            'vehicle_count': len(vehicles),
            'speeds': speeds,
            'avg_speed': np.mean(speeds) if speeds else 0,
            'detections': vehicles,
            'processing_node': 'edge',
            'foreground_mask': fg_mask
        }
    
    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask
    
    def _detect_vehicles(self, fg_mask: np.ndarray) -> List[Dict]:
        """Detect vehicles from foreground mask"""
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        vehicles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.config['min_area'] < area < self.config['max_area']:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Filter by aspect ratio
                if 0.2 < aspect_ratio < 5.0:
                    vehicles.append({
                        'bbox': (x, y, w, h),
                        'area': area,
                        'centroid': (x + w//2, y + h//2),
                        'confidence': 0.7
                    })
        
        return vehicles
    
    def _track_and_estimate_speeds(self, vehicles: List[Dict]) -> List[float]:
        """Track vehicles and estimate speeds"""
        speeds = []
        matched_tracks = set()
        
        for vehicle in vehicles:
            best_match = None
            min_distance = float('inf')
            
            for track_id, track in self.vehicle_tracks.items():
                if track_id in matched_tracks:
                    continue
                
                last_pos = track['positions'][-1]
                dist = np.sqrt((vehicle['centroid'][0] - last_pos[0])**2 +
                              (vehicle['centroid'][1] - last_pos[1])**2)
                
                if dist < min_distance and dist < self.detection_config['match_distance']:
                    min_distance = dist
                    best_match = track_id
            
            if best_match is not None:
                self.vehicle_tracks[best_match]['positions'].append(
                    vehicle['centroid'])
                matched_tracks.add(best_match)
                
                # Calculate speed
                positions = self.vehicle_tracks[best_match]['positions']
                if len(positions) >= 5:  # Need enough points
                    speed = self._calculate_speed(positions[-5:])
                    if self.detection_config['min_speed'] < speed < self.detection_config['max_speed']:
                        speeds.append(speed)
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
            if current_time - track['start_time'] < self.detection_config['track_timeout']
        }
        
        return speeds
    
    def _calculate_speed(self, positions: List[Tuple]) -> float:
        """Calculate speed from position history"""
        if len(positions) < 2:
            return 0
        
        # Calculate total displacement
        total_dist = 0
        for i in range(len(positions) - 1):
            p1, p2 = positions[i], positions[i+1]
            total_dist += np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
        # Pixels per second
        time_interval = (len(positions) - 1) / self.fps
        if time_interval > 0:
            pps = total_dist / time_interval
            return self._pixels_to_mph(pps)
        return 0
    
    def _pixels_to_mph(self, pixels_per_second: float) -> float:
        """Convert pixels/second to mph"""
        meters_per_second = pixels_per_second / self.calibration['pixels_per_meter']
        mph = meters_per_second * 2.23694
        return mph
    
    def _detect_congestion(self, fg_mask: np.ndarray, 
                          vehicles: List[Dict]) -> bool:
        """Detect traffic congestion"""
        # Simple congestion: high foreground area + many vehicles
        fg_ratio = np.sum(fg_mask > 0) / (fg_mask.shape[0] * fg_mask.shape[1])
        
        if (fg_ratio > 0.25 and 
            len(vehicles) >= self.detection_config['congestion_vehicle_threshold']):
            self.congestion_frames += 1
        else:
            self.congestion_frames = max(0, self.congestion_frames - 1)
        
        threshold_frames = self.detection_config['congestion_time_threshold'] * self.fps
        return self.congestion_frames > threshold_frames