import time
import random
import numpy as np
from typing import Optional

class NetworkSimulator:
    """Simulates network conditions between edge and cloud"""
    
    def __init__(self, threshold_kbps: int = 300):
        self.threshold_kbps = threshold_kbps
        self.current_bandwidth = 1000
        self.frame_drop_probability = 0.0
        
    def set_condition(self, bandwidth_kbps: int):
        """Manually set network bandwidth"""
        self.current_bandwidth = bandwidth_kbps
        self._update_frame_drop_probability()
        
    def _update_frame_drop_probability(self):
        """Calculate frame drop probability based on bandwidth"""
        if self.current_bandwidth >= self.threshold_kbps:
            self.frame_drop_probability = 0.0
        else:
            ratio = self.current_bandwidth / self.threshold_kbps
            self.frame_drop_probability = max(0, 1.0 - ratio)
    
    def should_use_cloud(self) -> bool:
        """Determine if cloud processing should be used"""
        return self.current_bandwidth >= self.threshold_kbps
    
    def simulate_frame_transfer(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Simulate frame transfer with possible packet loss"""
        if random.random() < self.frame_drop_probability:
            return None
        return frame
    
    def get_video_quality_factor(self) -> float:
        """Get quality degradation factor (0-1)"""
        return min(1.0, self.current_bandwidth / self.threshold_kbps)
