"""
Utility Functions
Configuration loading and directory management
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """Load and manage configuration files"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load YAML configuration file
        
        Args:
            config_path: Path to YAML config file
            
        Returns:
            Dictionary containing configuration
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required sections
        required_sections = ['system', 'edge', 'cloud', 'detection']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
        
        return config
    
    @staticmethod
    def create_directories():
        """Create required project directories"""
        directories = [
            'data',
            'data/input_videos',
            'data/results',
            'data/results/edge',
            'data/results/cloud',
            'data/results/hybrid',
            'data/results/asymmetric',
            'models',
            'config'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        print("✅ Project directories created/verified")
    
    @staticmethod
    def save_config(config: Dict[str, Any], output_path: str):
        """
        Save configuration to YAML file
        
        Args:
            config: Configuration dictionary
            output_path: Path to save config file
        """
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"✅ Config saved to: {output_path}")
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        Get default configuration
        
        Returns:
            Default configuration dictionary
        """
        return {
            'system': {
                'network_threshold': 300,
                'frame_buffer_size': 10
            },
            'edge': {
                'resolution': [320, 180],
                'gmm_history': 500,
                'gmm_var_threshold': 16,
                'color_space': 'gray',
                'learning_rate': -1,
                'morphological_ops': True,
                'min_area': 150,
                'max_area': 50000
            },
            'cloud': {
                'resolution': [1280, 720],
                'model_path': 'models/yolov8n.pt',
                'confidence_threshold': 0.5,
                'iou_threshold': 0.45,
                'gmm_history': 500,
                'gmm_var_threshold': 16,
                'gfm_components': 5
            },
            'detection': {
                'match_distance': 100,
                'min_speed': 1,
                'max_speed': 150,
                'track_timeout': 5,
                'congestion_threshold': 10
            }
        }


class FileManager:
    """Manage file operations"""
    
    @staticmethod
    def get_video_files(folder_path: str, extensions: list = None) -> list:
        """
        Get all video files from folder
        
        Args:
            folder_path: Path to folder
            extensions: List of video extensions (default: mp4, avi, mov, mkv)
            
        Returns:
            List of video file paths
        """
        if extensions is None:
            extensions = ['.mp4', '.avi', '.mov', '.mkv']
        
        folder = Path(folder_path)
        video_files = []
        
        for ext in extensions:
            video_files.extend(folder.glob(f'*{ext}'))
            video_files.extend(folder.glob(f'*{ext.upper()}'))
        
        return sorted(video_files)
    
    @staticmethod
    def ensure_path(filepath: str):
        """Ensure directory for filepath exists"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def get_output_filename(input_path: str, suffix: str = '', extension: str = None) -> str:
        """
        Generate output filename from input path
        
        Args:
            input_path: Input file path
            suffix: Suffix to add to filename
            extension: New extension (if changing)
            
        Returns:
            Output filename
        """
        path = Path(input_path)
        name = path.stem + suffix
        ext = extension if extension else path.suffix
        return name + ext


class Logger:
    """Simple logging utility"""
    
    def __init__(self, log_file: str = None):
        self.log_file = log_file
        self.logs = []
    
    def log(self, message: str, level: str = 'INFO'):
        """Log a message"""
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        print(log_entry)
        self.logs.append(log_entry)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(log_entry + '\n')
    
    def info(self, message: str):
        """Log info message"""
        self.log(message, 'INFO')
    
    def warning(self, message: str):
        """Log warning message"""
        self.log(message, 'WARNING')
    
    def error(self, message: str):
        """Log error message"""
        self.log(message, 'ERROR')
    
    def save_logs(self, output_path: str):
        """Save all logs to file"""
        with open(output_path, 'w') as f:
            f.write('\n'.join(self.logs))
        print(f"✅ Logs saved to: {output_path}")