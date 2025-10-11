#!/usr/bin/env python3
"""
Sample test data and fixtures for comprehensive testing

Provides consistent test data across all test suites to ensure
reproducible and reliable testing.
"""

import numpy as np
import cv2
import os
import tempfile
import shutil
from datetime import datetime

class TestDataGenerator:
    """Generate consistent test data for all tests"""
    
    @staticmethod
    def create_sample_images(output_dir, count=5):
        """Create sample images with known features"""
        images = []
        
        for i in range(count):
            # Create base image
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Add a distinctive feature that moves between images
            feature_size = 50
            feature_x = 200 + i * 10  # Moves 10 pixels per image
            feature_y = 200 + i * 5   # Moves 5 pixels per image
            
            img[feature_y:feature_y+feature_size, feature_x:feature_x+feature_size] = [255, 0, 0]
            
            # Add some noise
            noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
            img = cv2.add(img, noise)
            
            # Save image
            img_path = os.path.join(output_dir, f"ISS_{i:03d}.jpg")
            cv2.imwrite(img_path, img)
            images.append(img_path)
        
        return images
    
    @staticmethod
    def get_sample_gps_data():
        """Get sample GPS data for testing"""
        return [
            {
                'latitude': 40.7128 + i * 0.0001,
                'longitude': -74.0060 + i * 0.0001,
                'altitude': 100.0 + i * 10,
                'timestamp': f'2023:01:01 12:00:{i:02d}'
            }
            for i in range(5)
        ]
    
    @staticmethod
    def get_sample_matches():
        """Get sample match data for testing"""
        return [
            {
                'speed': 100.0 + i * 10,
                'pixel_distance': 10.0 + i,
                'match_distance': 0.1 + i * 0.01,
                'pt1': [100 + i, 100 + i],
                'pt2': [110 + i, 110 + i],
                'time_difference': 1.0,
                'image1_properties': {
                    'latitude': 40.7128,
                    'longitude': -74.0060,
                    'altitude': 100.0,
                    'timestamp': '2023:01:01 12:00:00'
                },
                'image2_properties': {
                    'latitude': 40.7129,
                    'longitude': -74.0059,
                    'altitude': 100.0,
                    'timestamp': '2023:01:01 12:00:01'
                },
                'cloudiness': 'clear' if i < 3 else 'cloudy',
                'ml_classification': 'good' if i < 3 else 'poor',
                'ml_confidence': 0.9 if i < 3 else 0.3,
                'pair_index': i // 2
            }
            for i in range(10)
        ]
    
    @staticmethod
    def get_sample_statistics():
        """Get sample statistics for testing"""
        speeds = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0]
        
        return {
            'mean': np.mean(speeds),
            'median': np.median(speeds),
            'std_dev': np.std(speeds),
            'min': min(speeds),
            'max': max(speeds),
            'count': len(speeds),
            'range': max(speeds) - min(speeds),
            'mode': 100.0  # Most common value
        }
    
    @staticmethod
    def get_sample_plot_data():
        """Get sample plot data for testing"""
        speeds = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0]
        
        return {
            'histogram': {
                'speeds': speeds,
                'median': np.median(speeds),
                'mean': np.mean(speeds)
            },
            'pairs': {
                'pairs': [1, 2, 3, 4, 5],
                'means': [105.0, 125.0, 145.0, 165.0, 185.0],
                'medians': [105.0, 125.0, 145.0, 165.0, 185.0],
                'stds': [5.0, 5.0, 5.0, 5.0, 5.0],
                'colors': ['#007bff', '#28a745', '#ffc107', '#dc3545', '#6f42c1']
            }
        }

class TestEnvironment:
    """Manage test environment setup and cleanup"""
    
    def __init__(self):
        self.test_dir = None
        self.photos_dir = None
        self.cache_dir = None
    
    def setup(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp(prefix="iss_test_")
        self.photos_dir = os.path.join(self.test_dir, "photos-test")
        self.cache_dir = os.path.join(self.test_dir, "cache")
        
        os.makedirs(self.photos_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create sample images
        TestDataGenerator.create_sample_images(self.photos_dir, 10)
        
        return self.test_dir
    
    def cleanup(self):
        """Clean up test environment"""
        if self.test_dir and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def __enter__(self):
        return self.setup()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

# Test data constants
SAMPLE_SPEEDS = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0]
SAMPLE_GPS_COORDINATES = [
    (40.7128, -74.0060),
    (40.7129, -74.0059),
    (40.7130, -74.0058),
    (40.7131, -74.0057),
    (40.7132, -74.0056)
]

# Expected results for validation
EXPECTED_STATISTICS = {
    'mean': 145.0,
    'median': 145.0,
    'min': 100.0,
    'max': 190.0,
    'count': 10,
    'range': 90.0
}

# Test configuration
TEST_CONFIG = {
    'algorithms': ['ORB', 'SIFT'],
    'image_count': 10,
    'match_count_per_pair': 5,
    'speed_range': (50.0, 500.0),
    'pixel_distance_range': (5.0, 100.0),
    'time_difference_range': (0.5, 5.0)
}
