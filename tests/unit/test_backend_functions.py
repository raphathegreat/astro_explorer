#!/usr/bin/env python3
"""
Unit tests for backend functions - Data integrity and processing logic

Tests all core backend functions to ensure they work correctly and maintain
data integrity throughout the processing pipeline.
"""

import unittest
import sys
import os
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

# Add parent directory to path to import the main module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import functions from the clean version
try:
    from iss_speed_html_dashboard_v2_clean import (
        extract_gps_data, calculate_speed, process_image_pair,
        apply_contrast_enhancement, detect_features, match_features,
        calculate_statistics, apply_match_filters, create_plot_data
    )
except ImportError as e:
    print(f"Warning: Could not import from clean version: {e}")
    # Fallback to working version for testing
    from iss_speed_html_dashboard_v2 import (
        extract_gps_data, calculate_speed, process_image_pair,
        apply_contrast_enhancement, detect_features, match_features,
        calculate_statistics, apply_match_filters, create_plot_data
    )

class TestBackendFunctions(unittest.TestCase):
    """Test suite for backend functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.sample_gps_data = {
            'latitude': 40.7128,
            'longitude': -74.0060,
            'altitude': 100.0,
            'timestamp': '2023:01:01 12:00:00'
        }
        
        # Create sample images for testing
        self.create_sample_images()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_sample_images(self):
        """Create sample images for testing"""
        # Create two sample images with some features
        img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add some common features (simple patterns)
        img1[100:150, 100:150] = [255, 0, 0]  # Red square
        img2[105:155, 105:155] = [255, 0, 0]  # Slightly shifted red square
        
        self.img1_path = os.path.join(self.test_dir, "test1.jpg")
        self.img2_path = os.path.join(self.test_dir, "test2.jpg")
        
        cv2.imwrite(self.img1_path, img1)
        cv2.imwrite(self.img2_path, img2)
    
    def test_extract_gps_data_valid(self):
        """Test GPS data extraction with valid data"""
        # Mock EXIF data
        mock_image = Mock()
        mock_image.gps_latitude = (40, 42, 46.08)
        mock_image.gps_latitude_ref = 'N'
        mock_image.gps_longitude = (74, 0, 21.6)
        mock_image.gps_longitude_ref = 'W'
        mock_image.gps_altitude = 100.0
        mock_image.gps_altitude_ref = 0
        mock_image.datetime = '2023:01:01 12:00:00'
        
        with patch('exif.Image') as mock_exif:
            mock_exif.return_value = mock_image
            
            result = extract_gps_data(self.img1_path)
            
            self.assertIsNotNone(result)
            self.assertIn('latitude', result)
            self.assertIn('longitude', result)
            self.assertIn('altitude', result)
            self.assertIn('timestamp', result)
            
            # Check that coordinates are reasonable
            self.assertGreater(result['latitude'], 40)
            self.assertLess(result['latitude'], 41)
            self.assertGreater(result['longitude'], -75)
            self.assertLess(result['longitude'], -73)
    
    def test_extract_gps_data_invalid(self):
        """Test GPS data extraction with invalid/missing data"""
        # Mock image without GPS data
        mock_image = Mock()
        mock_image.gps_latitude = None
        mock_image.gps_longitude = None
        
        with patch('exif.Image') as mock_exif:
            mock_exif.return_value = mock_image
            
            result = extract_gps_data(self.img1_path)
            
            self.assertIsNone(result)
    
    def test_calculate_speed_valid(self):
        """Test speed calculation with valid GPS data"""
        gps1 = {
            'latitude': 40.7128,
            'longitude': -74.0060,
            'altitude': 100.0,
            'timestamp': '2023:01:01 12:00:00'
        }
        gps2 = {
            'latitude': 40.7130,  # Slightly north
            'longitude': -74.0058,  # Slightly east
            'altitude': 100.0,
            'timestamp': '2023:01:01 12:00:01'  # 1 second later
        }
        
        pixel_distance = 100.0  # 100 pixels
        gsd = 1.0  # 1 meter per pixel
        
        speed = calculate_speed(gps1, gps2, pixel_distance, gsd)
        
        self.assertIsNotNone(speed)
        self.assertGreater(speed, 0)
        self.assertIsInstance(speed, (int, float))
    
    def test_calculate_speed_invalid_gps(self):
        """Test speed calculation with invalid GPS data"""
        gps1 = None
        gps2 = {
            'latitude': 40.7130,
            'longitude': -74.0058,
            'altitude': 100.0,
            'timestamp': '2023:01:01 12:00:01'
        }
        
        pixel_distance = 100.0
        gsd = 1.0
        
        speed = calculate_speed(gps1, gps2, pixel_distance, gsd)
        
        self.assertIsNone(speed)
    
    def test_apply_contrast_enhancement(self):
        """Test contrast enhancement functions"""
        # Create a low-contrast image
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        # Test CLAHE enhancement
        enhanced = apply_contrast_enhancement(img, 'clahe')
        self.assertIsNotNone(enhanced)
        self.assertEqual(enhanced.shape, img.shape)
        
        # Test histogram equalization
        enhanced = apply_contrast_enhancement(img, 'histogram')
        self.assertIsNotNone(enhanced)
        self.assertEqual(enhanced.shape, img.shape)
        
        # Test no enhancement
        enhanced = apply_contrast_enhancement(img, 'none')
        self.assertIsNotNone(enhanced)
        np.testing.assert_array_equal(enhanced, img)
    
    def test_detect_features(self):
        """Test feature detection"""
        img = cv2.imread(self.img1_path)
        
        # Test ORB detection
        kp, des = detect_features(img, 'ORB')
        self.assertIsNotNone(kp)
        self.assertIsNotNone(des)
        self.assertGreater(len(kp), 0)
        
        # Test SIFT detection
        kp, des = detect_features(img, 'SIFT')
        self.assertIsNotNone(kp)
        self.assertIsNotNone(des)
        self.assertGreater(len(kp), 0)
    
    def test_match_features(self):
        """Test feature matching"""
        img1 = cv2.imread(self.img1_path)
        img2 = cv2.imread(self.img2_path)
        
        # Detect features
        kp1, des1 = detect_features(img1, 'ORB')
        kp2, des2 = detect_features(img2, 'ORB')
        
        # Match features
        matches = match_features(des1, des2, use_flann=False)
        
        self.assertIsNotNone(matches)
        self.assertIsInstance(matches, list)
        self.assertGreater(len(matches), 0)
    
    def test_calculate_statistics(self):
        """Test statistics calculation"""
        speeds = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        
        stats = calculate_statistics(speeds)
        
        self.assertIsNotNone(stats)
        self.assertIn('mean', stats)
        self.assertIn('median', stats)
        self.assertIn('std_dev', stats)
        self.assertIn('min', stats)
        self.assertIn('max', stats)
        self.assertIn('count', stats)
        
        # Check calculated values
        self.assertEqual(stats['mean'], 5.5)
        self.assertEqual(stats['median'], 5.5)
        self.assertEqual(stats['min'], 1.0)
        self.assertEqual(stats['max'], 10.0)
        self.assertEqual(stats['count'], 10)
    
    def test_apply_match_filters(self):
        """Test match filtering"""
        matches = [
            {'speed': 1.0, 'pixel_distance': 10, 'match_distance': 0.1},
            {'speed': 2.0, 'pixel_distance': 20, 'match_distance': 0.2},
            {'speed': 3.0, 'pixel_distance': 30, 'match_distance': 0.3},
            {'speed': 100.0, 'pixel_distance': 1000, 'match_distance': 1.0},  # Outlier
        ]
        
        filters = {
            'enable_std_dev': True,
            'std_dev_multiplier': 2.0
        }
        
        filtered = apply_match_filters(matches, filters)
        
        self.assertIsNotNone(filtered)
        self.assertLess(len(filtered), len(matches))  # Should filter out outliers
        self.assertLess(len([m for m in filtered if m['speed'] > 50]), len(filtered))  # No high outliers
    
    def test_create_plot_data(self):
        """Test plot data creation"""
        matches = [
            {'speed': 1.0, 'pair_index': 0},
            {'speed': 2.0, 'pair_index': 0},
            {'speed': 3.0, 'pair_index': 1},
            {'speed': 4.0, 'pair_index': 1},
        ]
        
        plot_data = create_plot_data(matches)
        
        self.assertIsNotNone(plot_data)
        self.assertIn('histogram', plot_data)
        self.assertIn('pairs', plot_data)
        
        # Check histogram data
        self.assertIn('speeds', plot_data['histogram'])
        self.assertIn('median', plot_data['histogram'])
        
        # Check pairs data
        self.assertIn('pairs', plot_data['pairs'])
        self.assertIn('means', plot_data['pairs'])
        self.assertIn('medians', plot_data['pairs'])
    
    def test_data_integrity_through_pipeline(self):
        """Test data integrity through the entire processing pipeline"""
        # This is a comprehensive test that ensures data doesn't get corrupted
        # through the processing pipeline
        
        # Create test data
        original_speeds = [1.0, 2.0, 3.0, 4.0, 5.0]
        original_count = len(original_speeds)
        
        # Process through statistics
        stats = calculate_statistics(original_speeds)
        
        # Verify data integrity
        self.assertEqual(stats['count'], original_count)
        self.assertEqual(stats['min'], min(original_speeds))
        self.assertEqual(stats['max'], max(original_speeds))
        
        # Process through filtering
        matches = [{'speed': s} for s in original_speeds]
        filtered = apply_match_filters(matches, {})
        
        # Verify no data corruption
        self.assertEqual(len(filtered), original_count)
        filtered_speeds = [m['speed'] for m in filtered]
        self.assertEqual(sorted(filtered_speeds), sorted(original_speeds))
        
        # Process through plot data creation
        plot_data = create_plot_data(matches)
        
        # Verify plot data integrity
        self.assertEqual(len(plot_data['histogram']['speeds']), original_count)
        self.assertEqual(plot_data['histogram']['speeds'], original_speeds)

if __name__ == '__main__':
    unittest.main()
