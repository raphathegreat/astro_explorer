#!/usr/bin/env python3
"""
Unit tests for data processing functions - Image processing and feature matching

Tests the core image processing pipeline to ensure accurate feature detection,
matching, and speed calculations.
"""

import unittest
import sys
import os
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from iss_speed_html_dashboard_v2_clean import (
        process_image_pair, extract_gps_data, calculate_speed,
        apply_ransac_filtering, calculate_gsd
    )
except ImportError:
    from iss_speed_html_dashboard_v2 import (
        process_image_pair, extract_gps_data, calculate_speed,
        apply_ransac_filtering, calculate_gsd
    )

class TestDataProcessing(unittest.TestCase):
    """Test suite for data processing functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.create_test_images()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_test_images(self):
        """Create test images with known features"""
        # Create two images with a clear feature that moves
        img1 = np.zeros((480, 640, 3), dtype=np.uint8)
        img2 = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add a white square in img1
        img1[200:250, 200:250] = [255, 255, 255]
        
        # Add the same square shifted by 10 pixels in img2
        img2[200:250, 210:260] = [255, 255, 255]
        
        # Add some noise to make it more realistic
        noise1 = np.random.randint(0, 50, img1.shape, dtype=np.uint8)
        noise2 = np.random.randint(0, 50, img2.shape, dtype=np.uint8)
        
        img1 = cv2.add(img1, noise1)
        img2 = cv2.add(img2, noise2)
        
        self.img1_path = os.path.join(self.test_dir, "test1.jpg")
        self.img2_path = os.path.join(self.test_dir, "test2.jpg")
        
        cv2.imwrite(self.img1_path, img1)
        cv2.imwrite(self.img2_path, img2)
    
    def test_process_image_pair_basic(self):
        """Test basic image pair processing"""
        gps1 = {
            'latitude': 40.7128,
            'longitude': -74.0060,
            'altitude': 100.0,
            'timestamp': '2023:01:01 12:00:00'
        }
        gps2 = {
            'latitude': 40.7129,
            'longitude': -74.0059,
            'altitude': 100.0,
            'timestamp': '2023:01:01 12:00:01'
        }
        
        with patch('iss_speed_html_dashboard_v2_clean.extract_gps_data') as mock_gps:
            mock_gps.side_effect = [gps1, gps2]
            
            result = process_image_pair(
                self.img1_path, self.img2_path, 0,
                algorithm='ORB', use_flann=False, use_ransac=False
            )
            
            self.assertIsNotNone(result)
            self.assertIsInstance(result, list)
            self.assertGreater(len(result), 0)
            
            # Check that each match has required fields
            for match in result:
                self.assertIn('speed', match)
                self.assertIn('pixel_distance', match)
                self.assertIn('match_distance', match)
                self.assertIn('pt1', match)
                self.assertIn('pt2', match)
                self.assertIn('time_difference', match)
    
    def test_process_image_pair_with_ransac(self):
        """Test image pair processing with RANSAC filtering"""
        gps1 = {
            'latitude': 40.7128,
            'longitude': -74.0060,
            'altitude': 100.0,
            'timestamp': '2023:01:01 12:00:00'
        }
        gps2 = {
            'latitude': 40.7129,
            'longitude': -74.0059,
            'altitude': 100.0,
            'timestamp': '2023:01:01 12:00:01'
        }
        
        with patch('iss_speed_html_dashboard_v2_clean.extract_gps_data') as mock_gps:
            mock_gps.side_effect = [gps1, gps2]
            
            result = process_image_pair(
                self.img1_path, self.img2_path, 0,
                algorithm='ORB', use_flann=False, use_ransac=True,
                ransac_threshold=5.0, ransac_min_matches=4
            )
            
            self.assertIsNotNone(result)
            self.assertIsInstance(result, list)
    
    def test_process_image_pair_sift(self):
        """Test image pair processing with SIFT algorithm"""
        gps1 = {
            'latitude': 40.7128,
            'longitude': -74.0060,
            'altitude': 100.0,
            'timestamp': '2023:01:01 12:00:00'
        }
        gps2 = {
            'latitude': 40.7129,
            'longitude': -74.0059,
            'altitude': 100.0,
            'timestamp': '2023:01:01 12:00:01'
        }
        
        with patch('iss_speed_html_dashboard_v2_clean.extract_gps_data') as mock_gps:
            mock_gps.side_effect = [gps1, gps2]
            
            result = process_image_pair(
                self.img1_path, self.img2_path, 0,
                algorithm='SIFT', use_flann=True, use_ransac=False
            )
            
            self.assertIsNotNone(result)
            self.assertIsInstance(result, list)
    
    def test_process_image_pair_no_gps(self):
        """Test image pair processing without GPS data"""
        with patch('iss_speed_html_dashboard_v2_clean.extract_gps_data') as mock_gps:
            mock_gps.return_value = None
            
            result = process_image_pair(
                self.img1_path, self.img2_path, 0,
                algorithm='ORB', use_flann=False, use_ransac=False
            )
            
            # Should still return matches but without speed calculations
            self.assertIsNotNone(result)
            self.assertIsInstance(result, list)
            
            for match in result:
                self.assertIsNone(match['speed'])
    
    def test_calculate_gsd(self):
        """Test Ground Sample Distance calculation"""
        # Test with known parameters
        altitude = 100.0  # meters
        focal_length = 50.0  # mm
        sensor_width = 36.0  # mm
        image_width = 640  # pixels
        
        gsd = calculate_gsd(altitude, focal_length, sensor_width, image_width)
        
        self.assertIsNotNone(gsd)
        self.assertGreater(gsd, 0)
        self.assertIsInstance(gsd, (int, float))
        
        # GSD should be proportional to altitude
        gsd2 = calculate_gsd(altitude * 2, focal_length, sensor_width, image_width)
        self.assertAlmostEqual(gsd2, gsd * 2, places=2)
    
    def test_apply_ransac_filtering(self):
        """Test RANSAC filtering of matches"""
        # Create sample keypoints and matches
        kp1 = [cv2.KeyPoint(100, 100, 10) for _ in range(10)]
        kp2 = [cv2.KeyPoint(110, 110, 10) for _ in range(10)]
        
        matches = [cv2.DMatch(i, i, 0, 0.1) for i in range(10)]
        
        # Add some outliers
        matches.append(cv2.DMatch(0, 9, 0, 0.1))  # Outlier
        
        filtered_matches = apply_ransac_filtering(kp1, kp2, matches, 5.0, 4)
        
        self.assertIsNotNone(filtered_matches)
        self.assertLessEqual(len(filtered_matches), len(matches))
        self.assertGreater(len(filtered_matches), 0)
    
    def test_speed_calculation_accuracy(self):
        """Test speed calculation accuracy with known values"""
        # Create GPS data for a known movement
        gps1 = {
            'latitude': 40.7128,
            'longitude': -74.0060,
            'altitude': 100.0,
            'timestamp': '2023:01:01 12:00:00'
        }
        gps2 = {
            'latitude': 40.7128 + 0.0001,  # Small movement north
            'longitude': -74.0060,
            'altitude': 100.0,
            'timestamp': '2023:01:01 12:00:01'  # 1 second later
        }
        
        pixel_distance = 10.0  # 10 pixels
        gsd = 1.0  # 1 meter per pixel
        
        speed = calculate_speed(gps1, gps2, pixel_distance, gsd)
        
        self.assertIsNotNone(speed)
        self.assertGreater(speed, 0)
        
        # Speed should be reasonable (not too high or too low)
        self.assertLess(speed, 1000)  # Less than 1000 m/s
        self.assertGreater(speed, 0.1)  # Greater than 0.1 m/s
    
    def test_contrast_enhancement_effects(self):
        """Test that contrast enhancement improves feature detection"""
        # Create a low-contrast image
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        # Add a subtle feature
        img[40:60, 40:60] = [140, 140, 140]
        
        # Test without enhancement
        kp1, des1 = detect_features(img, 'ORB')
        
        # Apply contrast enhancement
        enhanced = apply_contrast_enhancement(img, 'clahe')
        kp2, des2 = detect_features(enhanced, 'ORB')
        
        # Enhanced image should detect more features
        self.assertGreaterEqual(len(kp2), len(kp1))
    
    def test_feature_matching_consistency(self):
        """Test that feature matching is consistent"""
        img1 = cv2.imread(self.img1_path)
        img2 = cv2.imread(self.img2_path)
        
        # Run matching multiple times
        results = []
        for _ in range(3):
            kp1, des1 = detect_features(img1, 'ORB')
            kp2, des2 = detect_features(img2, 'ORB')
            matches = match_features(des1, des2, use_flann=False)
            results.append(len(matches))
        
        # Results should be consistent (within reasonable variance)
        self.assertLess(max(results) - min(results), max(results) * 0.2)
    
    def test_data_types_and_formats(self):
        """Test that all data types and formats are correct"""
        gps1 = {
            'latitude': 40.7128,
            'longitude': -74.0060,
            'altitude': 100.0,
            'timestamp': '2023:01:01 12:00:00'
        }
        gps2 = {
            'latitude': 40.7129,
            'longitude': -74.0059,
            'altitude': 100.0,
            'timestamp': '2023:01:01 12:00:01'
        }
        
        with patch('iss_speed_html_dashboard_v2_clean.extract_gps_data') as mock_gps:
            mock_gps.side_effect = [gps1, gps2]
            
            result = process_image_pair(
                self.img1_path, self.img2_path, 0,
                algorithm='ORB', use_flann=False, use_ransac=False
            )
            
            for match in result:
                # Check data types
                self.assertIsInstance(match['speed'], (int, float, type(None)))
                self.assertIsInstance(match['pixel_distance'], (int, float))
                self.assertIsInstance(match['match_distance'], (int, float))
                self.assertIsInstance(match['pt1'], (list, tuple))
                self.assertIsInstance(match['pt2'], (list, tuple))
                self.assertIsInstance(match['time_difference'], (int, float))
                
                # Check that points are 2D coordinates
                self.assertEqual(len(match['pt1']), 2)
                self.assertEqual(len(match['pt2']), 2)
                
                # Check that coordinates are reasonable
                self.assertGreaterEqual(match['pt1'][0], 0)
                self.assertGreaterEqual(match['pt1'][1], 0)
                self.assertGreaterEqual(match['pt2'][0], 0)
                self.assertGreaterEqual(match['pt2'][1], 0)

if __name__ == '__main__':
    unittest.main()
