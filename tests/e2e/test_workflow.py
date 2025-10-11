#!/usr/bin/env python3
"""
End-to-end workflow tests - Complete data processing pipeline

Tests the complete workflow from image loading to final results display,
ensuring data integrity throughout the entire process.
"""

import unittest
import sys
import os
import json
import tempfile
import shutil
import time
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from iss_speed_html_dashboard_v2_clean import app
except ImportError:
    from iss_speed_html_dashboard_v2 import app

class TestEndToEndWorkflow(unittest.TestCase):
    """Test suite for end-to-end workflows"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.app.testing = True
        
        self.test_dir = tempfile.mkdtemp()
        self.create_test_environment()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_test_environment(self):
        """Create a complete test environment with photos and cache"""
        # Create photos folder
        photos_dir = os.path.join(self.test_dir, "photos-workflow-test")
        os.makedirs(photos_dir, exist_ok=True)
        
        # Create cache directory
        cache_dir = os.path.join(self.test_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create sample images (we'll mock the actual image processing)
        for i in range(10):
            dummy_file = os.path.join(photos_dir, f"ISS_{i:03d}.jpg")
            with open(dummy_file, 'w') as f:
                f.write("dummy jpg content")
    
    def test_complete_workflow_basic(self):
        """Test complete workflow: load -> process -> filter -> display"""
        
        # Step 1: Get available folders
        with patch('os.listdir') as mock_listdir:
            mock_listdir.side_effect = [
                ['photos-workflow-test', 'other-folder'],  # Root directory
                ['ISS_001.jpg', 'ISS_002.jpg', 'ISS_003.jpg']  # Photos directory
            ]
            
            with patch('os.path.isdir') as mock_isdir:
                mock_isdir.side_effect = lambda x: x == 'photos-workflow-test'
                
                response = self.app.get('/api/folders')
                self.assertEqual(response.status_code, 200)
                folders = json.loads(response.data)
                self.assertGreater(len(folders), 0)
        
        # Step 2: Process image range
        with patch('iss_speed_html_dashboard_v2_clean.process_image_pair') as mock_process:
            # Mock successful image processing
            mock_process.return_value = [
                {
                    'speed': 100.0 + i * 10,  # Varying speeds
                    'pixel_distance': 10.0 + i,
                    'match_distance': 0.1 + i * 0.01,
                    'pt1': [100 + i, 100 + i],
                    'pt2': [110 + i, 110 + i],
                    'time_difference': 1.0,
                    'image1_properties': {'latitude': 40.7128, 'longitude': -74.0060},
                    'image2_properties': {'latitude': 40.7129, 'longitude': -74.0059},
                    'cloudiness': 'clear',
                    'ml_classification': None
                }
                for i in range(5)  # 5 matches per pair
            ]
            
            with patch('iss_speed_html_dashboard_v2_clean.extract_gps_data') as mock_gps:
                mock_gps.return_value = {
                    'latitude': 40.7128,
                    'longitude': -74.0060,
                    'altitude': 100.0,
                    'timestamp': '2023:01:01 12:00:00'
                }
                
                response = self.app.post('/api/process-range',
                    data=json.dumps({
                        'folder': 'photos-workflow-test',
                        'start_idx': 0,
                        'end_idx': 2,
                        'algorithm': 'ORB',
                        'use_flann': False,
                        'use_ransac': False
                    }),
                    content_type='application/json'
                )
                
                self.assertEqual(response.status_code, 200)
                result = json.loads(response.data)
                self.assertEqual(result['status'], 'success')
        
        # Step 3: Get statistics
        response = self.app.get('/api/statistics')
        self.assertEqual(response.status_code, 200)
        stats = json.loads(response.data)
        
        # Verify statistics are reasonable
        self.assertGreater(stats['count'], 0)
        self.assertGreater(stats['mean'], 0)
        self.assertGreater(stats['max'], stats['min'])
        
        # Step 4: Get plot data
        response = self.app.get('/api/plot-data')
        self.assertEqual(response.status_code, 200)
        plot_data = json.loads(response.data)
        
        # Verify plot data structure
        self.assertIn('histogram', plot_data)
        self.assertIn('pairs', plot_data)
        self.assertIn('speeds', plot_data['histogram'])
        self.assertIn('means', plot_data['pairs'])
        
        # Step 5: Apply filters
        response = self.app.post('/api/apply-filters',
            data=json.dumps({
                'enable_std_dev': True,
                'std_dev_multiplier': 2.0
            }),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        filter_result = json.loads(response.data)
        self.assertEqual(filter_result['status'], 'success')
        
        # Step 6: Verify filtered statistics
        response = self.app.get('/api/statistics')
        self.assertEqual(response.status_code, 200)
        filtered_stats = json.loads(response.data)
        
        # Filtered data should have same or fewer matches
        self.assertLessEqual(filtered_stats['count'], stats['count'])
    
    def test_workflow_with_different_algorithms(self):
        """Test workflow with different algorithms"""
        algorithms = ['ORB', 'SIFT']
        
        for algorithm in algorithms:
            with self.subTest(algorithm=algorithm):
                with patch('iss_speed_html_dashboard_v2_clean.process_image_pair') as mock_process:
                    mock_process.return_value = [
                        {
                            'speed': 100.0,
                            'pixel_distance': 10.0,
                            'match_distance': 0.1,
                            'pt1': [100, 100],
                            'pt2': [110, 110],
                            'time_difference': 1.0,
                            'image1_properties': {},
                            'image2_properties': {},
                            'cloudiness': 'clear',
                            'ml_classification': None
                        }
                    ]
                    
                    with patch('iss_speed_html_dashboard_v2_clean.extract_gps_data') as mock_gps:
                        mock_gps.return_value = {
                            'latitude': 40.7128,
                            'longitude': -74.0060,
                            'altitude': 100.0,
                            'timestamp': '2023:01:01 12:00:00'
                        }
                        
                        response = self.app.post('/api/process-range',
                            data=json.dumps({
                                'folder': 'photos-workflow-test',
                                'start_idx': 0,
                                'end_idx': 1,
                                'algorithm': algorithm,
                                'use_flann': algorithm == 'SIFT',
                                'use_ransac': False
                            }),
                            content_type='application/json'
                        )
                        
                        self.assertEqual(response.status_code, 200)
                        result = json.loads(response.data)
                        self.assertEqual(result['status'], 'success')
    
    def test_workflow_with_ransac_filtering(self):
        """Test workflow with RANSAC filtering enabled"""
        with patch('iss_speed_html_dashboard_v2_clean.process_image_pair') as mock_process:
            mock_process.return_value = [
                {
                    'speed': 100.0,
                    'pixel_distance': 10.0,
                    'match_distance': 0.1,
                    'pt1': [100, 100],
                    'pt2': [110, 110],
                    'time_difference': 1.0,
                    'image1_properties': {},
                    'image2_properties': {},
                    'cloudiness': 'clear',
                    'ml_classification': None
                }
            ]
            
            with patch('iss_speed_html_dashboard_v2_clean.extract_gps_data') as mock_gps:
                mock_gps.return_value = {
                    'latitude': 40.7128,
                    'longitude': -74.0060,
                    'altitude': 100.0,
                    'timestamp': '2023:01:01 12:00:00'
                }
                
                response = self.app.post('/api/process-range',
                    data=json.dumps({
                        'folder': 'photos-workflow-test',
                        'start_idx': 0,
                        'end_idx': 1,
                        'algorithm': 'ORB',
                        'use_flann': False,
                        'use_ransac': True,
                        'ransac_threshold': 5.0,
                        'ransac_min_matches': 4
                    }),
                    content_type='application/json'
                )
                
                self.assertEqual(response.status_code, 200)
                result = json.loads(response.data)
                self.assertEqual(result['status'], 'success')
    
    def test_workflow_with_multiple_filters(self):
        """Test workflow with multiple filters applied"""
        # First, process some data
        with patch('iss_speed_html_dashboard_v2_clean.process_image_pair') as mock_process:
            mock_process.return_value = [
                {
                    'speed': 100.0 + i * 50,  # Wide range of speeds
                    'pixel_distance': 10.0 + i,
                    'match_distance': 0.1 + i * 0.01,
                    'pt1': [100 + i, 100 + i],
                    'pt2': [110 + i, 110 + i],
                    'time_difference': 1.0,
                    'image1_properties': {},
                    'image2_properties': {},
                    'cloudiness': 'clear' if i < 3 else 'cloudy',
                    'ml_classification': 'good' if i < 3 else 'poor'
                }
                for i in range(10)  # 10 matches with varying quality
            ]
            
            with patch('iss_speed_html_dashboard_v2_clean.extract_gps_data') as mock_gps:
                mock_gps.return_value = {
                    'latitude': 40.7128,
                    'longitude': -74.0060,
                    'altitude': 100.0,
                    'timestamp': '2023:01:01 12:00:00'
                }
                
                response = self.app.post('/api/process-range',
                    data=json.dumps({
                        'folder': 'photos-workflow-test',
                        'start_idx': 0,
                        'end_idx': 2,
                        'algorithm': 'ORB',
                        'use_flann': False,
                        'use_ransac': False
                    }),
                    content_type='application/json'
                )
                
                self.assertEqual(response.status_code, 200)
        
        # Get initial statistics
        response = self.app.get('/api/statistics')
        initial_stats = json.loads(response.data)
        
        # Apply multiple filters
        filters = [
            {'enable_std_dev': True, 'std_dev_multiplier': 1.5},
            {'enable_cloudiness': True, 'include_mostly_cloudy': False},
            {'enable_ml_classification': True, 'include_good': True}
        ]
        
        for filter_config in filters:
            response = self.app.post('/api/apply-filters',
                data=json.dumps(filter_config),
                content_type='application/json'
            )
            
            self.assertEqual(response.status_code, 200)
            result = json.loads(response.data)
            self.assertEqual(result['status'], 'success')
        
        # Get final statistics
        response = self.app.get('/api/statistics')
        final_stats = json.loads(response.data)
        
        # Final stats should have fewer matches (filtered out)
        self.assertLess(final_stats['count'], initial_stats['count'])
    
    def test_workflow_cache_management(self):
        """Test workflow with cache management"""
        # Test cache status
        response = self.app.get('/api/cache-status')
        self.assertEqual(response.status_code, 200)
        cache_status = json.loads(response.data)
        
        self.assertIn('total_files', cache_status)
        self.assertIn('total_size', cache_status)
        self.assertIn('data_loaded', cache_status)
        
        # Process some data (this should create cache)
        with patch('iss_speed_html_dashboard_v2_clean.process_image_pair') as mock_process:
            mock_process.return_value = [
                {
                    'speed': 100.0,
                    'pixel_distance': 10.0,
                    'match_distance': 0.1,
                    'pt1': [100, 100],
                    'pt2': [110, 110],
                    'time_difference': 1.0,
                    'image1_properties': {},
                    'image2_properties': {},
                    'cloudiness': 'clear',
                    'ml_classification': None
                }
            ]
            
            with patch('iss_speed_html_dashboard_v2_clean.extract_gps_data') as mock_gps:
                mock_gps.return_value = {
                    'latitude': 40.7128,
                    'longitude': -74.0060,
                    'altitude': 100.0,
                    'timestamp': '2023:01:01 12:00:00'
                }
                
                response = self.app.post('/api/process-range',
                    data=json.dumps({
                        'folder': 'photos-workflow-test',
                        'start_idx': 0,
                        'end_idx': 1,
                        'algorithm': 'ORB',
                        'use_flann': False,
                        'use_ransac': False
                    }),
                    content_type='application/json'
                )
                
                self.assertEqual(response.status_code, 200)
        
        # Clear cache
        response = self.app.post('/api/clear-v2-cache')
        self.assertEqual(response.status_code, 200)
        clear_result = json.loads(response.data)
        self.assertEqual(clear_result['status'], 'success')
        
        # Verify cache is cleared
        response = self.app.get('/api/cache-status')
        cache_status_after = json.loads(response.data)
        self.assertFalse(cache_status_after['data_loaded'])
    
    def test_workflow_algorithm_comparison(self):
        """Test workflow with algorithm comparison"""
        with patch('iss_speed_html_dashboard_v2_clean.run_algorithm_comparison') as mock_comparison:
            # Start algorithm comparison
            response = self.app.post('/api/algorithm-comparison',
                data=json.dumps({
                    'algorithms': ['ORB', 'SIFT'],
                    'folder': 'photos-workflow-test',
                    'start_idx': 0,
                    'end_idx': 2
                }),
                content_type='application/json'
            )
            
            self.assertEqual(response.status_code, 200)
            result = json.loads(response.data)
            self.assertEqual(result['status'], 'started')
            
            # Check processing status
            response = self.app.get('/api/processing-status')
            self.assertEqual(response.status_code, 200)
            status = json.loads(response.data)
            
            self.assertIn('status', status)
            self.assertIn('progress', status)
    
    def test_workflow_data_consistency(self):
        """Test that data remains consistent throughout the workflow"""
        # This test ensures that data doesn't get corrupted or lost
        # during the complete workflow
        
        # Process data
        with patch('iss_speed_html_dashboard_v2_clean.process_image_pair') as mock_process:
            original_matches = [
                {
                    'speed': 100.0 + i * 10,
                    'pixel_distance': 10.0 + i,
                    'match_distance': 0.1 + i * 0.01,
                    'pt1': [100 + i, 100 + i],
                    'pt2': [110 + i, 110 + i],
                    'time_difference': 1.0,
                    'image1_properties': {},
                    'image2_properties': {},
                    'cloudiness': 'clear',
                    'ml_classification': None
                }
                for i in range(5)
            ]
            
            mock_process.return_value = original_matches
            
            with patch('iss_speed_html_dashboard_v2_clean.extract_gps_data') as mock_gps:
                mock_gps.return_value = {
                    'latitude': 40.7128,
                    'longitude': -74.0060,
                    'altitude': 100.0,
                    'timestamp': '2023:01:01 12:00:00'
                }
                
                response = self.app.post('/api/process-range',
                    data=json.dumps({
                        'folder': 'photos-workflow-test',
                        'start_idx': 0,
                        'end_idx': 1,
                        'algorithm': 'ORB',
                        'use_flann': False,
                        'use_ransac': False
                    }),
                    content_type='application/json'
                )
                
                self.assertEqual(response.status_code, 200)
        
        # Get statistics multiple times and verify consistency
        stats_responses = []
        for _ in range(3):
            response = self.app.get('/api/statistics')
            self.assertEqual(response.status_code, 200)
            stats_responses.append(json.loads(response.data))
        
        # All statistics should be identical
        for i in range(1, len(stats_responses)):
            self.assertEqual(stats_responses[0], stats_responses[i])
        
        # Get plot data multiple times and verify consistency
        plot_responses = []
        for _ in range(3):
            response = self.app.get('/api/plot-data')
            self.assertEqual(response.status_code, 200)
            plot_responses.append(json.loads(response.data))
        
        # All plot data should be identical
        for i in range(1, len(plot_responses)):
            self.assertEqual(plot_responses[0], plot_responses[i])
        
        # Verify that statistics and plot data are consistent
        stats = stats_responses[0]
        plot_data = plot_responses[0]
        
        self.assertEqual(stats['count'], len(plot_data['histogram']['speeds']))
        self.assertEqual(stats['mean'], sum(plot_data['histogram']['speeds']) / len(plot_data['histogram']['speeds']))
        self.assertEqual(stats['min'], min(plot_data['histogram']['speeds']))
        self.assertEqual(stats['max'], max(plot_data['histogram']['speeds']))

if __name__ == '__main__':
    unittest.main()
