#!/usr/bin/env python3
"""
Integration tests for API endpoints - Data flow from frontend to backend

Tests all API endpoints to ensure they work correctly and maintain data integrity
through the request/response cycle.
"""

import unittest
import sys
import os
import json
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from iss_speed_html_dashboard_v2_clean import app
except ImportError:
    from iss_speed_html_dashboard_v2 import app

class TestAPIEndpoints(unittest.TestCase):
    """Test suite for API endpoints"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.app.testing = True
        
        self.test_dir = tempfile.mkdtemp()
        self.create_test_photos_folder()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_test_photos_folder(self):
        """Create a test photos folder with sample images"""
        photos_dir = os.path.join(self.test_dir, "photos-test")
        os.makedirs(photos_dir, exist_ok=True)
        
        # Create some dummy JPG files
        for i in range(5):
            dummy_file = os.path.join(photos_dir, f"test_{i:03d}.jpg")
            with open(dummy_file, 'w') as f:
                f.write("dummy jpg content")
    
    def test_folders_endpoint(self):
        """Test /api/folders endpoint"""
        with patch('os.listdir') as mock_listdir:
            mock_listdir.return_value = ['photos-test', 'other-folder']
            
            with patch('os.path.isdir') as mock_isdir:
                mock_isdir.side_effect = lambda x: x == 'photos-test'
                
                with patch('os.listdir') as mock_photos_listdir:
                    mock_photos_listdir.return_value = ['test_001.jpg', 'test_002.jpg']
                    
                    response = self.app.get('/api/folders')
                    
                    self.assertEqual(response.status_code, 200)
                    data = json.loads(response.data)
                    
                    self.assertIsInstance(data, list)
                    self.assertGreater(len(data), 0)
                    
                    folder = data[0]
                    self.assertIn('name', folder)
                    self.assertIn('count', folder)
                    self.assertEqual(folder['name'], 'photos-test')
                    self.assertEqual(folder['count'], 2)
    
    def test_process_range_endpoint(self):
        """Test /api/process-range endpoint"""
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
                        'folder': 'photos-test',
                        'start_idx': 0,
                        'end_idx': 2,
                        'algorithm': 'ORB',
                        'use_flann': False,
                        'use_ransac': False
                    }),
                    content_type='application/json'
                )
                
                self.assertEqual(response.status_code, 200)
                data = json.loads(response.data)
                
                self.assertIn('status', data)
                self.assertEqual(data['status'], 'success')
                self.assertIn('message', data)
    
    def test_statistics_endpoint(self):
        """Test /api/statistics endpoint"""
        # First, set up some processed data
        with patch('iss_speed_html_dashboard_v2_clean.processed_matches', [
            {'speed': 100.0, 'pixel_distance': 10.0},
            {'speed': 200.0, 'pixel_distance': 20.0},
            {'speed': 300.0, 'pixel_distance': 30.0}
        ]):
            response = self.app.get('/api/statistics')
            
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            
            self.assertIn('mean', data)
            self.assertIn('median', data)
            self.assertIn('std_dev', data)
            self.assertIn('min', data)
            self.assertIn('max', data)
            self.assertIn('count', data)
            
            # Check calculated values
            self.assertEqual(data['count'], 3)
            self.assertEqual(data['mean'], 200.0)
            self.assertEqual(data['min'], 100.0)
            self.assertEqual(data['max'], 300.0)
    
    def test_statistics_endpoint_no_data(self):
        """Test /api/statistics endpoint with no data"""
        with patch('iss_speed_html_dashboard_v2_clean.processed_matches', []):
            response = self.app.get('/api/statistics')
            
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            
            # Should return default values
            self.assertEqual(data['count'], 0)
            self.assertEqual(data['mean'], 0.0)
            self.assertEqual(data['median'], 0.0)
    
    def test_plot_data_endpoint(self):
        """Test /api/plot-data endpoint"""
        with patch('iss_speed_html_dashboard_v2_clean.processed_matches', [
            {'speed': 100.0, 'pair_index': 0},
            {'speed': 200.0, 'pair_index': 0},
            {'speed': 300.0, 'pair_index': 1},
            {'speed': 400.0, 'pair_index': 1}
        ]):
            response = self.app.get('/api/plot-data')
            
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            
            self.assertIn('histogram', data)
            self.assertIn('pairs', data)
            
            # Check histogram data
            self.assertIn('speeds', data['histogram'])
            self.assertIn('median', data['histogram'])
            
            # Check pairs data
            self.assertIn('pairs', data['pairs'])
            self.assertIn('means', data['pairs'])
            self.assertIn('medians', data['pairs'])
    
    def test_apply_filters_endpoint(self):
        """Test /api/apply-filters endpoint"""
        # Set up some processed data
        with patch('iss_speed_html_dashboard_v2_clean.processed_matches', [
            {'speed': 100.0, 'pixel_distance': 10.0},
            {'speed': 200.0, 'pixel_distance': 20.0},
            {'speed': 1000.0, 'pixel_distance': 100.0},  # Outlier
        ]):
            response = self.app.post('/api/apply-filters',
                data=json.dumps({
                    'enable_std_dev': True,
                    'std_dev_multiplier': 2.0
                }),
                content_type='application/json'
            )
            
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            
            self.assertIn('status', data)
            self.assertEqual(data['status'], 'success')
            self.assertIn('filtered_count', data)
    
    def test_clear_cache_endpoint(self):
        """Test /api/clear-v2-cache endpoint"""
        response = self.app.post('/api/clear-v2-cache')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'success')
    
    def test_cache_status_endpoint(self):
        """Test /api/cache-status endpoint"""
        response = self.app.get('/api/cache-status')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertIn('total_files', data)
        self.assertIn('total_size', data)
        self.assertIn('data_loaded', data)
    
    def test_algorithm_comparison_endpoint(self):
        """Test /api/algorithm-comparison endpoint"""
        with patch('iss_speed_html_dashboard_v2_clean.run_algorithm_comparison') as mock_comparison:
            response = self.app.post('/api/algorithm-comparison',
                data=json.dumps({
                    'algorithms': ['ORB', 'SIFT'],
                    'folder': 'photos-test',
                    'start_idx': 0,
                    'end_idx': 2
                }),
                content_type='application/json'
            )
            
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            
            self.assertIn('status', data)
            self.assertEqual(data['status'], 'started')
    
    def test_processing_status_endpoint(self):
        """Test /api/processing-status endpoint"""
        response = self.app.get('/api/processing-status')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertIn('status', data)
        self.assertIn('progress', data)
    
    def test_pair_endpoint(self):
        """Test /api/pair/<int:pair_num> endpoint"""
        with patch('iss_speed_html_dashboard_v2_clean.processed_matches', [
            {
                'pair_index': 0,
                'speed': 100.0,
                'image1_path': 'test1.jpg',
                'image2_path': 'test2.jpg'
            }
        ]):
            response = self.app.get('/api/pair/1')
            
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            
            self.assertIn('pair_num', data)
            self.assertIn('match_count', data)
            self.assertIn('avg_speed', data)
            self.assertEqual(data['pair_num'], 1)
    
    def test_pair_endpoint_not_found(self):
        """Test /api/pair/<int:pair_num> endpoint with non-existent pair"""
        with patch('iss_speed_html_dashboard_v2_clean.processed_matches', []):
            response = self.app.get('/api/pair/999')
            
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            
            self.assertIn('error', data)
            self.assertEqual(data['error'], 'Pair not found')
    
    def test_images_endpoint(self):
        """Test /api/images/<path:folder_name> endpoint"""
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            
            with patch('os.listdir') as mock_listdir:
                mock_listdir.return_value = ['test1.jpg', 'test2.jpg', 'readme.txt']
                
                response = self.app.get('/api/images/photos-test')
                
                self.assertEqual(response.status_code, 200)
                data = json.loads(response.data)
                
                self.assertIn('folder', data)
                self.assertIn('images', data)
                self.assertIn('count', data)
                
                self.assertEqual(data['folder'], 'photos-test')
                self.assertEqual(data['count'], 2)  # Only JPG files
                self.assertIn('test1.jpg', data['images'])
                self.assertIn('test2.jpg', data['images'])
    
    def test_analyze_reference_endpoint(self):
        """Test /api/analyze-reference endpoint"""
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            
            with patch('cv2.imread') as mock_imread:
                mock_imread.return_value = np.ones((100, 100, 3), dtype=np.uint8) * 128
                
                with patch('cv2.ORB_create') as mock_orb:
                    mock_detector = Mock()
                    mock_detector.detectAndCompute.return_value = ([Mock()] * 10, np.ones((10, 32), dtype=np.uint8))
                    mock_orb.return_value = mock_detector
                    
                    response = self.app.post('/api/analyze-reference',
                        data=json.dumps({
                            'folder': 'photos-test',
                            'image_name': 'test1.jpg'
                        }),
                        content_type='application/json'
                    )
                    
                    self.assertEqual(response.status_code, 200)
                    data = json.loads(response.data)
                    
                    self.assertIn('image_name', data)
                    self.assertIn('brightness', data)
                    self.assertIn('contrast', data)
                    self.assertIn('keypoint_count', data)
                    self.assertIn('width', data)
                    self.assertIn('height', data)
    
    def test_data_integrity_through_api(self):
        """Test data integrity through API calls"""
        # This test ensures data doesn't get corrupted through API calls
        
        # Set up test data
        original_matches = [
            {'speed': 100.0, 'pair_index': 0, 'pixel_distance': 10.0},
            {'speed': 200.0, 'pair_index': 0, 'pixel_distance': 20.0},
            {'speed': 300.0, 'pair_index': 1, 'pixel_distance': 30.0}
        ]
        
        with patch('iss_speed_html_dashboard_v2_clean.processed_matches', original_matches):
            # Get statistics
            stats_response = self.app.get('/api/statistics')
            stats_data = json.loads(stats_response.data)
            
            # Get plot data
            plot_response = self.app.get('/api/plot-data')
            plot_data = json.loads(plot_response.data)
            
            # Verify data integrity
            self.assertEqual(stats_data['count'], len(original_matches))
            self.assertEqual(len(plot_data['histogram']['speeds']), len(original_matches))
            
            # Verify speeds are preserved
            original_speeds = [m['speed'] for m in original_matches]
            plot_speeds = plot_data['histogram']['speeds']
            self.assertEqual(sorted(original_speeds), sorted(plot_speeds))
            
            # Verify statistics are calculated correctly
            self.assertEqual(stats_data['mean'], sum(original_speeds) / len(original_speeds))
            self.assertEqual(stats_data['min'], min(original_speeds))
            self.assertEqual(stats_data['max'], max(original_speeds))
    
    def test_error_handling(self):
        """Test error handling in API endpoints"""
        # Test with invalid JSON
        response = self.app.post('/api/process-range',
            data="invalid json",
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        
        # Test with missing required fields
        response = self.app.post('/api/process-range',
            data=json.dumps({}),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

if __name__ == '__main__':
    unittest.main()
