#!/usr/bin/env python3
"""
Independent Baseline Generation - Uses only OpenCV + GSD, everything else is our own implementation

This creates a truly independent baseline that can validate the clean version.
"""

import os
import cv2
import numpy as np
import statistics
import time
import json
from datetime import datetime
from collections import Counter
import re

# Reuse only the GSD constant from clean version
GSD = 12648  # Ground Sample Distance (from clean version)

def extract_timestamp_from_filename(filename):
    """Extract timestamp from ISS image filename - independent implementation"""
    # Pattern: YYYYMMDD-HHMMSS_XXXXXXXXX_o.jpg
    pattern = r'(\d{8})-(\d{6})_'
    match = re.search(pattern, filename)
    if match:
        date_str = match.group(1)  # YYYYMMDD
        time_str = match.group(2)  # HHMMSS
        
        # Convert to datetime
        year = int(date_str[:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        hour = int(time_str[:2])
        minute = int(time_str[2:4])
        second = int(time_str[4:6])
        
        from datetime import datetime
        return datetime(year, month, day, hour, minute, second)
    return None

def get_time_difference_independent(image1_path, image2_path):
    """Calculate time difference between two images - independent implementation"""
    timestamp1 = extract_timestamp_from_filename(os.path.basename(image1_path))
    timestamp2 = extract_timestamp_from_filename(os.path.basename(image2_path))
    
    if timestamp1 and timestamp2:
        time_diff = (timestamp2 - timestamp1).total_seconds()
        return time_diff
    return 0.0

def enhance_image_contrast_independent(image, method='clahe', clip_limit=3.0):
    """Independent contrast enhancement implementation"""
    if method == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
        return clahe.apply(image)
    elif method == 'histogram_eq':
        return cv2.equalizeHist(image)
    elif method == 'gamma':
        gamma = 1.5
        enhanced = np.power(image / 255.0, gamma) * 255.0
        return enhanced.astype(np.uint8)
    elif method == 'unsharp':
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
    else:
        return image

def detect_features_independent(image, algorithm, max_features):
    """Independent feature detection using OpenCV"""
    if algorithm == 'ORB':
        detector = cv2.ORB_create(nfeatures=max_features)
    elif algorithm == 'SIFT':
        detector = cv2.SIFT_create(nfeatures=max_features)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    keypoints, descriptors = detector.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features_independent(descriptors1, descriptors2, use_flann):
    """Independent feature matching using OpenCV"""
    if descriptors1 is None or descriptors2 is None:
        return []
    
    if use_flann:
        # FLANN-based matching
        if descriptors1.dtype == np.uint8:  # ORB descriptors
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                              table_number=6,
                              key_size=12,
                              multi_probe_level=1)
        else:  # SIFT descriptors
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        return good_matches
    else:
        # Brute force matching
        if descriptors1.dtype == np.uint8:  # ORB descriptors
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:  # SIFT descriptors
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
        matches = matcher.match(descriptors1, descriptors2)
        # Sort by distance and take best matches
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

def apply_ransac_filtering_independent(keypoints1, keypoints2, matches, threshold=5.0, min_matches=10):
    """Independent RANSAC filtering implementation"""
    if len(matches) < min_matches:
        return matches
    
    # Extract matched points
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Find homography using RANSAC
    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, threshold)
    
    if homography is not None and mask is not None:
        # Filter matches based on inliers
        inlier_matches = [matches[i] for i in range(len(matches)) if mask[i][0] == 1]
        return inlier_matches
    
    return matches

def calculate_speed_independent(match, keypoints1, keypoints2, time_diff):
    """Independent speed calculation implementation"""
    if time_diff <= 0:
        return None
    
    # Get matched keypoints
    kp1 = keypoints1[match.queryIdx]
    kp2 = keypoints2[match.trainIdx]
    
    # Calculate pixel distance
    dx = kp2.pt[0] - kp1.pt[0]
    dy = kp2.pt[1] - kp1.pt[1]
    pixel_distance = np.sqrt(dx*dx + dy*dy)
    
    # Convert to real distance using GSD (GSD is in cm/pixel, convert to km)
    real_distance = pixel_distance * GSD / 100000  # Convert cm to km
    
    # Calculate speed (km/s)
    speed = real_distance / time_diff
    
    return speed

def process_image_pair_independent(image1_path, image2_path, algorithm, use_flann, use_ransac=False, 
                                 ransac_threshold=5.0, ransac_min_matches=10, contrast_enhancement='clahe', 
                                 max_features=1000):
    """Independent image pair processing - our own implementation"""
    try:
        # Load images
        img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            return []
        
        # Apply contrast enhancement
        if contrast_enhancement != 'none':
            img1 = enhance_image_contrast_independent(img1, contrast_enhancement)
            img2 = enhance_image_contrast_independent(img2, contrast_enhancement)
        
        # Detect features
        kp1, des1 = detect_features_independent(img1, algorithm, max_features)
        kp2, des2 = detect_features_independent(img2, algorithm, max_features)
        
        if des1 is None or des2 is None:
            return []
        
        # Match features
        matches = match_features_independent(des1, des2, use_flann)
        
        # Apply RANSAC filtering if requested
        if use_ransac and len(matches) >= ransac_min_matches:
            matches = apply_ransac_filtering_independent(kp1, kp2, matches, ransac_threshold, ransac_min_matches)
        
        # Calculate time difference
        time_diff = get_time_difference_independent(image1_path, image2_path)
        
        # Calculate speeds for all matches
        match_results = []
        for i, match in enumerate(matches):
            speed = calculate_speed_independent(match, kp1, kp2, time_diff)
            if speed is not None:
                match_results.append({
                    'match_index': i,
                    'speed': speed,
                    'time_difference': time_diff,
                    'algorithm': algorithm,
                    'use_flann': use_flann,
                    'use_ransac': use_ransac,
                    'contrast_enhancement': contrast_enhancement,
                    'max_features': max_features
                })
        
        return match_results
        
    except Exception as e:
        print(f"Error processing image pair: {e}")
        return []

def calculate_statistics_independent(speeds, matches):
    """Independent statistics calculation - our own implementation"""
    if not speeds:
        return None
    
    # Mean: is the mean of all the matches from all pairs loaded
    mean = statistics.mean(speeds)
    
    # Median: is the median of all the matches from all pairs loaded
    median = statistics.median(speeds)
    
    # Standard deviation: stddev of the speed of all the matches
    std_dev = statistics.stdev(speeds) if len(speeds) > 1 else 0.0
    
    # Total matches: total number of matches across all the pairs
    total_matches = len(matches)
    
    # Total pairs: How many pairs are processed
    total_pairs = len(set(match.get('pair_index', 0) for match in matches))
    
    # Match mode speed: the most common speed at one decimal place from all the matches from all the pairs
    speeds_rounded = [round(speed, 1) for speed in speeds]
    match_mode = Counter(speeds_rounded).most_common(1)[0][0] if speeds_rounded else None
    
    # Pair mode speed: the most average pair speed at one decimal place
    # Group matches by pair and calculate average speed per pair
    pair_speeds = {}
    for match in matches:
        pair_idx = match.get('pair_index', 0)
        speed = match.get('speed', 0)
        if pair_idx not in pair_speeds:
            pair_speeds[pair_idx] = []
        pair_speeds[pair_idx].append(speed)
    
    # Calculate average speed for each pair
    pair_averages = [statistics.mean(pair_speeds[pair_idx]) for pair_idx in pair_speeds]
    pair_averages_rounded = [round(avg, 1) for avg in pair_averages]
    pair_mode = Counter(pair_averages_rounded).most_common(1)[0][0] if pair_averages_rounded else None
    
    return {
        'mean': mean,
        'median': median,
        'std_dev': std_dev,
        'total_matches': total_matches,
        'total_pairs': total_pairs,
        'match_mode': match_mode,
        'pair_mode': pair_mode
    }

def generate_independent_baseline():
    """Generate baseline statistics using independent implementation"""
    print("🚀 Generating INDEPENDENT Baseline Statistics")
    print("=" * 60)
    print("✅ Using: OpenCV + GSD constant")
    print("✅ Independent: All calculations, processing, statistics")
    print("=" * 60)
    
    # Test parameters
    photos_dir = "photos-1"
    start_idx = 0
    end_idx = 2  # First 3 images (0-2) = 2 pairs
    
    # Verify photos-1 directory exists
    if not os.path.exists(photos_dir):
        print(f"❌ Photos directory {photos_dir} not found")
        return None
    
    # Get first 3 image files
    image_files = sorted([f for f in os.listdir(photos_dir) if f.lower().endswith('.jpg')])
    if len(image_files) < 3:
        print(f"❌ Not enough images in {photos_dir} (found {len(image_files)}, need 3)")
        return None
    
    image_files = image_files[:3]
    print(f"📸 Processing images: {image_files[0]} → {image_files[1]} → {image_files[2]} (2 pairs)")
    
    # Test key algorithm combinations
    test_combinations = [
        {'algorithm': 'ORB', 'use_flann': False, 'use_ransac': False, 'contrast_enhancement': 'none', 'max_features': 1000},
        {'algorithm': 'ORB', 'use_flann': False, 'use_ransac': False, 'contrast_enhancement': 'clahe', 'max_features': 1000},  # Your UI parameters
        {'algorithm': 'ORB', 'use_flann': True, 'use_ransac': True, 'contrast_enhancement': 'clahe', 'max_features': 1000},
        {'algorithm': 'SIFT', 'use_flann': False, 'use_ransac': False, 'contrast_enhancement': 'none', 'max_features': 1000},
        {'algorithm': 'SIFT', 'use_flann': True, 'use_ransac': True, 'contrast_enhancement': 'clahe', 'max_features': 1000},
    ]
    
    baseline_stats = {}
    
    for i, combo in enumerate(test_combinations):
        combination_name = f"{combo['algorithm']}_FLANN{combo['use_flann']}_RANSAC{combo['use_ransac']}_{combo['contrast_enhancement']}_FEAT{combo['max_features']}"
        print(f"🔍 [{i+1}/{len(test_combinations)}] Generating INDEPENDENT baseline: {combination_name}")
        
        try:
            matches = []
            start_time = time.time()
            
            # Process both pairs (0→1 and 1→2) using INDEPENDENT implementation
            for j in range(2):  # 2 pairs
                img1_path = os.path.join(photos_dir, image_files[j])
                img2_path = os.path.join(photos_dir, image_files[j+1])
                
                pair_matches = process_image_pair_independent(
                    img1_path, img2_path, combo['algorithm'], combo['use_flann'], 
                    use_ransac=combo['use_ransac'],
                    ransac_threshold=5.0,
                    ransac_min_matches=10,
                    contrast_enhancement=combo['contrast_enhancement'],
                    max_features=combo['max_features']
                )
                
                if pair_matches:
                    # Set the correct pair_index for each match
                    for match in pair_matches:
                        match['pair_index'] = j
                    matches.extend(pair_matches)
            
            processing_time = time.time() - start_time
            
            if matches:
                speeds = [match.get('speed', 0) for match in matches if match.get('speed') is not None]
                if speeds:
                    # Calculate statistics using INDEPENDENT implementation
                    stats = calculate_statistics_independent(speeds, matches)
                    
                    baseline_stats[combination_name] = {
                        'algorithm': combo['algorithm'],
                        'use_flann': combo['use_flann'],
                        'use_ransac': combo['use_ransac'],
                        'contrast_enhancement': combo['contrast_enhancement'],
                        'max_features': combo['max_features'],
                        'total_matches': len(matches),
                        'valid_speeds': len(speeds),
                        'processing_time': processing_time,
                        'statistics': stats,
                        'success': True,
                        'timestamp': datetime.now().isoformat(),
                        'implementation': 'INDEPENDENT'
                    }
                    
                    print(f"  ✅ INDEPENDENT baseline generated: {len(speeds)} speeds, {processing_time:.2f}s")
                    print(f"     Mean: {stats['mean']:.2f} km/s")
                    print(f"     Median: {stats['median']:.2f} km/s")
                    print(f"     Pair Mode: {stats['pair_mode']:.1f} km/s")
                    print(f"     Match Mode: {stats['match_mode']:.1f} km/s")
                    print(f"     Total Matches: {stats['total_matches']}")
                    print(f"     Total Pairs: {stats['total_pairs']}")
                    print(f"     Std Dev: {stats['std_dev']:.2f} km/s")
                else:
                    baseline_stats[combination_name] = {
                        **combo,
                        'total_matches': len(matches),
                        'valid_speeds': 0,
                        'processing_time': processing_time,
                        'statistics': None,
                        'success': False,
                        'error': 'No valid speeds calculated',
                        'implementation': 'INDEPENDENT'
                    }
                    print(f"  ❌ No valid speeds")
            else:
                baseline_stats[combination_name] = {
                    **combo,
                    'total_matches': 0,
                    'valid_speeds': 0,
                    'processing_time': processing_time,
                    'statistics': None,
                    'success': False,
                    'error': 'No matches found',
                    'implementation': 'INDEPENDENT'
                }
                print(f"  ❌ No matches found")
                
        except Exception as e:
            baseline_stats[combination_name] = {
                **combo,
                'total_matches': 0,
                'valid_speeds': 0,
                'processing_time': 0,
                'statistics': None,
                'success': False,
                'error': str(e),
                'implementation': 'INDEPENDENT'
            }
            print(f"  ❌ Error: {e}")
    
    # Save baseline to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    baseline_file = f"independent_baseline_stats_{timestamp}.json"
    
    with open(baseline_file, 'w') as f:
        json.dump(baseline_stats, f, indent=2)
    
    successful_combinations = sum(1 for combo in baseline_stats.values() if combo.get('success', False))
    
    print(f"\n📊 INDEPENDENT BASELINE STATISTICS GENERATED")
    print("=" * 60)
    print(f"Total combinations: {len(test_combinations)}")
    print(f"Successful combinations: {successful_combinations}")
    print(f"Baseline file: {baseline_file}")
    print(f"Implementation: INDEPENDENT (OpenCV + GSD only)")
    
    print(f"\n✅ Independent baseline statistics saved to: {baseline_file}")
    print("🎯 This baseline can now validate the clean version independently!")
    
    return baseline_file

if __name__ == "__main__":
    generate_independent_baseline()
