#!/usr/bin/env python3
"""
Diagnostic script to analyze a specific image pair with default parameters.
This script processes the problematic pair and collects detailed diagnostic data.
"""

import sys
import os
import json
from datetime import datetime
import numpy as np

# Add current directory to path to import the main module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import iss_speed_html_dashboard_v2_clean as app_module

def diagnose_pair(image1_path, image2_path, pair_name="Test Pair"):
    """
    Process a single image pair with default parameters and collect diagnostic data.
    
    Args:
        image1_path: Path to first image
        image2_path: Path to second image
        pair_name: Name/description of the pair
    """
    print(f"\n{'='*80}")
    print(f"🔍 DIAGNOSTIC ANALYSIS: {pair_name}")
    print(f"{'='*80}\n")
    
    # Verify files exist
    if not os.path.exists(image1_path):
        print(f"❌ ERROR: Image 1 not found: {image1_path}")
        return None
    if not os.path.exists(image2_path):
        print(f"❌ ERROR: Image 2 not found: {image2_path}")
        return None
    
    print(f"📸 Image 1: {os.path.basename(image1_path)}")
    print(f"📸 Image 2: {os.path.basename(image2_path)}\n")
    
    # Collect diagnostic data
    diagnostic_data = {
        'pair_name': pair_name,
        'image1': os.path.basename(image1_path),
        'image2': os.path.basename(image2_path),
        'image1_path': image1_path,
        'image2_path': image2_path,
        'timestamp': datetime.now().isoformat(),
        'default_parameters': {
            'max_features': app_module.MAX_FEATURES,
            'gsd': 12648,  # Default ESA GSD
            'clear_brightness_min': 120,
            'clear_contrast_min': 50,
            'cloudy_brightness_max': 60,
            'cloudy_contrast_max': 15,
        }
    }
    
    # 1. Get time difference
    print("⏱️  Step 1: Calculating time difference...")
    time_difference = app_module.get_time_difference(image1_path, image2_path)
    diagnostic_data['time_difference'] = time_difference
    print(f"   Time difference: {time_difference:.3f} seconds")
    
    if time_difference <= 0:
        print("   ⚠️  WARNING: Invalid time difference!")
        return diagnostic_data
    
    # 2. Analyze image characteristics
    print("\n📊 Step 2: Analyzing image characteristics...")
    img1_chars = app_module.analyze_image_characteristics(
        image1_path, 
        clear_brightness_min=120,
        clear_contrast_min=50,
        cloudy_brightness_max=60,
        cloudy_contrast_max=15
    )
    img2_chars = app_module.analyze_image_characteristics(
        image2_path,
        clear_brightness_min=120,
        clear_contrast_min=50,
        cloudy_brightness_max=60,
        cloudy_contrast_max=15
    )
    
    diagnostic_data['image1_characteristics'] = img1_chars
    diagnostic_data['image2_characteristics'] = img2_chars
    
    avg_brightness = (img1_chars['brightness'] + img2_chars['brightness']) / 2
    avg_contrast = (img1_chars['contrast'] + img2_chars['contrast']) / 2
    
    if img1_chars['cloudiness'] == 'clear' and img2_chars['cloudiness'] == 'clear':
        overall_cloudiness = 'clear'
    elif img1_chars['cloudiness'] == 'mostly cloudy' or img2_chars['cloudiness'] == 'mostly cloudy':
        overall_cloudiness = 'mostly cloudy'
    else:
        overall_cloudiness = 'partly cloudy'
    
    diagnostic_data['overall_cloudiness'] = overall_cloudiness
    diagnostic_data['avg_brightness'] = avg_brightness
    diagnostic_data['avg_contrast'] = avg_contrast
    
    print(f"   Image 1: brightness={img1_chars['brightness']:.1f}, contrast={img1_chars['contrast']:.1f}, cloudiness={img1_chars['cloudiness']}")
    print(f"   Image 2: brightness={img2_chars['brightness']:.1f}, contrast={img2_chars['contrast']:.1f}, cloudiness={img2_chars['cloudiness']}")
    print(f"   Overall: brightness={avg_brightness:.1f}, contrast={avg_contrast:.1f}, cloudiness={overall_cloudiness}")
    
    # 3. Process image pair (using the SAME function the application uses)
    print("\n🔍 Step 3: Processing image pair with application's default parameters...")
    print(f"   Using MAX_FEATURES={app_module.MAX_FEATURES}")
    print(f"   Algorithm: ORB (default)")
    print(f"   FLANN: False (default)")
    print(f"   RANSAC/Homography: False (default)")
    print(f"   Contrast Enhancement: CLAHE (default)")
    
    # Set enhancement method (same as application does)
    app_module.process_image_pair.enhancement_method = 'clahe'
    
    # Use the application's process_image_pair function (the one at line 3315)
    # This is the function used by /api/process-range
    keypoint_data = app_module.process_image_pair(
        image1_path,
        image2_path,
        algorithm='ORB',  # Default algorithm
        use_flann=False,  # Default
        use_ransac_homography=False,  # Default
        ransac_threshold=5.0,  # Default
        ransac_min_matches=10,  # Default
        max_features=app_module.MAX_FEATURES
    )
    
    if not keypoint_data:
        print("   ❌ ERROR: No keypoint data returned!")
        diagnostic_data['error'] = "No keypoint data returned"
        return diagnostic_data
    
    print(f"   ✅ Found {len(keypoint_data)} keypoint matches")
    
    # 4. Analyze keypoint data
    print("\n📈 Step 4: Analyzing keypoint matches...")
    
    # Extract pixel distances and speeds
    # The application's process_image_pair returns matches with 'pixel_distance' and 'speed'
    pixel_distances = []
    speeds = []
    for kp in keypoint_data:
        dist = kp.get('pixel_distance') or kp.get('distance', 0)
        speed = kp.get('speed', 0)
        pixel_distances.append(dist)
        speeds.append(speed)
    
    diagnostic_data['match_count'] = len(keypoint_data)
    diagnostic_data['pixel_distances'] = {
        'min': float(np.min(pixel_distances)),
        'max': float(np.max(pixel_distances)),
        'mean': float(np.mean(pixel_distances)),
        'median': float(np.median(pixel_distances)),
        'std': float(np.std(pixel_distances)),
        'percentiles': {
            '5': float(np.percentile(pixel_distances, 5)),
            '25': float(np.percentile(pixel_distances, 25)),
            '50': float(np.percentile(pixel_distances, 50)),
            '75': float(np.percentile(pixel_distances, 75)),
            '95': float(np.percentile(pixel_distances, 95)),
        }
    }
    
    diagnostic_data['speeds'] = {
        'min': float(np.min(speeds)),
        'max': float(np.max(speeds)),
        'mean': float(np.mean(speeds)),
        'median': float(np.median(speeds)),
        'std': float(np.std(speeds)),
        'percentiles': {
            '5': float(np.percentile(speeds, 5)),
            '25': float(np.percentile(speeds, 25)),
            '50': float(np.percentile(speeds, 50)),
            '75': float(np.percentile(speeds, 75)),
            '95': float(np.percentile(speeds, 95)),
        }
    }
    
    print(f"   Pixel distances: min={diagnostic_data['pixel_distances']['min']:.2f}, "
          f"max={diagnostic_data['pixel_distances']['max']:.2f}, "
          f"mean={diagnostic_data['pixel_distances']['mean']:.2f}, "
          f"median={diagnostic_data['pixel_distances']['median']:.2f}")
    print(f"   Speeds: min={diagnostic_data['speeds']['min']:.3f} km/s, "
          f"max={diagnostic_data['speeds']['max']:.3f} km/s, "
          f"mean={diagnostic_data['speeds']['mean']:.3f} km/s, "
          f"median={diagnostic_data['speeds']['median']:.3f} km/s")
    
    # 5. Calculate expected speed
    print("\n🎯 Step 5: Speed analysis...")
    expected_speed = 7.66  # Target ISS speed
    mean_speed = diagnostic_data['speeds']['mean']
    speed_ratio = mean_speed / expected_speed
    
    diagnostic_data['speed_analysis'] = {
        'expected_speed': expected_speed,
        'mean_speed': mean_speed,
        'speed_ratio': speed_ratio,
        'is_high': mean_speed > expected_speed * 1.5,
        'is_low': mean_speed < expected_speed * 0.5,
    }
    
    print(f"   Expected speed: {expected_speed:.3f} km/s")
    print(f"   Mean speed: {mean_speed:.3f} km/s")
    print(f"   Ratio: {speed_ratio:.2f}x")
    
    if speed_ratio > 1.5:
        print(f"   ⚠️  WARNING: Speed is {speed_ratio:.2f}x higher than expected!")
    elif speed_ratio < 0.5:
        print(f"   ⚠️  WARNING: Speed is {speed_ratio:.2f}x lower than expected!")
    else:
        print(f"   ✅ Speed is within reasonable range")
    
    # 6. Identify potential issues
    print("\n🔎 Step 6: Identifying potential issues...")
    issues = []
    
    # Check for outliers
    if diagnostic_data['speeds']['std'] > diagnostic_data['speeds']['mean'] * 0.5:
        issues.append("High speed variance (many outliers)")
        print("   ⚠️  High speed variance detected - likely outliers present")
    
    # Check pixel distances
    if diagnostic_data['pixel_distances']['max'] > diagnostic_data['pixel_distances']['mean'] * 3:
        issues.append("Very large pixel distances (bad matches)")
        print("   ⚠️  Very large pixel distances detected - possible bad matches")
    
    # Check match count
    if len(keypoint_data) < 10:
        issues.append("Low match count")
        print(f"   ⚠️  Low match count: {len(keypoint_data)} matches")
    
    # Check time difference
    if time_difference < 5 or time_difference > 30:
        issues.append(f"Unusual time difference: {time_difference:.3f}s")
        print(f"   ⚠️  Unusual time difference: {time_difference:.3f}s")
    
    if not issues:
        print("   ✅ No obvious issues detected")
    
    diagnostic_data['issues'] = issues
    
    # 7. Sample keypoint data (first 10)
    print("\n📋 Step 7: Sample keypoint data (first 10 matches)...")
    diagnostic_data['sample_keypoints'] = []
    for i, kp in enumerate(keypoint_data[:10]):
        dist = kp.get('pixel_distance') or kp.get('distance', 0)
        sample = {
            'index': i,
            'pixel_distance': float(dist),
            'speed': float(kp.get('speed', 0)),
            'time_diff': float(time_difference),  # Time diff is same for all matches in a pair
            'match_distance': float(kp.get('match_distance', 0)),  # Quality metric
        }
        diagnostic_data['sample_keypoints'].append(sample)
        print(f"   Match {i+1}: distance={sample['pixel_distance']:.2f}px, "
              f"speed={sample['speed']:.3f} km/s")
    
    print(f"\n{'='*80}")
    print("✅ DIAGNOSTIC ANALYSIS COMPLETE")
    print(f"{'='*80}\n")
    
    return diagnostic_data

def compare_pairs(pair_data_list):
    """Compare multiple pairs to identify differences."""
    print(f"\n{'='*80}")
    print("📊 COMPARING PAIRS")
    print(f"{'='*80}\n")
    
    for i, data in enumerate(pair_data_list):
        print(f"Pair {i+1}: {data['pair_name']}")
        print(f"  Mean speed: {data['speeds']['mean']:.3f} km/s")
        print(f"  Match count: {data['match_count']}")
        print(f"  Mean pixel distance: {data['pixel_distances']['mean']:.2f} px")
        print(f"  Time difference: {data['time_difference']:.3f} s")
        print()

def main():
    """Main function to run diagnostics."""
    # Default paths - adjust these to your photos-1 folder
    photos_dir = "photos-1"
    
    if len(sys.argv) > 1:
        photos_dir = sys.argv[1]
    
    if not os.path.exists(photos_dir):
        print(f"❌ ERROR: Directory not found: {photos_dir}")
        print(f"Usage: {sys.argv[0]} [photos_directory]")
        return
    
    # The problematic pair
    image1 = os.path.join(photos_dir, "20230423-112638_53238988145_o.jpg")
    image2 = os.path.join(photos_dir, "20230423-112652_53238502736_o.jpg")
    
    # Adjacent pairs for comparison
    image_before = os.path.join(photos_dir, "20230423-112624_53238989495_o.jpg")
    image_after = os.path.join(photos_dir, "20230423-112706_53238502731_o.jpg")
    
    results = []
    
    # Process problematic pair
    if os.path.exists(image1) and os.path.exists(image2):
        data = diagnose_pair(image1, image2, "Problematic Pair (19.929 km/s)")
        if data:
            results.append(data)
    else:
        print(f"❌ Problematic pair images not found in {photos_dir}")
    
    # Process pair before
    if os.path.exists(image_before) and os.path.exists(image1):
        data = diagnose_pair(image_before, image1, "Pair Before (8.322 km/s)")
        if data:
            results.append(data)
    
    # Process pair after
    if os.path.exists(image2) and os.path.exists(image_after):
        data = diagnose_pair(image2, image_after, "Pair After (7.338 km/s)")
        if data:
            results.append(data)
    
    # Compare pairs
    if len(results) > 1:
        compare_pairs(results)
    
    # Save results to JSON
    output_file = f"diagnostic_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("📊 SUMMARY")
    print(f"{'='*80}\n")
    
    for data in results:
        print(f"{data['pair_name']}:")
        print(f"  Mean speed: {data['speeds']['mean']:.3f} km/s")
        print(f"  Issues: {', '.join(data.get('issues', [])) if data.get('issues') else 'None'}")
        print()

if __name__ == "__main__":
    main()

