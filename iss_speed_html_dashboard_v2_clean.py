
from exif import Image
from datetime import datetime
import cv2
import math
import os
import numpy as np
import json
import base64
import pickle
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, send_file
import logging
import threading
# Lazy import TensorFlow to avoid blocking startup
TENSORFLOW_AVAILABLE = None
_tf_module = None

def _check_tensorflow():
    """Lazy check for TensorFlow availability"""
    global TENSORFLOW_AVAILABLE, _tf_module
    if TENSORFLOW_AVAILABLE is None:
        try:
            import tensorflow as tf
            _tf_module = tf
            TENSORFLOW_AVAILABLE = True
        except ImportError:
            TENSORFLOW_AVAILABLE = False
            print("⚠️ TensorFlow not available - ML classification will be disabled")
    return TENSORFLOW_AVAILABLE

# For backward compatibility, expose tf when needed
def get_tf():
    """Get TensorFlow module (lazy loaded)"""
    if _check_tensorflow():
        return _tf_module
    return None

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ PIL not available - ML classification will be disabled")
import time
import pickle
from collections import Counter
import statistics
import hashlib
import json

app = Flask(__name__)

# Application version - read from version.py file (updated before each commit)
def get_app_version():
    """Get application version from version.py file (updated before each commit)"""
    try:
        # Try to import version from version.py file
        import version
        if hasattr(version, 'VERSION') and version.VERSION:
            return version.VERSION
    except ImportError:
        # version.py doesn't exist - fallback
        pass
    except Exception:
        # Error reading version.py - fallback
        pass
    
    # Fallback: try git commit hash (for local development)
    try:
        import subprocess
        git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], 
                                          stderr=subprocess.DEVNULL, timeout=2).decode('utf-8').strip()
        git_date = subprocess.check_output(['git', 'log', '-1', '--format=%cd', '--date=short'], 
                                          stderr=subprocess.DEVNULL, timeout=2).decode('utf-8').strip()
        if git_hash and git_date:
            return f"2.1.0-{git_hash} ({git_date})"
    except Exception:
        pass
    
    # Final fallback
    return "2.1.0"

APP_VERSION = get_app_version()

# Detect environment (local vs Railway)
def is_railway_deployment():
    """Detect if running on Railway or locally"""
    return os.environ.get('RAILWAY_ENVIRONMENT') is not None or os.environ.get('PORT') is not None

# Configure comprehensive logging system
import logging
import sys
import json
from datetime import datetime

# Create dedicated loggers for different purposes
def setup_comprehensive_logging():
    """Setup comprehensive logging system for user interactions and data processing"""
    
    # Create main application logger
    app_logger = logging.getLogger('dashboard_app')
    app_logger.setLevel(logging.INFO)
    app_logger.handlers.clear()
    
    # Create user interaction logger
    ui_logger = logging.getLogger('user_interactions')
    ui_logger.setLevel(logging.INFO)
    ui_logger.handlers.clear()
    
    # Create data processing logger
    data_logger = logging.getLogger('data_processing')
    data_logger.setLevel(logging.INFO)
    data_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Clear log files on restart to prevent them from becoming too large
    app_log_file = os.path.join(script_dir, 'dashboard_application.log')
    ui_log_file = os.path.join(script_dir, 'user_interactions.log')
    data_log_file = os.path.join(script_dir, 'data_processing.log')
    
    # Clear existing log files
    for log_file in [app_log_file, ui_log_file, data_log_file]:
        if os.path.exists(log_file):
            with open(log_file, 'w') as f:
                f.write('')  # Clear the file
    
    # Setup main application logger (for general app logs)
    app_handler = logging.FileHandler(app_log_file, mode='a')
    app_handler.setFormatter(formatter)
    app_logger.addHandler(app_handler)
    app_logger.propagate = False
    
    # Setup user interaction logger (for UI interactions)
    ui_handler = logging.FileHandler(ui_log_file, mode='a')
    ui_handler.setFormatter(formatter)
    ui_logger.addHandler(ui_handler)
    ui_logger.propagate = False
    
    # Setup data processing logger (for statistics and graph data)
    data_handler = logging.FileHandler(data_log_file, mode='a')
    data_handler.setFormatter(formatter)
    data_logger.addHandler(data_handler)
    data_logger.propagate = False
    
    return app_logger, ui_logger, data_logger

# Setup loggers
logger, ui_logger, data_logger = setup_comprehensive_logging()

# Test logging configuration
logger.info("🔧 MAIN LOGGER TEST - Application logging configured")
ui_logger.info("🔧 UI LOGGER TEST - User interaction logging configured")
data_logger.info("🔧 DATA LOGGER TEST - Data processing logging configured")

# Hello world test to confirm updated code is running
print("🚀 HELLO WORLD - UPDATED CODE IS RUNNING!")
logger.info("🚀 HELLO WORLD - UPDATED CODE IS RUNNING!")
data_logger.info("🚀 HELLO WORLD - UPDATED CODE IS RUNNING!")

# Log application startup
logger.info("🚀 APPLICATION STARTED - Log files cleared and ready for new session")
ui_logger.info("🚀 NEW SESSION STARTED - User interaction logging ready")
data_logger.info("🚀 NEW SESSION STARTED - Data processing logging ready")

print("🔧 LOGGING SYSTEM INITIALIZED - Log files cleared and ready for new session")

# Global variables to store processed data
processed_matches = []  # This will store the ORIGINAL data (never overwritten)
current_filtered_matches = []  # This will store the CURRENT filtered data for display
current_filters = {}
processing_status = {'progress': 0, 'current_pair': 0, 'total_pairs': 0, 'status': 'idle'}
cache_cleared_by_user = False  # Flag to prevent auto-loading after user clears cache

# Core algorithm configuration
MAX_FEATURES = 1000  # Default maximum number of features for ORB and SIFT detection

# ML model will be loaded when first needed
ml_model_loaded = False

# Cache directory
CACHE_DIR = 'cache'
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def calculate_image_properties(image_path):
    """Calculate contrast, brightness, and sharpness of an image"""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        # Brightness (mean pixel value)
        brightness = np.mean(img)
        
        # Contrast (standard deviation)
        contrast = np.std(img)
        
        # Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        sharpness = laplacian.var()
        
        return {
            'brightness': float(brightness),
            'contrast': float(contrast),
            'sharpness': float(sharpness)
        }
    except Exception as e:
        print(f"Error calculating image properties for {image_path}: {e}")
        return None

# ML Model Global Variables
ml_interpreter = None
ml_labels = {}

def load_ml_model():
    """Load the TensorFlow Lite ML model - tries EdgeTPU model first (if pycoral available), then fallback to regular model"""
    global ml_interpreter, ml_labels, ml_model_loaded
    
    # Lazy load TensorFlow
    if not _check_tensorflow() or not PIL_AVAILABLE:
        print("⚠️ Required dependencies not available for ML classification")
        ml_model_loaded = False
        return False
    
    tf = get_tf()  # Get TensorFlow module
    if tf is None:
        ml_model_loaded = False
        return False
    
    # Check if pycoral (EdgeTPU runtime) is available
    try:
        import pycoral
        EDGETPU_AVAILABLE = True
    except ImportError:
        EDGETPU_AVAILABLE = False
    
    # Try EdgeTPU model first (if available and pycoral is installed)
    edgetpu_model_path = "model_edgetpu.tflite"
    edgetpu_labels_path = "edgetpu_labels.txt"
    
    if os.path.exists(edgetpu_model_path) and os.path.exists(edgetpu_labels_path):
        if EDGETPU_AVAILABLE:
            try:
                print("🔄 Attempting to load EdgeTPU model with pycoral...")
                from pycoral.utils import edgetpu
                from pycoral.utils import dataset
                
                # Use EdgeTPU delegate for hardware acceleration (if available)
                # Falls back to CPU if no EdgeTPU device found
                try:
                    delegates = [edgetpu.load_edgetpu_delegate()]
                    print("✅ EdgeTPU device found, using hardware acceleration")
                except Exception:
                    delegates = None
                    print("ℹ️ No EdgeTPU device found, will use CPU")
                
                tf = get_tf()  # Get TensorFlow module
                ml_interpreter = tf.lite.Interpreter(
                    model_path=edgetpu_model_path,
                    experimental_delegates=delegates if delegates else None
                )
                ml_interpreter.allocate_tensors()
                
                # Load labels
                with open(edgetpu_labels_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split(' ', 1)
                            if len(parts) == 2:
                                class_id = int(parts[0])
                                class_name = parts[1]
                                ml_labels[class_id] = class_name
                
                print(f"✅ EdgeTPU ML model loaded successfully. Classes: {list(ml_labels.values())}")
                ml_model_loaded = True
                return True
            except Exception as e:
                print(f"⚠️ Failed to load EdgeTPU model with pycoral ({e})")
                print("   Note: EdgeTPU models require EdgeTPU hardware or pycoral runtime")
                print("   Trying fallback model...")
                ml_interpreter = None
                ml_labels = {}
        else:
            # Try EdgeTPU model without pycoral (will likely fail but worth trying)
            try:
                print("🔄 Attempting to load EdgeTPU model without pycoral (may not work)...")
                tf = get_tf()  # Get TensorFlow module
                ml_interpreter = tf.lite.Interpreter(model_path=edgetpu_model_path)
                ml_interpreter.allocate_tensors()
                
                # Load labels
                with open(edgetpu_labels_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split(' ', 1)
                            if len(parts) == 2:
                                class_id = int(parts[0])
                                class_name = parts[1]
                                ml_labels[class_id] = class_name
                
                print(f"✅ EdgeTPU ML model loaded (running on CPU). Classes: {list(ml_labels.values())}")
                ml_model_loaded = True
                return True
            except Exception as e:
                error_msg = str(e)
                if "edgetpu-custom-op" in error_msg or "unresolved custom op" in error_msg.lower():
                    print("❌ EdgeTPU model cannot run without EdgeTPU hardware or pycoral library")
                    print("   EdgeTPU models contain custom operations that require:")
                    print("   1. Physical EdgeTPU hardware (Coral USB Accelerator), OR")
                    print("   2. pycoral library (pip install pycoral)")
                    print("   Railway (cloud) does not have EdgeTPU hardware")
                    print("   Solution: Export a regular TensorFlow Lite model from Teachable Machine")
                    print("            (select 'TensorFlow Lite' not 'Edge TPU' when exporting)")
                else:
                    print(f"⚠️ Failed to load EdgeTPU model: {e}")
                print("   Falling back to regular model...")
                ml_interpreter = None
                ml_labels = {}
    
    # Fallback to regular TensorFlow Lite model
    try:
        model_path = "model_unquant.tflite"
        labels_path = "labels.txt"
        
        if not os.path.exists(model_path) or not os.path.exists(labels_path):
            print("⚠️ ML model files not found, ML classification will be disabled")
            ml_model_loaded = False
            return False
        
        print("🔄 Loading regular TensorFlow Lite model...")
        # Load model
        tf = get_tf()  # Get TensorFlow module
        ml_interpreter = tf.lite.Interpreter(model_path=model_path)
        ml_interpreter.allocate_tensors()
        
        # Load labels
        with open(labels_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        class_id = int(parts[0])
                        class_name = parts[1]
                        ml_labels[class_id] = class_name
        
        print(f"✅ Regular ML model loaded successfully. Classes: {list(ml_labels.values())}")
        ml_model_loaded = True
        return True
        
    except Exception as e:
        print(f"❌ Error loading ML model: {e}")
        ml_interpreter = None
        ml_labels = {}
        ml_model_loaded = False
        return False

def preprocess_image_for_ml(image_path, input_size=(224, 224)):
    """Preprocess image for ML model"""
    if not PIL_AVAILABLE:
        return None
        
    try:
        # Load and resize image
        image = PILImage.open(image_path).convert('RGB')
        image = image.resize(input_size)
        
        # Convert to numpy array and normalize
        image_array = np.array(image, dtype=np.float32)
        image_array = image_array / 255.0  # Normalize to [0, 1]
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        print(f"Error preprocessing image {image_path} for ML: {e}")
        return None

def classify_image_with_ml(image_path):
    """Classify image using ML model with timeout protection"""
    global ml_interpreter, ml_labels, ml_model_loaded

    # Lazy load TensorFlow
    if not _check_tensorflow() or not PIL_AVAILABLE:
        return None, 0.0

    # Load model if not already loaded
    if not ml_model_loaded:
        load_ml_model()

    if ml_interpreter is None or not ml_labels:
        return None, 0.0

    try:
        # Preprocess image
        image_array = preprocess_image_for_ml(image_path)
        if image_array is None:
            return None, 0.0

        # Get input and output details
        input_details = ml_interpreter.get_input_details()
        output_details = ml_interpreter.get_output_details()

        # Set input tensor
        ml_interpreter.set_tensor(input_details[0]['index'], image_array)

        # Run inference
        ml_interpreter.invoke()

        # Get output - copy the data to avoid reference issues
        output_data = ml_interpreter.get_tensor(output_details[0]['index']).copy()

        # Ensure we have valid output data
        if output_data is None or len(output_data) == 0:
            print(f"⚠️ Invalid output data for {image_path}")
            return None, 0.0

        # Get class with highest probability
        class_id = np.argmax(output_data[0])
        confidence = float(output_data[0][class_id])
        class_name_raw = ml_labels.get(class_id, f"Class_{class_id}")
        
        # Normalize class name to handle different label formats (GOOD/Not_Good/etc.)
        # Convert to standard format: "Good" or "Not_Good"
        class_name_normalized = class_name_raw.upper().strip()
        if class_name_normalized in ['GOOD', 'GOOD_']:
            class_name = 'Good'
        elif class_name_normalized in ['NOT_GOOD', 'NOT GOOD', 'BAD', 'NOTGOOD']:
            class_name = 'Not_Good'
        else:
            # Unknown class - return as is but log warning
            print(f"⚠️ Unknown class name from model: {class_name_raw} (normalized: {class_name_normalized})")
            class_name = class_name_raw

        print(f"🤖 ML Classification for {os.path.basename(image_path)}: {class_name} (confidence: {confidence:.3f}, raw: {class_name_raw})")
        return class_name, confidence

    except Exception as e:
        print(f"Error classifying image {image_path} with ML: {e}")
        return None, 0.0

def get_gps_coordinates_removed(image_path):
    """Extract GPS coordinates from image EXIF data"""
    try:
        with open(image_path, 'rb') as f:
            img = Image(f)
        
        if img.has_exif:
            lat = img.gps_latitude
            lon = img.gps_longitude
            lat_ref = getattr(img, 'gps_latitude_ref', 'N')
            lon_ref = getattr(img, 'gps_longitude_ref', 'E')
            
            if lat is not None and lon is not None:
                # Convert to decimal degrees if needed
                if isinstance(lat, tuple):
                    lat = lat[0] + lat[1]/60.0 + lat[2]/3600.0
                if isinstance(lon, tuple):
                    lon = lon[0] + lon[1]/60.0 + lon[2]/3600.0
                
                # Apply reference corrections
                if lat_ref == 'S':
                    lat = -lat
                if lon_ref == 'W':
                    lon = -lon
                
                return lat, lon
        
        return None, None
    except Exception as e:
        print(f"Error extracting GPS coordinates from {image_path}: {e}")
        return None, None

def calculate_gps_speed(lat1, lon1, lat2, lon2, time_diff):
    """Calculate speed between two GPS coordinates using ISS-specific haversine formula"""
    if time_diff <= 0:
        return None
    
    try:
        # Use the improved haversine_distance function (now ISS-specific)
        distance_km = haversine_distance(lat1, lon1, lat2, lon2)
        
        # Speed in km/s
        speed_km_s = distance_km / time_diff
        
        # Apply error correction for Earth radius variance (0.5% correction)
        haversine_error_correction_percentage = 0.05
        corrected_speed = speed_km_s * (1 + haversine_error_correction_percentage)
        
        return corrected_speed
        
    except Exception as e:
        print(f"Error calculating GPS speed: {e}")
        return None

def calculate_statistics(speeds, all_match_speeds=None):
    """Calculate mean, median, and mode of speeds"""
    if not speeds:
        return None
    
    # Round speeds to 1 decimal place for mode calculation
    rounded_speeds = [round(speed, 1) for speed in speeds]
    
    mean_speed = statistics.mean(speeds)
    median_speed = statistics.median(speeds)
    
    # Calculate mode for speeds (1 decimal place)
    counter = Counter(rounded_speeds)
    mode_speed = counter.most_common(1)[0][0] if counter else None
    
    result = {
        'mean': mean_speed,
        'median': median_speed,
        'mode': mode_speed,
        'count': len(speeds),
        'std_dev': statistics.stdev(speeds) if len(speeds) > 1 else 0
    }
    
    # Calculate mode for individual match speeds if provided
    if all_match_speeds:
        # Round individual match speeds to 1 decimal place
        rounded_match_speeds = [round(speed, 1) for speed in all_match_speeds]
        match_counter = Counter(rounded_match_speeds)
        match_mode_speed = match_counter.most_common(1)[0][0] if match_counter else None
        result['match_mode'] = match_mode_speed
        result['match_count'] = len(all_match_speeds)
    else:
        # If no separate match speeds provided, use the same data for both
        result['match_mode'] = mode_speed
        result['match_count'] = len(speeds)
    
    # Add pair_mode calculation (most common pair speed at one decimal place)
    # This will be calculated in the API endpoint using pair_speeds
    
    # GPS speed statistics removed - not allowed to use GPS location data
    
    return result

# GPS functions for enhanced GSD calculation
def extract_gps_from_image(image_path):
    """Extract GPS coordinates from EXIF data"""
    try:
        with open(image_path, 'rb') as image_file:
            img = Image(image_file)
            
            if img.has_exif:
                lat = img.get('gps_latitude')
                lon = img.get('gps_longitude')
                lat_ref = img.get('gps_latitude_ref')
                lon_ref = img.get('gps_longitude_ref')
                
                if lat and lon and lat_ref and lon_ref:
                    # Convert to decimal degrees
                    lat_decimal = convert_dms_to_decimal(lat, lat_ref)
                    lon_decimal = convert_dms_to_decimal(lon, lon_ref)
                    print(f"    GPS extracted: {lat_decimal:.6f}, {lon_decimal:.6f}")
                    return (lat_decimal, lon_decimal)
                else:
                    print(f"    GPS data incomplete: lat={lat}, lon={lon}, lat_ref={lat_ref}, lon_ref={lon_ref}")
            else:
                print(f"    No EXIF data found")
    except Exception as e:
        print(f"Error extracting GPS from {image_path}: {e}")
    return None

def convert_dms_to_decimal(dms_tuple, ref):
    """Convert DMS to decimal degrees"""
    degrees, minutes, seconds = dms_tuple
    decimal = degrees + minutes/60 + seconds/3600
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two GPS points in km - ISS-specific version"""
    # Use ISS-specific radius (Earth + ISS altitude)
    radius_of_earth = 6378.137  # More precise Earth radius in km
    mean_distance_from_iss_earth = 400  # ISS altitude in km
    radius_from_centre_of_earth_to_iss = radius_of_earth + mean_distance_from_iss_earth
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Calculate differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversine formula
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Distance using ISS orbital radius
    return c * radius_from_centre_of_earth_to_iss

def calculate_gps_gsd(image1_path, image2_path, avg_pixel_distance):
    """Calculate GSD using GPS coordinates and average pixel distance from keypoint matches"""
    gps1 = extract_gps_from_image(image1_path)
    gps2 = extract_gps_from_image(image2_path)
    
    if not gps1 or not gps2 or avg_pixel_distance <= 0:
        return None
    
    # Calculate real distance between GPS points
    lat1, lon1 = gps1
    lat2, lon2 = gps2
    real_distance_km = haversine_distance(lat1, lon1, lat2, lon2)
    
    # Calculate GSD: real distance / average pixel distance
    real_distance_m = real_distance_km * 1000  # Convert to meters
    gsd = real_distance_m / avg_pixel_distance  # meters per pixel
    
    return {
        'gsd': gsd,
        'real_distance_km': real_distance_km,
        'gps1': gps1,
        'gps2': gps2,
        'avg_pixel_distance': avg_pixel_distance
    }

def calculate_gps_only_speed(image1_path, image2_path):
    """Calculate speed using only GPS coordinates and time difference (no keypoint matching)"""
    print(f"    Extracting GPS from image 1: {os.path.basename(image1_path)}")
    gps1 = extract_gps_from_image(image1_path)
    print(f"    Extracting GPS from image 2: {os.path.basename(image2_path)}")
    gps2 = extract_gps_from_image(image2_path)
    
    if not gps1 or not gps2:
        print(f"    GPS extraction failed: gps1={gps1}, gps2={gps2}")
        return None
    
    # Calculate real distance between GPS points
    lat1, lon1 = gps1
    lat2, lon2 = gps2
    real_distance_km = haversine_distance(lat1, lon1, lat2, lon2)
    print(f"    GPS distance: {real_distance_km:.3f} km")
    
    # Calculate time difference
    time_difference = get_time_difference(image1_path, image2_path)
    if time_difference <= 0:
        print(f"    Invalid time difference: {time_difference}")
        return None
    
    # Calculate speed directly from GPS
    speed_kmps = real_distance_km / time_difference
    
    # Apply error correction for Earth radius variance (0.5% correction)
    haversine_error_correction_percentage = 0.05
    corrected_speed_kmps = speed_kmps * (1 + haversine_error_correction_percentage)
    
    print(f"    GPS-only speed (raw): {speed_kmps:.3f} km/s")
    print(f"    GPS-only speed (corrected): {corrected_speed_kmps:.3f} km/s")
    print(f"    GPS coordinates: {gps1[0]:.6f},{gps1[1]:.6f} -> {gps2[0]:.6f},{gps2[1]:.6f}")
    print(f"    GPS distance: {real_distance_km:.3f} km, time: {time_difference:.3f}s")
    
    return {
        'speed_kmps': corrected_speed_kmps,  # Use corrected speed
        'real_distance_km': real_distance_km,
        'time_difference': time_difference,
        'gps1': gps1,
        'gps2': gps2
    }

def check_folder_has_gps(folder_path):
    """Check if a folder contains images with GPS data"""
    try:
        image_files = [f for f in os.listdir(folder_path) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            return False
        
        # Check first few images for GPS data
        for i, image_file in enumerate(image_files[:5]):
            image_path = os.path.join(folder_path, image_file)
            gps = extract_gps_from_image(image_path)
            if gps:
                return True
        
        return False
    except:
        return False

# Global data storage
global_data = {
    'all_keypoints': [],
    'pair_results': [],
    'pair_characteristics': {},
    'photos_dir': None,
    'processed': False,
    'raw_keypoints': [],
    'pairs_to_keep': [],
    'processing': False,
    'status': 'Ready',
    'current_folder': None,
    'gps_enabled': False,
    'gps_gsd_data': {}
}

def get_cache_file_path(folder_path):
    """Get the cache file path for a given folder"""
    folder_name = os.path.basename(folder_path)
    return os.path.join(folder_path, f".{folder_name}_keypoints_cache.pkl")

def is_cache_valid(folder_path, cache_file):
    """Check if cache file exists and is newer than the images in the folder"""
    if not os.path.exists(cache_file):
        return False
    
    cache_time = os.path.getmtime(cache_file)
    
    # Check if any image file is newer than the cache
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            if os.path.getmtime(image_path) > cache_time:
                return False
    
    return True

def save_keypoints_cache(folder_path, raw_keypoints, pair_characteristics):
    """Save keypoints and pair characteristics to cache file"""
    try:
        cache_file = get_cache_file_path(folder_path)
        cache_data = {
            'raw_keypoints': raw_keypoints,
            'pair_characteristics': pair_characteristics,
            'timestamp': datetime.now().isoformat(),
            'folder_path': folder_path
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"💾 Cache saved: {cache_file}")
        return True
    except Exception as e:
        print(f"❌ Error saving cache: {e}")
        return False

def load_keypoints_cache(folder_path):
    """Load keypoints and pair characteristics from cache file"""
    try:
        cache_file = get_cache_file_path(folder_path)
        if not is_cache_valid(folder_path, cache_file):
            return None, None
        
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        print(f"⚡ Cache loaded: {cache_file}")
        print(f"📅 Cache timestamp: {cache_data.get('timestamp', 'Unknown')}")
        return cache_data['raw_keypoints'], cache_data['pair_characteristics']
    except Exception as e:
        print(f"❌ Error loading cache: {e}")
        return None, None

def clear_cache(folder_path):
    """Clear cache file for a given folder"""
    try:
        cache_file = get_cache_file_path(folder_path)
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"🗑️ Cache cleared: {cache_file}")
            return True
        return False
    except Exception as e:
        print(f"❌ Error clearing cache: {e}")
        return False

def generate_cache_key(folder_path, start_idx, end_idx, algorithm, use_flann, use_ransac_homography=False, ransac_threshold=5.0, ransac_min_matches=10, contrast_enhancement='clahe', max_features=1000):
    """Generate a unique cache key based on processing parameters"""
    # Create a string with all parameters
    params = {
        'folder': folder_path,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'algorithm': algorithm,
        'use_flann': use_flann,
        'use_ransac_homography': use_ransac_homography,
        'ransac_threshold': ransac_threshold,
        'ransac_min_matches': ransac_min_matches,
        'contrast_enhancement': contrast_enhancement,
        'max_features': max_features
    }
    
    # Convert to JSON string and create hash
    params_str = json.dumps(params, sort_keys=True)
    cache_key = hashlib.md5(params_str.encode()).hexdigest()
    return cache_key

def get_v2_cache_file_path(cache_key):
    """Get cache file path for v2 processing"""
    return os.path.join(CACHE_DIR, f'v2_cache_{cache_key}.pkl')

def is_v2_cache_valid(cache_key, folder_path):
    """Check if v2 cache is valid"""
    cache_file = get_v2_cache_file_path(cache_key)
    if not os.path.exists(cache_file):
        return False
    
    # Check if cache is newer than any image in the folder
    try:
        cache_time = os.path.getmtime(cache_file)
        image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
        
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            if os.path.getmtime(image_path) > cache_time:
                return False
        
        return True
    except:
        return False

def save_v2_cache(cache_key, matches_data):
    """Save v2 processing results to cache"""
    try:
        cache_file = get_v2_cache_file_path(cache_key)
        with open(cache_file, 'wb') as f:
            pickle.dump(matches_data, f)
        print(f"💾 Cache saved: {cache_file}")
        return True
    except Exception as e:
        print(f"❌ Error saving cache: {e}")
        return False

def load_v2_cache(cache_key):
    """Load v2 processing results from cache"""
    try:
        cache_file = get_v2_cache_file_path(cache_key)
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                matches_data = pickle.load(f)
            print(f"📂 Cache loaded: {cache_file}")
            return matches_data
        return None
    except Exception as e:
        print(f"❌ Error loading cache: {e}")
        return None

# Alias functions for test compatibility
def save_cache(cache_key, data):
    """Alias for save_v2_cache for test compatibility"""
    return save_v2_cache(cache_key, data)

def load_cache(cache_key):
    """Alias for load_v2_cache for test compatibility"""
    return load_v2_cache(cache_key)

def get_time(image):
    with open(image, 'rb') as image_file:
        img = Image(image_file)
        time_str = img.get("datetime_original")
        time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
        return time

def get_time_difference(image1_path, image2_path):
    """Get time difference between two images in seconds, rounded to 3 decimal places for consistency"""
    try:
        with open(image1_path, 'rb') as f:
            img1 = Image(f)
        with open(image2_path, 'rb') as f:
            img2 = Image(f)
        
        if img1.has_exif and img2.has_exif:
            datetime1_str = img1.datetime_original
            datetime2_str = img2.datetime_original
            
            if datetime1_str and datetime2_str:
                # Parse string datetime to datetime object
                datetime1 = datetime.strptime(datetime1_str, '%Y:%m:%d %H:%M:%S')
                datetime2 = datetime.strptime(datetime2_str, '%Y:%m:%d %H:%M:%S')
                # Round to 3 decimal places for consistency in speed calculations
                return round((datetime2 - datetime1).total_seconds(), 3)
        
        return 0.0
    except Exception as e:
        print(f"Error getting time difference: {e}")
        return 0.0

def enhance_image_contrast(image, method='clahe', clip_limit=3.0, tile_size=(8,8)):
    """Enhance image contrast for better feature detection"""
    if method == 'clahe':
        # --- SAFER CLAHE (same behaviour, more robust inputs) ---

        img = image

        # If accidentally given a colour image, convert to grayscale
        if img.ndim == 3 and img.shape[2] in (3, 4):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # CLAHE requires uint8
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img = img.astype(np.uint8)

        # Ensure tile size is valid
        tile_x = max(1, int(tile_size[0]))
        tile_y = max(1, int(tile_size[1]))

        clahe = cv2.createCLAHE(
            clipLimit=float(clip_limit),
            tileGridSize=(tile_x, tile_y)
        )
        enhanced = clahe.apply(img)

    elif method == 'histogram_eq':
        # Global histogram equalization
        enhanced = cv2.equalizeHist(image)

    elif method == 'gamma':
        # Gamma correction
        gamma = 1.5
        enhanced = np.power(image / 255.0, gamma) * 255.0
        enhanced = enhanced.astype(np.uint8)

    elif method == 'unsharp':
        # Unsharp masking
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        enhanced = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

    else:
        enhanced = image

    return enhanced

# GitHub Code Functions for Comparison
def github_get_time(image_path):
    """Get the time an image was taken (GitHub version)"""
    with open(image_path, 'rb') as image_file:
        img = Image(image_file)
        time_string = img.get("datetime_original")
        time = datetime.strptime(time_string, '%Y:%m:%d %H:%M:%S')
        return time

def github_get_time_difference(image_1_path, image_2_path):
    """Calculate time difference between two images (GitHub version)"""
    time_1 = github_get_time(image_1_path)
    time_2 = github_get_time(image_2_path)
    time_difference = time_2 - time_1
    return time_difference.total_seconds()

def github_convert_to_cv(image_1_path, image_2_path):
    """Convert images to OpenCV format (GitHub version)"""
    image_1_cv = cv2.imread(image_1_path, 0)  # 0 for grayscale
    image_2_cv = cv2.imread(image_2_path, 0)
    return image_1_cv, image_2_cv

def github_calculate_features(image_1_cv, image_2_cv, feature_number):
    """Calculate ORB features for both images (GitHub version)"""
    orb = cv2.ORB_create(nfeatures=feature_number)
    keypoints_1, descriptors_1 = orb.detectAndCompute(image_1_cv, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_2_cv, None)
    return keypoints_1, keypoints_2, descriptors_1, descriptors_2

def github_calculate_matches(descriptors_1, descriptors_2):
    """Calculate matches between descriptors using brute force (GitHub version)"""
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def github_find_matching_coordinates(keypoints_1, keypoints_2, matches):
    """Find coordinates of matching features (GitHub version)"""
    coordinates_1 = []
    coordinates_2 = []
    
    for match in matches:
        image_1_idx = match.queryIdx
        image_2_idx = match.trainIdx
        (x1, y1) = keypoints_1[image_1_idx].pt
        (x2, y2) = keypoints_2[image_2_idx].pt
        coordinates_1.append((x1, y1))
        coordinates_2.append((x2, y2))
    
    return coordinates_1, coordinates_2

def github_calculate_mean_distance(coordinates_1, coordinates_2):
    """Calculate average distance between matching coordinates (GitHub version)"""
    all_distances = []
    
    merged_coordinates = list(zip(coordinates_1, coordinates_2))
    
    for coordinate in merged_coordinates:
        x_difference = coordinate[0][0] - coordinate[1][0]
        y_difference = coordinate[0][1] - coordinate[1][1]
        distance = math.hypot(x_difference, y_difference)
        all_distances.append(distance)
    
    # Remove outliers using 2 standard deviations
    all_distances = github_remove_outliers(all_distances, 2)
    
    if len(all_distances) == 0:
        return 0
    
    return sum(all_distances) / len(all_distances)

def github_calculate_speed_in_kmps(feature_distance, GSD, time_difference):
    """Calculate speed in km/s (GitHub version)"""
    # GSD is in centimeters/pixels, convert to km
    distance = feature_distance * GSD / 100000  # Convert cm to km
    speed = distance / time_difference
    return speed

def github_remove_outliers(arr, n):
    """Remove outliers using standard deviation method (GitHub version)"""
    elements = np.array(arr)
    mean = np.mean(elements, axis=0)
    sd = np.std(elements, axis=0)
    
    final_list = [x for x in arr if (x > mean - n * sd)]
    final_list = [x for x in final_list if (x < mean + n * sd)]
    
    return final_list

def run_github_comparison(image_paths, max_features=MAX_FEATURES):
    """Run GitHub code comparison on a list of image paths"""
    print("🔄 Running GitHub code comparison...")
    
    total_speeds = []
    successful_pairs = 0
    failed_pairs = 0
    pair_details = []
    
    # Process consecutive image pairs
    for i in range(len(image_paths) - 1):
        try:
            image_1_path = image_paths[i]
            image_2_path = image_paths[i + 1]
            
            # Get time difference
            time_difference = github_get_time_difference(image_1_path, image_2_path)
            
            if time_difference <= 0:
                failed_pairs += 1
                continue
            
            # Convert to OpenCV format
            image_1_cv, image_2_cv = github_convert_to_cv(image_1_path, image_2_path)
            
            if image_1_cv is None or image_2_cv is None:
                failed_pairs += 1
                continue
            
            # Calculate features
            keypoints_1, keypoints_2, descriptors_1, descriptors_2 = github_calculate_features(
                image_1_cv, image_2_cv, max_features
            )
            
            if descriptors_1 is None or descriptors_2 is None:
                failed_pairs += 1
                continue
            
            # Calculate matches
            matches = github_calculate_matches(descriptors_1, descriptors_2)
            
            if len(matches) < 10:  # Need minimum matches
                failed_pairs += 1
                continue
            
            # Find matching coordinates
            coordinates_1, coordinates_2 = github_find_matching_coordinates(
                keypoints_1, keypoints_2, matches
            )
            
            # Calculate average feature distance
            average_feature_distance = github_calculate_mean_distance(coordinates_1, coordinates_2)
            
            if average_feature_distance == 0:
                failed_pairs += 1
                continue
            
            # Calculate speed using the same GSD as the original GitHub code
            speed = github_calculate_speed_in_kmps(average_feature_distance, 12648, time_difference)
            
            total_speeds.append(speed)
            successful_pairs += 1
            
            # Store pair details
            pair_details.append({
                'pair_index': i,
                'image1': os.path.basename(image_1_path),
                'image2': os.path.basename(image_2_path),
                'matches': len(matches),
                'distance': average_feature_distance,
                'speed': speed,
                'time_diff': time_difference
            })
            
        except Exception as e:
            print(f"❌ Error processing pair {i+1}: {e}")
            failed_pairs += 1
            continue
    
    # Calculate final results
    if len(total_speeds) == 0:
        return {
            'success': False,
            'error': 'No successful speed calculations',
            'total_speeds': 0,
            'successful_pairs': 0,
            'failed_pairs': failed_pairs
        }
    
    # Remove outliers from all speeds
    total_speeds_clean = github_remove_outliers(total_speeds, 1)
    
    if len(total_speeds_clean) == 0:
        final_speeds = total_speeds
    else:
        final_speeds = total_speeds_clean
    
    # Calculate statistics
    average_speed = sum(final_speeds) / len(final_speeds)
    min_speed = min(final_speeds)
    max_speed = max(final_speeds)
    std_speed = np.std(final_speeds)
    
    return {
        'success': True,
        'average_speed': average_speed,
        'min_speed': min_speed,
        'max_speed': max_speed,
        'std_speed': std_speed,
        'total_speeds': len(total_speeds),
        'final_speeds': len(final_speeds),
        'successful_pairs': successful_pairs,
        'failed_pairs': failed_pairs,
        'pair_details': pair_details,
        'target_speed': 7.66,
        'accuracy': abs(average_speed - 7.66)
    }

# GPS-based comparison functions (GitHub Project 2)
def gps_extract_coordinates_and_timestamp(image_path):
    """Extract GPS coordinates and timestamp from an image using its Exif data"""
    try:
        from PIL import Image
        # Open the image file
        img = Image.open(image_path)
        
        # Extract GPS Info from Exif data
        exif_data = img._getexif()
        
        # Check if GPSInfo and DateTimeOriginal exist in the Exif data
        if exif_data and 0x8825 in exif_data and 0x9003 in exif_data:
            gps_info = exif_data[0x8825]
            
            # Extract latitude and longitude
            latitude = gps_info[2][0] + gps_info[2][1] / 60 + gps_info[2][2] / 3600
            longitude = gps_info[4][0] + gps_info[4][1] / 60 + gps_info[4][2] / 3600
            
            # Check the hemisphere (N/S, E/W)
            if gps_info[3] == 'S':
                latitude = -latitude
            if gps_info[1] == 'W':
                longitude = -longitude
            
            # Extract timestamp and convert it to datetime
            timestamp_str = exif_data[0x9003]
            timestamp = datetime.strptime(timestamp_str, '%Y:%m:%d %H:%M:%S')
            
            return latitude, longitude, timestamp
        else:
            return None
    except Exception as e:
        print(f"Error extracting GPS data from {os.path.basename(image_path)}: {e}")
        return None

def gps_haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate haversine distance between two sets of latitude and longitude coordinates"""
    radius_of_earth = 6378.137
    mean_distance_from_iss_earth = 400
    # Earth's radius in kilometers added to the mean distance of the ISS from the Earth
    radius_from_centre_of_earth_to_iss = radius_of_earth + mean_distance_from_iss_earth
    
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Calculate differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversine formula
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Calculate distance
    distance = radius_from_centre_of_earth_to_iss * c
    return distance

def gps_calculate_speed(distance, time_difference):
    """Calculate speed given the distance between two points and the time difference"""
    # Convert time difference to seconds
    time_in_seconds = time_difference.total_seconds()
    
    # Calculate speed in kilometers per second
    speed = distance / time_in_seconds
    return abs(speed)  # Always positive

def run_gps_comparison(image_paths):
    """Run GPS-based comparison on a list of image paths"""
    print("🔄 Running GPS-based comparison...")
    
    speed_data = []
    successful_pairs = 0
    failed_pairs = 0
    pair_details = []
    
    # Process consecutive image pairs
    for i in range(len(image_paths) - 1):
        try:
            image_1_path = image_paths[i]
            image_2_path = image_paths[i + 1]
            
            # Extract GPS data from both images
            data1 = gps_extract_coordinates_and_timestamp(image_1_path)
            data2 = gps_extract_coordinates_and_timestamp(image_2_path)
            
            if data1 and data2:
                latitude1, longitude1, timestamp1 = data1
                latitude2, longitude2, timestamp2 = data2
                
                # Check if the images are the same or have zero distance or time
                if (image_1_path == image_2_path or 
                    gps_haversine_distance(latitude1, longitude1, latitude2, longitude2) == 0 or 
                    (timestamp2 - timestamp1).total_seconds() == 0):
                    failed_pairs += 1
                    continue
                
                # Calculate distance between the two points
                distance = gps_haversine_distance(latitude1, longitude1, latitude2, longitude2)
                
                # Calculate time difference
                time_difference = timestamp2 - timestamp1
                
                # Calculate speed
                speed = gps_calculate_speed(distance, time_difference)
                
                speed_data.append(speed)
                successful_pairs += 1
                
                # Store pair details
                pair_details.append({
                    'pair_index': i,
                    'image1': os.path.basename(image_1_path),
                    'image2': os.path.basename(image_2_path),
                    'distance': distance,
                    'speed': speed,
                    'time_diff': time_difference.total_seconds()
                })
                
            else:
                failed_pairs += 1
                
        except Exception as e:
            print(f"❌ Error processing pair {i+1}: {e}")
            failed_pairs += 1
            continue
    
    if not speed_data:
        return {
            'success': False,
            'error': 'No successful speed calculations',
            'total_speeds': 0,
            'successful_pairs': 0,
            'failed_pairs': failed_pairs
        }
    
    # Calculate statistics
    average_speed = sum(speed_data) / len(speed_data)
    min_speed = min(speed_data)
    max_speed = max(speed_data)
    std_speed = np.std(speed_data)
    
    # Apply the haversine error correction from the original code
    haversine_error_correction_percentage = 0.05  # 0.5% error correction
    error_corrected_average_speed = average_speed * (1 + haversine_error_correction_percentage)
    
    return {
        'success': True,
        'average_speed': error_corrected_average_speed,
        'min_speed': min_speed,
        'max_speed': max_speed,
        'std_speed': std_speed,
        'total_speeds': len(speed_data),
        'final_speeds': len(speed_data),
        'successful_pairs': successful_pairs,
        'failed_pairs': failed_pairs,
        'pair_details': pair_details,
        'target_speed': 7.66,
        'accuracy': abs(error_corrected_average_speed - 7.66),
        'error_correction_applied': haversine_error_correction_percentage
    }

def run_cchan083_comparison(image_paths, max_features=MAX_FEATURES):
    """Run cchan083/AstroPi SIFT-based comparison on a list of image paths"""
    print("🔄 Running cchan083/AstroPi SIFT comparison...")
    
    speed_data = []
    successful_pairs = 0
    failed_pairs = 0
    pair_details = []
    
    # Configuration from cchan083/AstroPi
    GSD = 12648  # Ground Sample Distance
    
    # Process consecutive image pairs
    for i in range(len(image_paths) - 1):
        try:
            image_1_path = image_paths[i]
            image_2_path = image_paths[i + 1]
            
            print(f"  Processing pair {i+1}: {os.path.basename(image_1_path)} -> {os.path.basename(image_2_path)}")
            
            # Calculate time difference
            time_delta = get_time_difference(image_1_path, image_2_path)
            if time_delta <= 0:
                print(f"    Invalid time difference: {time_delta}")
                failed_pairs += 1
                continue
            
            # Convert images to grayscale (as per cchan083 approach)
            img1_cv = cv2.cvtColor(cv2.imread(image_1_path), cv2.COLOR_BGR2GRAY)
            img2_cv = cv2.cvtColor(cv2.imread(image_2_path), cv2.COLOR_BGR2GRAY)
            
            # Detect SIFT features
            sift = cv2.SIFT_create(nfeatures=MAX_FEATURES)
            kp1, des1 = sift.detectAndCompute(img1_cv, None)
            kp2, des2 = sift.detectAndCompute(img2_cv, None)
            
            if des1 is None or des2 is None:
                print(f"    No features detected")
                failed_pairs += 1
                continue
            
            # Match features using brute force (as per cchan083 approach)
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            if len(matches) < 4:
                print(f"    Too few matches: {len(matches)}")
                failed_pairs += 1
                continue
            
            # Extract coordinates
            coords1 = [kp1[m.queryIdx].pt for m in matches]
            coords2 = [kp2[m.trainIdx].pt for m in matches]
            
            # Calculate mean distance
            total_distance = sum(math.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in zip(coords1, coords2))
            avg_dist = total_distance / len(coords1) if coords1 else 0
            
            # Calculate speed using cchan083 formula
            distance_km = (avg_dist * GSD) / 100000  # Convert cm to km
            speed = distance_km / time_delta  # km/s
            
            speed_data.append(speed)
            successful_pairs += 1
            
            print(f"    Matches: {len(matches)}, Avg distance: {avg_dist:.2f}px, Time: {time_delta:.1f}s, Speed: {speed:.4f} km/s")
            
            # Store pair details
            pair_details.append({
                'pair_index': i,
                'image1': os.path.basename(image_1_path),
                'image2': os.path.basename(image_2_path),
                'matches': len(matches),
                'avg_distance': avg_dist,
                'speed': speed,
                'time_diff': time_delta
            })
            
        except Exception as e:
            print(f"❌ Error processing pair {i+1}: {e}")
            failed_pairs += 1
            continue
    
    if not speed_data:
        return {
            'success': False,
            'error': 'No successful speed calculations',
            'total_speeds': 0,
            'successful_pairs': 0,
            'failed_pairs': failed_pairs
        }
    
    # Calculate statistics
    average_speed = sum(speed_data) / len(speed_data)
    min_speed = min(speed_data)
    max_speed = max(speed_data)
    std_speed = np.std(speed_data)
    
    return {
        'success': True,
        'average_speed': average_speed,
        'min_speed': min_speed,
        'max_speed': max_speed,
        'std_speed': std_speed,
        'total_speeds': len(speed_data),
        'final_speeds': len(speed_data),
        'successful_pairs': successful_pairs,
        'failed_pairs': failed_pairs,
        'pair_details': pair_details,
        'target_speed': 7.66,
        'accuracy': abs(average_speed - 7.66),
        'feature_count': MAX_FEATURES,
        'gsd_used': GSD
    }

def convert_to_cv(image_1, image_2, enhance_contrast=True, enhancement_method='clahe'):
    """Load and optionally enhance contrast of images for better feature detection"""
    image_1_cv = cv2.imread(image_1, 0)
    image_2_cv = cv2.imread(image_2, 0)
    
    if enhance_contrast and image_1_cv is not None and image_2_cv is not None:
        image_1_cv = enhance_image_contrast(image_1_cv, method=enhancement_method)
        image_2_cv = enhance_image_contrast(image_2_cv, method=enhancement_method)
    
    return image_1_cv, image_2_cv

def calculate_features(image_1, image_2, feature_number):
    orb = cv2.ORB_create(nfeatures = feature_number)
    keypoints_1, descriptors_1 = orb.detectAndCompute(image_1, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_2, None)
    return keypoints_1, keypoints_2, descriptors_1, descriptors_2

def calculate_matches(descriptors_1, descriptors_2):
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def find_matching_coordinates(keypoints_1, keypoints_2, matches):
    coordinates_1 = []
    coordinates_2 = []
    for match in matches:
        image_1_idx = match.queryIdx
        image_2_idx = match.trainIdx
        (x1,y1) = keypoints_1[image_1_idx].pt
        (x2,y2) = keypoints_2[image_2_idx].pt
        coordinates_1.append((x1,y1))
        coordinates_2.append((x2,y2))
    return coordinates_1, coordinates_2

def calculate_individual_distances(coordinates_1, coordinates_2):
    distances = []
    for coord1, coord2 in zip(coordinates_1, coordinates_2):
        x_difference = coord1[0] - coord2[0]
        y_difference = coord1[1] - coord2[1]
        distance = math.hypot(x_difference, y_difference)
        distances.append(distance)
    return distances

def calculate_speed_in_kmps(feature_distance, GSD, time_difference, is_gps_gsd=False):
    if is_gps_gsd:
        # GPS GSD is already in meters per pixel, convert to km
        distance = feature_distance * GSD / 1000  # Convert to km
    else:
        # ESA default GSD needs the /100000 factor
        distance = feature_distance * GSD / 100000
    speed = distance / time_difference
    return speed

def analyze_image_characteristics(image_path, clear_brightness_min=120, clear_contrast_min=50, 
                                cloudy_brightness_max=60, cloudy_contrast_max=15):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {'brightness': 0, 'contrast': 0, 'cloudiness': 'unknown'}
        
        brightness = np.mean(img)
        contrast = np.std(img)
        
        # More balanced 3-category system: clear, partly cloudy, mostly cloudy
        # Clear: brightness >= min_brightness AND contrast >= min_contrast
        if brightness >= clear_brightness_min and contrast >= clear_contrast_min:
            cloudiness = 'clear'
        # Mostly cloudy: brightness <= max_brightness OR contrast <= max_contrast
        elif brightness <= cloudy_brightness_max or contrast <= cloudy_contrast_max:
            cloudiness = 'mostly cloudy'
        else:
            cloudiness = 'partly cloudy'
        
        return {
            'brightness': brightness,
            'contrast': contrast,
            'cloudiness': cloudiness
        }
    except Exception as e:
        return {'brightness': 0, 'contrast': 0, 'cloudiness': 'unknown'}

def process_data(photos_dir, pair_percentile=10,  # Now represents minimum matches, not percentile
                clear_brightness_min=120, clear_contrast_min=50, 
                cloudy_brightness_max=60, cloudy_contrast_max=15,
                std_dev_multiplier=2.0,
                keypoint_percentile=5, match_quality_threshold=30, start_pair=1, end_pair=41):
    """Process all image pairs with configurable parameters"""
    try:
        print(f"🚀 Starting data processing for folder: {photos_dir}")
        
        # Check for GPS data availability
        gps_available = check_folder_has_gps(photos_dir)
        global_data['gps_enabled'] = gps_available
        
        if gps_available:
            print("🛰️ GPS data detected - will use dynamic GSD calculation")
        else:
            print("📸 No GPS data found - using default GSD values")
        
        # Check for cache first
        print(f"🔍 Checking for cache in: {photos_dir}")
        cached_keypoints, cached_characteristics = load_keypoints_cache(photos_dir)
        
        if cached_keypoints is not None and cached_characteristics is not None:
            print(f"⚡ Using cached data: {len(cached_keypoints)} keypoints")
            
            # Show progress bar even for cache loading
            global_data.update({
                'processing': True,
                'status': 'Loading from cache...',
                'progress': 0
            })
            
            # Simulate progress for cache loading (5 steps)
            for i in range(5):
                global_data['progress'] = (i + 1) * 20
                global_data['status'] = f'Loading from cache... {global_data["progress"]}%'
                time.sleep(0.1)  # Small delay to show progress
            
            # Reconstruct pair_results from cached data
            pair_results = []
            pair_count = 0
            for i, kp in enumerate(cached_keypoints):
                if kp['pair_num'] > pair_count:
                    pair_count = kp['pair_num']
                    if pair_count in cached_characteristics:
                        pair_speeds = [kp['speed'] for kp in cached_keypoints if kp['pair_num'] == pair_count]
                        pair_results.append({
                            'pair_num': pair_count,
                            'avg_speed': np.mean(pair_speeds),
                            'median_speed': np.median(pair_speeds),
                            'std_speed': np.std(pair_speeds),
                            'keypoint_count': len(pair_speeds),
                            'time_diff': kp['time_diff'],
                            'image1': kp.get('image1', ''),
                            'image2': kp.get('image2', ''),
                            'image1_path': kp.get('image1_path', ''),
                            'image2_path': kp.get('image2_path', '')
                        })
            
            # Store cached data
            global_data.update({
                'raw_keypoints': cached_keypoints,
                'pair_results': pair_results,
                'pair_characteristics': cached_characteristics,
                'photos_dir': photos_dir,
                'processed': True,
                'processing': False,
                'progress': 100,
                'status': f'Loaded {len(cached_keypoints)} keypoints from cache',
                'details': 'Data loaded from cache - ready for filtering'
            })
            
            return True
        
        print("🔄 Cache not found or invalid, processing images...")
        
        # Get list of image files
        image_files = sorted([f for f in os.listdir(photos_dir) if f.endswith('.jpg')])
        
        # Process images using sliding window approach
        all_keypoints = []
        pair_results = []
        pair_characteristics = {}
        pair_count = 0
        
        total_pairs = len(image_files) - 1  # Load ALL possible pairs
        
        for i in range(total_pairs):
            pair_count += 1
            image1_path = os.path.join(photos_dir, image_files[i])
            image2_path = os.path.join(photos_dir, image_files[i + 1])
            
            print(f"🔄 Processing pair {pair_count}/{total_pairs}: {os.path.basename(image1_path)} + {os.path.basename(image2_path)}")
            
            # Update progress
            progress = int((pair_count / total_pairs) * 80)  # 80% for processing pairs
            global_data['progress'] = progress
            global_data['details'] = f'Processing pair {pair_count}/{total_pairs}...'
            
            # Process with NO filtering to store all raw keypoints
            keypoint_data = process_image_pair(image1_path, image2_path, pair_count, 
                                             0.0, clear_brightness_min, 
                                             clear_contrast_min, cloudy_brightness_max, cloudy_contrast_max)
            all_keypoints.extend(keypoint_data)
            # Show speed summary for this pair
            if keypoint_data:
                dynamic_speeds = [kp['speed'] for kp in keypoint_data]
                gps_speeds = [kp['gps_only_speed'] for kp in keypoint_data if kp['gps_only_speed'] is not None]
                constant_speeds = [kp['constant_gsd_speed'] for kp in keypoint_data if kp['constant_gsd_speed'] is not None]
                
                print(f"  ✅ Pair {pair_count} completed: {len(keypoint_data)} keypoints")
                print(f"    Dynamic GSD avg: {np.mean(dynamic_speeds):.3f} km/s")
                if gps_speeds:
                    print(f"    GPS-only avg: {np.mean(gps_speeds):.3f} km/s")
                if constant_speeds:
                    print(f"    Constant GSD avg: {np.mean(constant_speeds):.3f} km/s")
                
                # Compare GPS-only vs Constant GSD accuracy
                if gps_speeds and constant_speeds:
                    gps_avg = np.mean(gps_speeds)
                    constant_avg = np.mean(constant_speeds)
                    expected_speed = 7.66  # Expected ISS speed
                    gps_error = abs(gps_avg - expected_speed) / expected_speed * 100
                    constant_error = abs(constant_avg - expected_speed) / expected_speed * 100
                    print(f"    Accuracy vs 7.66 km/s: GPS-only {gps_error:.1f}% error, Constant GSD {constant_error:.1f}% error")
            else:
                print(f"  ✅ Pair {pair_count} completed: 0 keypoints")
            
            if keypoint_data:
                pair_speeds = [kp['speed'] for kp in keypoint_data]
                pair_avg_speed = np.mean(pair_speeds)
                pair_median_speed = np.median(pair_speeds)
                pair_std_speed = np.std(pair_speeds)
                
                pair_results.append({
                    'pair_num': pair_count,
                    'avg_speed': pair_avg_speed,
                    'median_speed': pair_median_speed,
                    'std_speed': pair_std_speed,
                    'keypoint_count': len(keypoint_data),
                    'time_diff': keypoint_data[0]['time_diff'],
                    'image1': os.path.basename(image1_path),
                    'image2': os.path.basename(image2_path),
                    'image1_path': image1_path,
                    'image2_path': image2_path
                })
                
                pair_characteristics[pair_count] = {
                    'brightness': keypoint_data[0]['brightness'],
                    'contrast': keypoint_data[0]['contrast'],
                    'cloudiness': keypoint_data[0]['cloudiness']
                }
        
        # Debug: Show classification distribution
        clear_count = sum(1 for p in pair_characteristics.values() if p['cloudiness'] == 'clear')
        partly_cloudy_count = sum(1 for p in pair_characteristics.values() if p['cloudiness'] == 'partly cloudy')
        mostly_cloudy_count = sum(1 for p in pair_characteristics.values() if p['cloudiness'] == 'mostly cloudy')
        
        print(f"Debug: Raw data loaded - {len(all_keypoints)} total keypoints from {len(pair_results)} pairs")
        print(f"Debug: Classification distribution - Clear: {clear_count}, Partly Cloudy: {partly_cloudy_count}, Mostly Cloudy: {mostly_cloudy_count}")
        
        # Store ONLY raw data - no filtering applied during initial load
        print(f"✅ Processing complete! Storing {len(all_keypoints)} keypoints from {len(pair_results)} pairs")
        global_data.update({
            'raw_keypoints': all_keypoints,           # Complete unfiltered dataset
            'pair_results': pair_results,             # All pairs
            'pair_characteristics': pair_characteristics,
            'photos_dir': photos_dir,
            'processed': True,
            'progress': 100,
            'details': 'Raw data loaded - ready for filtering'
        })
        print("✅ Data stored in global_data, processed=True set")
        
        # Save to cache for future use
        print("💾 Saving processed data to cache...")
        save_keypoints_cache(photos_dir, all_keypoints, pair_characteristics)
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return False

def apply_filters_to_raw_data(enable_percentile=False, pair_percentile=10,  # Now represents minimum matches, not percentile
                             enable_std_dev=False, std_dev_multiplier=2.0,
                             enable_mad=False, mad_multiplier=3.0,
                             enable_keypoint_percentile=False, keypoint_percentile_bottom=5, keypoint_percentile_top=5,
                             enable_sequence=False, start_pair=1, end_pair=41,
                             enable_cloudiness=False, include_partly_cloudy=True, include_mostly_cloudy=True,
                             enable_gps_consistency=False, gps_tolerance_constant=0.783, filter_order=None):
    """Apply filters to the raw data and return filtered results"""
    if not global_data.get('processed') or not global_data.get('raw_keypoints'):
        return None
    
    raw_keypoints = global_data['raw_keypoints']
    pair_results = global_data['pair_results']
    pair_characteristics = global_data['pair_characteristics']
    
    print(f"Debug: Applying filters to {len(raw_keypoints)} raw keypoints")
    
    # Start with all raw keypoints
    filtered_keypoints = raw_keypoints.copy()
    
    # Default filter order if not provided
    if filter_order is None:
        filter_order = ['sequence', 'keypoint_percentile', 'percentile', 'std_dev', 'mad', 'cloudiness', 'gps_consistency']
    
    print(f"Debug: Applying filters in order: {filter_order}")
    
    # Apply filters in the specified order
    for step, filter_type in enumerate(filter_order, 1):
        if filter_type == 'sequence' and enable_sequence:
            # Apply sequence range filter (filter by pair numbers)
            filtered_keypoints = [kp for kp in filtered_keypoints if start_pair <= kp['pair_num'] <= end_pair]
            print(f"Debug: Step {step} - After sequence filter (pairs {start_pair}-{end_pair}): {len(filtered_keypoints)} keypoints")
    
        elif filter_type == 'gps_consistency' and enable_gps_consistency and global_data.get('gps_enabled', False):
            # Apply GPS consistency filter - remove keypoints with Constant GSD speeds outside GPS-only speed range
            print(f"Debug: Step {step} - GPS consistency filter ENABLED - GPS enabled: {global_data.get('gps_enabled', False)}")
            if len(filtered_keypoints) > 0:
                # Get all GPS-only speeds for the current filtered keypoints
                gps_only_speeds = [kp['gps_only_speed'] for kp in filtered_keypoints if kp.get('gps_only_speed') is not None]
                
                print(f"Debug: Step {step} - GPS consistency filter - Found {len(gps_only_speeds)} GPS-only speeds from {len(filtered_keypoints)} keypoints")
                
                if gps_only_speeds:
                    # Calculate GPS-only speed range with tolerance
                    min_gps_speed = min(gps_only_speeds) - gps_tolerance_constant
                    max_gps_speed = max(gps_only_speeds) + gps_tolerance_constant
                    
                    print(f"Debug: Step {step} - GPS consistency filter - GPS speed range: {min_gps_speed:.3f} to {max_gps_speed:.3f} km/s (tolerance: ±{gps_tolerance_constant:.3f})")
                    print(f"Debug: Step {step} - Before GPS consistency filter: {len(filtered_keypoints)} keypoints")
                    
                    # Count keypoints that will be removed
                    removed_count = 0
                    kept_count = 0
                    
                    # Filter out keypoints where Constant GSD speed is outside GPS range (with tolerance)
                    new_filtered_keypoints = []
                    for kp in filtered_keypoints:
                        if kp.get('constant_gsd_speed') is not None:
                            if min_gps_speed <= kp['constant_gsd_speed'] <= max_gps_speed:
                                new_filtered_keypoints.append(kp)
                                kept_count += 1
                            else:
                                removed_count += 1
                        else:
                            # Keep keypoints without constant_gsd_speed (shouldn't happen with GPS data)
                            new_filtered_keypoints.append(kp)
                            kept_count += 1
                    
                    filtered_keypoints = new_filtered_keypoints
                    
                    print(f"Debug: Step {step} - After GPS consistency filter: {len(filtered_keypoints)} keypoints (removed: {removed_count}, kept: {kept_count})")
                    
                    # Show some examples of removed keypoints
                    if removed_count > 0:
                        print(f"Debug: Step {step} - Example removed keypoints (first 3):")
                        # Get examples from the original list before filtering
                        original_keypoints = [kp for kp in raw_keypoints if kp.get('constant_gsd_speed') is not None and not (min_gps_speed <= kp['constant_gsd_speed'] <= max_gps_speed)][:3]
                        for i, kp in enumerate(original_keypoints):
                            print(f"  {i+1}. Constant GSD: {kp['constant_gsd_speed']:.3f} km/s, GPS-only: {kp.get('gps_only_speed', 'N/A'):.3f} km/s")
                else:
                    print(f"Debug: Step {step} - GPS consistency filter - No GPS-only speeds available, skipping filter")
            else:
                print(f"Debug: Step {step} - GPS consistency filter - No keypoints to filter")
        elif filter_type == 'gps_consistency' and enable_gps_consistency and not global_data.get('gps_enabled', False):
            print(f"Debug: Step {step} - GPS consistency filter ENABLED but GPS not available - skipping filter")
        elif filter_type == 'gps_consistency' and not enable_gps_consistency:
            print(f"Debug: Step {step} - GPS consistency filter DISABLED - skipping filter")
        elif filter_type == 'keypoint_percentile' and enable_keypoint_percentile and len(filtered_keypoints) > 0:
            # Apply keypoint percentile filtering (remove bottom X% and top Y% by speed)
            speeds = [kp['speed'] for kp in filtered_keypoints]
            
            # Calculate thresholds for bottom and top percentiles separately
            bottom_threshold = np.percentile(speeds, keypoint_percentile_bottom)
            top_threshold = np.percentile(speeds, 100 - keypoint_percentile_top)
            
            print(f"Debug: Step {step} - Keypoint percentile filter - removing bottom {keypoint_percentile_bottom}% and top {keypoint_percentile_top}%")
            print(f"Debug: Step {step} - Bottom threshold: {bottom_threshold:.3f} km/s, Top threshold: {top_threshold:.3f} km/s")
            print(f"Debug: Step {step} - Before keypoint filtering: {len(filtered_keypoints)} keypoints")
            
            # Keep only keypoints within the percentile range
            filtered_keypoints = [kp for kp in filtered_keypoints 
                                if bottom_threshold <= kp['speed'] <= top_threshold]
            
            print(f"Debug: Step {step} - After keypoint filtering: {len(filtered_keypoints)} keypoints")
            
        elif filter_type == 'percentile' and enable_percentile:
            # Apply minimum matches filtering to pairs
            pair_counts = {}
            for kp in filtered_keypoints:
                pair_num = kp['pair_num']
                pair_counts[pair_num] = pair_counts.get(pair_num, 0) + 1
            
            if pair_counts:
                # Use minimum match count instead of percentile
                min_matches = pair_percentile  # Now represents minimum matches, not percentile
                pairs_to_keep = [pair_num for pair_num, count in pair_counts.items() if count >= min_matches]
                
                # Safety check: ensure we keep at least 3 pairs or 50% of pairs, whichever is smaller
                min_pairs_to_keep = min(3, max(1, len(pair_counts) // 2))
                if len(pairs_to_keep) < min_pairs_to_keep and len(pair_counts) >= min_pairs_to_keep:
                    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
                    pairs_to_keep = [pair_num for pair_num, count in sorted_pairs[:min_pairs_to_keep]]
                    print(f"Debug: Step {step} - Safety check activated - keeping top {min_pairs_to_keep} pairs instead")
                
                print(f"Debug: Step {step} - Pair filtering - min_matches={min_matches}, pairs_before={len(pair_counts)}, pairs_after={len(pairs_to_keep)}")
                
                # Filter keypoints to only include pairs above threshold
                filtered_keypoints = [kp for kp in filtered_keypoints if kp['pair_num'] in pairs_to_keep]
            else:
                print(f"Debug: Step {step} - No pair counts available, using all {len(filtered_keypoints)} keypoints")
                
        elif filter_type == 'std_dev' and enable_std_dev and len(filtered_keypoints) > 0 and std_dev_multiplier > 0:
            # Apply standard deviation filter (outlier removal)
            speeds = [kp['speed'] for kp in filtered_keypoints]
            mean_speed = np.mean(speeds)
            std_speed = np.std(speeds)
            
            # Calculate the range
            min_allowed = mean_speed - (std_dev_multiplier * std_speed)
            max_allowed = mean_speed + (std_dev_multiplier * std_speed)
            
            print(f"Debug: Step {step} - BEFORE std dev filter - {len(filtered_keypoints)} keypoints")
            print(f"Debug: Step {step} - Speed range before: {min(speeds):.3f} to {max(speeds):.3f}")
            print(f"Debug: Step {step} - Mean: {mean_speed:.3f}, Std: {std_speed:.3f}")
            print(f"Debug: Step {step} - Allowed range: {min_allowed:.3f} to {max_allowed:.3f}")
            
            # Keep only speeds within std_dev_multiplier standard deviations
            filtered_keypoints = [kp for kp in filtered_keypoints 
                                 if abs(kp['speed'] - mean_speed) <= std_dev_multiplier * std_speed]
            
            if filtered_keypoints:
                filtered_speeds = [kp['speed'] for kp in filtered_keypoints]
                print(f"Debug: Step {step} - AFTER std dev filter - {len(filtered_keypoints)} keypoints")
                print(f"Debug: Step {step} - Speed range after: {min(filtered_speeds):.3f} to {max(filtered_speeds):.3f}")
            else:
                print(f"Debug: Step {step} - AFTER std dev filter - NO KEYPOINTS REMAIN!")
            
            print(f"Debug: Step {step} - Std dev filter ({std_dev_multiplier}σ) completed")
            
        elif filter_type == 'mad' and enable_mad and len(filtered_keypoints) > 0 and mad_multiplier > 0:
            # Apply MAD (Median Absolute Deviation) filter (robust outlier removal)
            speeds = [kp['speed'] for kp in filtered_keypoints]
            median_speed = np.median(speeds)
            
            # Calculate MAD: median of absolute deviations from median
            absolute_deviations = [abs(speed - median_speed) for speed in speeds]
            mad = np.median(absolute_deviations)
            
            # Calculate the range
            min_allowed = median_speed - (mad_multiplier * mad)
            max_allowed = median_speed + (mad_multiplier * mad)
            
            print(f"Debug: Step {step} - BEFORE MAD filter - {len(filtered_keypoints)} keypoints")
            print(f"Debug: Step {step} - Speed range before: {min(speeds):.3f} to {max(speeds):.3f}")
            print(f"Debug: Step {step} - Median: {median_speed:.3f}, MAD: {mad:.3f}")
            print(f"Debug: Step {step} - Allowed range: {min_allowed:.3f} to {max_allowed:.3f}")
            
            # Keep only speeds within mad_multiplier * MAD from median
            filtered_keypoints = [kp for kp in filtered_keypoints 
                                 if abs(kp['speed'] - median_speed) <= mad_multiplier * mad]
            
            if filtered_keypoints:
                filtered_speeds = [kp['speed'] for kp in filtered_keypoints]
                print(f"Debug: Step {step} - AFTER MAD filter - {len(filtered_keypoints)} keypoints")
                print(f"Debug: Step {step} - Speed range after: {min(filtered_speeds):.3f} to {max(filtered_speeds):.3f}")
            else:
                print(f"Debug: Step {step} - AFTER MAD filter - NO KEYPOINTS REMAIN!")
            
            print(f"Debug: Step {step} - MAD filter ({mad_multiplier}*MAD) completed")
            
        elif filter_type == 'cloudiness' and enable_cloudiness:
            # Apply cloudiness filter
            cloudiness_filtered = []
            for kp in filtered_keypoints:
                pair_num = kp['pair_num']
                if pair_num in pair_characteristics:
                    cloudiness = pair_characteristics[pair_num]['cloudiness']
                    if cloudiness == 'clear':
                        cloudiness_filtered.append(kp)
                    elif cloudiness == 'partly cloudy' and include_partly_cloudy:
                        cloudiness_filtered.append(kp)
                    elif cloudiness == 'mostly cloudy' and include_mostly_cloudy:
                        cloudiness_filtered.append(kp)
            
            filtered_keypoints = cloudiness_filtered
            print(f"Debug: Step {step} - After cloudiness filter (partly: {include_partly_cloudy}, mostly: {include_mostly_cloudy}): {len(filtered_keypoints)} keypoints")
    
    final_keypoints = filtered_keypoints
    
    # Update pair characteristics with existing classifications
    for kp in final_keypoints:
        kp['cloudiness'] = pair_characteristics.get(kp['pair_num'], {}).get('cloudiness', 'unknown')
    
    return {
        'keypoints': final_keypoints,
        'pair_characteristics': pair_characteristics,
        'pairs_to_keep': pairs_to_keep if 'pairs_to_keep' in locals() else []
    }

def process_image_pair(image1_path, image2_path, pair_num, stationary_threshold, 
                      clear_brightness_min, clear_contrast_min, cloudy_brightness_max, cloudy_contrast_max):
    """Process a single image pair"""
    try:
        # Get time difference
        time_difference = get_time_difference(image1_path, image2_path)
        
        # Analyze image characteristics
        img1_chars = analyze_image_characteristics(image1_path, clear_brightness_min, clear_contrast_min, 
                                                 cloudy_brightness_max, cloudy_contrast_max)
        img2_chars = analyze_image_characteristics(image2_path, clear_brightness_min, clear_contrast_min, 
                                                 cloudy_brightness_max, cloudy_contrast_max)
        
        avg_brightness = (img1_chars['brightness'] + img2_chars['brightness']) / 2
        avg_contrast = (img1_chars['contrast'] + img2_chars['contrast']) / 2
        
        if img1_chars['cloudiness'] == 'clear' and img2_chars['cloudiness'] == 'clear':
            overall_cloudiness = 'clear'
        elif img1_chars['cloudiness'] == 'mostly cloudy' or img2_chars['cloudiness'] == 'mostly cloudy':
            overall_cloudiness = 'mostly cloudy'
        else:
            overall_cloudiness = 'partly cloudy'
        
        # Convert to OpenCV format
        image_1_cv, image_2_cv = convert_to_cv(image1_path, image2_path)
        
        # Calculate features
        keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(image_1_cv, image_2_cv, MAX_FEATURES)
        
        # Calculate matches
        matches = calculate_matches(descriptors_1, descriptors_2)
        
        if len(matches) == 0:
            return []
        
        # Find matching coordinates
        coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, matches)
        
        # Calculate individual distances
        distances = calculate_individual_distances(coordinates_1, coordinates_2)
        
        # Calculate individual speeds
        keypoint_data = []
        
        # Calculate average pixel distance for GPS GSD calculation
        avg_pixel_distance = sum(distances) / len(distances) if distances else 0
        print(f"Pair {pair_num}: Average pixel distance: {avg_pixel_distance:.2f} pixels")
        
        # Use ESA default GSD only (GPS functionality disabled)
        gsd = 12648  # ESA default GSD
        gps_info = None
        print(f"Using ESA default GSD: {gsd} m/px for pair {pair_num}")
        
        # GPS-only speed calculation disabled
        gps_only_speed = None
        
        # Constant GSD speed calculation disabled (using main ESA default approach)
        constant_gsd_speed = None
        
        for i, distance in enumerate(distances):
            # Always use ESA default GSD (GPS disabled)
            speed = calculate_speed_in_kmps(distance, gsd, time_difference, is_gps_gsd=False)
            if i < 3:  # Debug first 3 keypoints
                print(f"  Keypoint {i}: distance={distance:.2f}px, gsd={gsd}m/px, time={time_difference:.3f}s")
                print(f"    ESA GSD speed: {speed:.3f}km/s")
            keypoint_data.append({
                'pair_num': pair_num,
                'keypoint_idx': i,
                'distance': distance,
                'speed': speed,  # ESA GSD speed
                'gsd_used': gsd,
                'time_diff': time_difference,
                'image1': os.path.basename(image1_path),
                'image2': os.path.basename(image2_path),
                'brightness': avg_brightness,
                'contrast': avg_contrast,
                'cloudiness': overall_cloudiness,
                'image1_path': image1_path,
                'image2_path': image2_path
            })
        
        # Filter out stationary objects
        filtered_keypoint_data = []
        for kp in keypoint_data:
            if kp['speed'] > stationary_threshold:
                filtered_keypoint_data.append(kp)
        
        # Add GPS-only speed to each keypoint for reference
        for kp in filtered_keypoint_data:
            kp['gps_only_speed'] = gps_only_speed
        
        return filtered_keypoint_data
        
    except Exception as e:
        return []

def get_current_cloudiness_color(pair_num, pair_characteristics, current_filters):
    """Get the current cloudiness color based on current filter settings"""
    if not current_filters.get('enable_cloudiness', False):
        # If cloudiness filter is not enabled, use original cloudiness
        return pair_characteristics.get(pair_num, {}).get('cloudiness', 'cloudy')
    
    # Get the pair characteristics
    pair_chars = pair_characteristics.get(pair_num, {})
    if not pair_chars:
        return 'cloudy'
    
    # Get current filter thresholds
    clear_brightness_min = current_filters.get('clear_brightness_min', 120)
    clear_contrast_min = current_filters.get('clear_contrast_min', 50)
    cloudy_brightness_max = current_filters.get('cloudy_brightness_max', 60)
    cloudy_contrast_max = current_filters.get('cloudy_contrast_max', 15)
    
    # Get brightness and contrast from pair characteristics
    brightness = pair_chars.get('brightness', 0)
    contrast = pair_chars.get('contrast', 0)
    
    # Recalculate cloudiness based on current thresholds
    if brightness >= clear_brightness_min and contrast >= clear_contrast_min:
        return 'clear'
    elif brightness <= cloudy_brightness_max or contrast <= cloudy_contrast_max:
        return 'mostly_cloudy'
    else:
        return 'partly_cloudy'

def create_plot_data(speed_mode='dynamic_gsd'):
    """Create plot data for the dashboard"""
    if not global_data['processed']:
        return None
    
    all_keypoints = global_data['raw_keypoints']
    pair_results = global_data['pair_results']
    pair_characteristics = global_data['pair_characteristics']
    
    # Select speeds based on mode
    if speed_mode == 'gps_only':
        speeds = [kp['gps_only_speed'] for kp in all_keypoints if kp.get('gps_only_speed') is not None]
    elif speed_mode == 'constant_gsd':
        speeds = [kp['constant_gsd_speed'] for kp in all_keypoints if kp.get('constant_gsd_speed') is not None]
    else:  # dynamic_gsd or default
        speeds = [kp['speed'] for kp in all_keypoints]
    
    # Calculate three different speed methods
    # 1. GPS-only method - direct GPS calculation
    gps_only_speeds = [kp['gps_only_speed'] for kp in all_keypoints if kp.get('gps_only_speed') is not None]
    gps_only_avg = np.mean(gps_only_speeds) if gps_only_speeds else 0
    
    # 2. Constant GSD method - ESA default GSD
    constant_gsd_speeds = [kp['constant_gsd_speed'] for kp in all_keypoints if kp.get('constant_gsd_speed') is not None]
    constant_gsd_avg = np.mean(constant_gsd_speeds) if constant_gsd_speeds else 0
    
    # 3. Dynamic GSD method - GPS GSD + keypoint matching (current method)
    dynamic_gsd_speeds = [kp['speed'] for kp in all_keypoints]
    dynamic_gsd_avg = np.mean(dynamic_gsd_speeds) if dynamic_gsd_speeds else 0
    
    # Overall average (all methods combined)
    overall_avg = np.mean(speeds) if speeds else 0
    
    # Prepare data for plots (convert numpy arrays to lists for JSON serialization)
    plot_data = {
        'speedometers': {
            'gps_only': {
                'speed': float(gps_only_avg),
                'count': len(gps_only_speeds),
                'label': 'GPS-Only'
            },
            'constant_gsd': {
                'speed': float(constant_gsd_avg),
                'count': len(constant_gsd_speeds),
                'label': 'Constant GSD (ESA)'
            },
            'dynamic_gsd': {
                'speed': float(dynamic_gsd_avg),
                'count': len(dynamic_gsd_speeds),
                'label': 'Dynamic GSD (GPS+Keypoints)'
            },
            'overall': {
                'speed': float(overall_avg),
                'count': len(speeds),
                'label': 'Overall Average'
            }
        },
        'histogram': {
            'speeds': speeds.tolist() if isinstance(speeds, np.ndarray) else speeds,
            'mean': float(np.mean(speeds)),
            'median': float(np.median(speeds)),
            'std': float(np.std(speeds))
        },
        'boxplot': {
            'speeds': speeds.tolist() if isinstance(speeds, np.ndarray) else speeds
        },
        'pairs': {
            'pairs': [p['pair_num'] for p in pair_results],
            'original_pairs': [p['pair_num'] for p in pair_results],  # Add original_pairs for compatibility
            'means': [p['avg_speed'] for p in pair_results],
            'medians': [p.get('median_speed', p['avg_speed']) for p in pair_results],
            'stds': [p['std_speed'] for p in pair_results],
            'colors': []
        },
        'cumulative': {
            'sorted_speeds': sorted(speeds),
            'cumulative': (np.arange(1, len(speeds) + 1) / len(speeds)).tolist()
        }
    }
    
    # Add colors for pairs
    for pair_num in plot_data['pairs']['pairs']:
        if pair_num in pair_characteristics:
            cloudiness = pair_characteristics[pair_num]['cloudiness']
            if cloudiness == 'clear':
                plot_data['pairs']['colors'].append('green')
            elif cloudiness == 'partly cloudy':
                plot_data['pairs']['colors'].append('orange')
            elif cloudiness == 'mostly cloudy':
                plot_data['pairs']['colors'].append('red')
            else:
                plot_data['pairs']['colors'].append('gray')
        else:
            plot_data['pairs']['colors'].append('gray')
    
    return plot_data

def create_plot_data_from_filtered(filtered_data, speed_mode='dynamic_gsd'):
    """Create plot data from filtered results"""
    keypoints = filtered_data['keypoints']
    pair_characteristics = filtered_data['pair_characteristics']
    pairs_to_keep = filtered_data['pairs_to_keep']
    
    if not keypoints:
        return None
    
    # Select speeds based on mode
    if speed_mode == 'gps_only':
        speeds = [kp['gps_only_speed'] for kp in keypoints if kp.get('gps_only_speed') is not None]
    elif speed_mode == 'constant_gsd':
        speeds = [kp['constant_gsd_speed'] for kp in keypoints if kp.get('constant_gsd_speed') is not None]
    else:  # dynamic_gsd or default
        speeds = [kp['speed'] for kp in keypoints]
    
    # If no speeds available for selected mode, fall back to dynamic_gsd
    if not speeds and speed_mode != 'dynamic_gsd':
        speeds = [kp['speed'] for kp in keypoints]
        speed_mode = 'dynamic_gsd'  # Update mode to reflect fallback
    
    # Group pairs by their characteristics for the pairs plot
    pair_groups = {}
    for kp in keypoints:
        pair_num = kp['pair_num']
        if pair_num not in pair_groups:
            pair_groups[pair_num] = {
                'speeds': [],
                'cloudiness': kp.get('cloudiness', 'unknown')
            }
        
        # Use the appropriate speed based on mode
        if speed_mode == 'gps_only' and kp.get('gps_only_speed') is not None:
            pair_groups[pair_num]['speeds'].append(kp['gps_only_speed'])
        elif speed_mode == 'constant_gsd' and kp.get('constant_gsd_speed') is not None:
            pair_groups[pair_num]['speeds'].append(kp['constant_gsd_speed'])
        else:  # dynamic_gsd or default
            pair_groups[pair_num]['speeds'].append(kp['speed'])
    
    # Calculate pair statistics
    pair_data = []
    # Sort pair numbers to create a continuous sequence
    sorted_pairs = sorted(pair_groups.keys())
    for i, pair_num in enumerate(sorted_pairs):
        group = pair_groups[pair_num]
        pair_speeds = group['speeds']
        pair_data.append({
            'pair_num': i + 1,  # Remap to continuous sequence starting from 1
            'original_pair_num': pair_num,  # Keep original for reference
            'avg_speed': np.mean(pair_speeds),
            'median_speed': np.median(pair_speeds),
            'std_speed': np.std(pair_speeds),
            'cloudiness': group['cloudiness']
        })
    
    # Calculate all three speed methods for speedometers
    gps_only_speeds = [kp['gps_only_speed'] for kp in keypoints if kp.get('gps_only_speed') is not None]
    constant_gsd_speeds = [kp['constant_gsd_speed'] for kp in keypoints if kp.get('constant_gsd_speed') is not None]
    dynamic_gsd_speeds = [kp['speed'] for kp in keypoints]

    gps_only_avg = np.mean(gps_only_speeds) if gps_only_speeds else 0
    constant_gsd_avg = np.mean(constant_gsd_speeds) if constant_gsd_speeds else 0
    dynamic_gsd_avg = np.mean(dynamic_gsd_speeds) if dynamic_gsd_speeds else 0
    overall_avg = np.mean(speeds) if speeds else 0
    
    # Debug output for GPS consistency filter
    print(f"Debug: Speedometer calculations - Total keypoints: {len(keypoints)}")
    print(f"Debug: GPS-only speeds: {len(gps_only_speeds)} keypoints, avg: {gps_only_avg:.3f} km/s")
    print(f"Debug: Constant GSD speeds: {len(constant_gsd_speeds)} keypoints, avg: {constant_gsd_avg:.3f} km/s")
    print(f"Debug: Dynamic GSD speeds: {len(dynamic_gsd_speeds)} keypoints, avg: {dynamic_gsd_avg:.3f} km/s")
    
    # Prepare data for plots
    plot_data = {
        'speedometers': {
            'gps_only': {
                'speed': float(gps_only_avg),
                'count': len(gps_only_speeds),
                'label': 'GPS-Only'
            },
            'constant_gsd': {
                'speed': float(constant_gsd_avg),
                'count': len(constant_gsd_speeds),
                'label': 'Constant GSD (ESA)'
            },
            'dynamic_gsd': {
                'speed': float(dynamic_gsd_avg),
                'count': len(dynamic_gsd_speeds),
                'label': 'Dynamic GSD (GPS+Keypoints)'
            },
            'overall': {
                'speed': float(overall_avg),
                'count': len(speeds),
                'label': 'Overall Average'
            }
        },
        'histogram': {
            'speeds': speeds,
            'mean': float(np.mean(speeds)),
            'median': float(np.median(speeds)),
            'std': float(np.std(speeds))
        },
        'pairs': {
            'pairs': [p['pair_num'] for p in pair_data],
            'original_pairs': [p['original_pair_num'] for p in pair_data],
            'means': [p['avg_speed'] for p in pair_data],
            'medians': [p['median_speed'] for p in pair_data],
            'stds': [p['std_speed'] for p in pair_data],
            'colors': []
        }
    }
    
    # Add colors for pairs
    for pair_info in pair_data:
        cloudiness = pair_info['cloudiness']
        if cloudiness == 'clear':
            plot_data['pairs']['colors'].append('green')
        elif cloudiness == 'partly cloudy':
            plot_data['pairs']['colors'].append('orange')
        elif cloudiness == 'mostly cloudy':
            plot_data['pairs']['colors'].append('red')
        else:
            plot_data['pairs']['colors'].append('gray')
    
    return plot_data

@app.route('/')
def index():
    return render_template('dashboard_v2_clean.html')

@app.route('/favicon.ico')
def favicon():
    """Serve favicon icon"""
    try:
        return send_file('static/favicon.ico', mimetype='image/vnd.microsoft.icon')
    except FileNotFoundError:
        # Fallback if favicon doesn't exist
        return '', 204

@app.route('/api/version')
def get_version():
    """Get application version"""
    return jsonify({
        "version": APP_VERSION,
        "application": "AstroPi Explorer Dashboard"
    })

@app.route('/health')
def health():
    """Simple health check endpoint for Railway"""
    try:
        # Basic health check - just return success
        return jsonify({
            "status": "healthy", 
            "message": "AstroPi Explorer Dashboard is running",
            "version": APP_VERSION,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Health check failed: {str(e)}"
        }), 500

# New API endpoints for improved workflow

@app.route('/api/process-range', methods=['POST'])
def process_range():
    """Process a range of image pairs with selected algorithms"""
    global processed_matches, processing_status, current_filtered_matches, current_filters
    
    try:
        # Log user interaction
        ui_logger.info("📂 DATA LOADING API CALL - User requested data processing")
        ui_logger.info(f"📂 Processing request: {json.dumps(request.json, indent=2)}")
        
        # Log data processing
        data_logger.info("📂 DATA LOADING STARTED")
        data_logger.info(f"📂 Processing parameters: {json.dumps(request.json, indent=2)}")
        data = request.json
        folder_name = data.get('folder')
        start_idx = data.get('start_idx', 0)
        end_idx = data.get('end_idx', 0)
        algorithm = data.get('algorithm', 'ORB')  # ORB or SIFT
        use_flann = data.get('use_flann', False)
        use_ransac_homography = data.get('use_ransac_homography', False)
        ransac_threshold = data.get('ransac_threshold', 5.0)
        ransac_min_matches = data.get('ransac_min_matches', 10)
        contrast_enhancement = data.get('contrast_enhancement', 'clahe')
        max_features = data.get('max_features', MAX_FEATURES)
        
        if not folder_name:
            return jsonify({'error': 'Folder name required'}), 400
        
        photos_dir = os.path.join(os.getcwd(), folder_name)
        if not os.path.exists(photos_dir):
            return jsonify({'error': 'Folder not found'}), 404
        
        # Generate cache key for this specific processing request
        cache_key = generate_cache_key(photos_dir, start_idx, end_idx, algorithm, use_flann, use_ransac_homography, 
                                     ransac_threshold, ransac_min_matches, contrast_enhancement, max_features)
        
        print(f"🔑 Cache key generated: {cache_key[:8]}...")
        print(f"📋 Processing parameters: {algorithm}, FLANN: {use_flann}, RANSAC/Homography: {use_ransac_homography}, Contrast: {contrast_enhancement}")
        if use_ransac_homography:
            print(f"🎯 RANSAC/Homography params: threshold={ransac_threshold}, min_matches={ransac_min_matches}")
        
        # Check if we have valid cached data
        if is_v2_cache_valid(cache_key, photos_dir):
            print(f"🚀 Loading from cache for key: {cache_key[:8]}...")
            cached_data = load_v2_cache(cache_key)
            if cached_data:
                processed_matches = cached_data
                # Reset filters and filtered data when loading from cache to ensure fresh state
                current_filters = {}
                current_filtered_matches = []
                print(f"🔄 Reset filters and cleared filtered data when loading from cache - current_filters: {current_filters}, current_filtered_matches cleared")
                processing_status = {
                    'progress': 100,
                    'current_pair': len(processed_matches),
                    'total_pairs': len(processed_matches),
                    'status': 'completed'
                }
                return jsonify({
                    'success': True,
                    'message': f'Loaded {len(processed_matches)} matches from cache',
                    'from_cache': True
                })
        
        # Get image files
        image_files = sorted([f for f in os.listdir(photos_dir) if f.endswith('.jpg')])
        
        if end_idx >= len(image_files):
            end_idx = len(image_files) - 1
        
        if start_idx >= end_idx:
            return jsonify({'error': 'Invalid range'}), 400
        
        # Initialize processing status
        total_pairs = end_idx - start_idx
        processing_status.update({
            'progress': 0,
            'current_pair': 0,
            'total_pairs': total_pairs,
            'status': 'processing'
        })
        print(f"📊 Initialized processing status: {processing_status}")
        
        print(f"🔄 Processing {total_pairs} pairs with {algorithm} (FLANN: {use_flann}, RANSAC/Homography: {use_ransac_homography}, Contrast: {contrast_enhancement})")
        if use_ransac_homography:
            print(f"🎯 RANSAC/Homography settings: threshold={ransac_threshold}, min_matches={ransac_min_matches}")
        
        # Process in background thread to allow progress polling
        def process_range_thread():
            global processed_matches, processing_status, current_filtered_matches, current_filters, cache_cleared_by_user
            
            try:
                # Set the enhancement method for the process_image_pair function
                process_image_pair.enhancement_method = contrast_enhancement
                
                # Process each pair in the range
                cache_cleared_by_user = False  # Reset flag when processing new data
                processed_matches = []
                # Clear filtered data when processing new images to avoid showing stale data
                current_filtered_matches = []
                current_filters = {}
                print(f"🔄 Cleared filtered data and filters for new image processing")
                
                for i in range(start_idx, end_idx):
                    # Update progress
                    current_pair = i - start_idx + 1
                    progress = (current_pair / total_pairs) * 100
                    
                    # Update status atomically
                    processing_status['progress'] = progress
                    processing_status['current_pair'] = current_pair
                    processing_status['status'] = 'processing'
                    
                    # Log progress for debugging (especially on Railway)
                    if current_pair % max(1, total_pairs // 10) == 0 or current_pair == 1:
                        print(f"📊 Processing progress: {progress:.1f}% - pair {current_pair}/{total_pairs}")
                    
                    image1_path = os.path.join(photos_dir, image_files[i])
                    image2_path = os.path.join(photos_dir, image_files[i + 1])
                    
                    # Calculate image properties
                    img1_props = calculate_image_properties(image1_path)
                    img2_props = calculate_image_properties(image2_path)
                    
                    # Calculate cloudiness classification for this pair
                    pair_cloudiness = None
                    if img1_props and img2_props:
                        avg_brightness = (img1_props['brightness'] + img2_props['brightness']) / 2
                        avg_contrast = (img1_props['contrast'] + img2_props['contrast']) / 2
                        
                        # Use the updated cloudiness classification logic
                        if avg_brightness >= 120 and avg_contrast >= 55:
                            pair_cloudiness = 'clear'
                        elif avg_brightness <= 60 or avg_contrast <= 40:
                            pair_cloudiness = 'mostly cloudy'
                    
                    # Process the pair
                    matches = process_image_pair(image1_path, image2_path, algorithm, use_flann, use_ransac_homography, 
                                               ransac_threshold, ransac_min_matches, max_features)
                    
                    # Calculate time difference for this pair
                    time_diff = get_time_difference(image1_path, image2_path)
                    
                    # Add metadata to each match
                    for match in matches:
                        match['pair_index'] = i
                        match['image1_name'] = image_files[i]
                        match['image2_name'] = image_files[i + 1]
                        match['image1_path'] = image1_path
                        match['image2_path'] = image2_path
                        match['algorithm'] = algorithm
                        match['use_flann'] = use_flann
                        match['use_ransac_homography'] = use_ransac_homography
                        match['ransac_threshold'] = ransac_threshold
                        match['ransac_min_matches'] = ransac_min_matches
                        match['time_difference'] = time_diff
                        match['image1_properties'] = img1_props
                        match['image2_properties'] = img2_props
                        match['cloudiness'] = pair_cloudiness
                        
                        # ML classification will be applied later when the filter is enabled
                        # This avoids the performance hit during initial data loading
                        match['ml_classification1'] = None
                        match['ml_confidence1'] = 0.0
                        match['ml_classification2'] = None
                        match['ml_confidence2'] = 0.0
                        match['ml_classification'] = None
                        match['ml_confidence'] = 0.0
                    
                    processed_matches.extend(matches)
                
                # Mark as completed
                processing_status['progress'] = 100
                processing_status['current_pair'] = total_pairs
                processing_status['status'] = 'completed'
                print(f"✅ Processing status set to completed: {processing_status}")
                
                # Print summary
                print(f"📊 Processing complete: {len(processed_matches)} total matches from {total_pairs} pairs")
                if use_ransac_homography:
                    print(f"🎯 RANSAC/Homography filtering was applied with threshold={ransac_threshold}")
                else:
                    print(f"ℹ️ No RANSAC/Homography filtering applied")
                
                # Summary of processing
                if processed_matches:
                    print(f"📊 Processing complete: {len(processed_matches)} total matches from {total_pairs} pairs")
                
                # Save results to cache
                print(f"💾 Saving {len(processed_matches)} matches to cache...")
                save_v2_cache(cache_key, processed_matches)
                
                # Log data loading completion
                data_logger.info("📂 DATA LOADING COMPLETED")
                data_logger.info(f"📂 Final results: {len(processed_matches)} matches from {end_idx - start_idx} pairs")
                data_logger.info(f"📂 Processing parameters used: {json.dumps(data, indent=2)}")
                
            except Exception as e:
                print(f"❌ Error in background processing thread: {e}")
                import traceback
                traceback.print_exc()
                processing_status.update({
                    'status': 'error',
                    'error': str(e)
                })
        
        # Start processing in background thread
        thread = threading.Thread(target=process_range_thread)
        thread.daemon = True
        thread.start()
        
        # Return immediately so frontend can poll for progress
        return jsonify({
            'success': True,
            'status': 'processing',
            'message': 'Processing started',
            'total_pairs': total_pairs,
            'from_cache': False
        })
        
    except Exception as e:
        processing_status['status'] = 'error'
        return jsonify({'error': str(e)}), 500

@app.route('/api/statistics')
def get_statistics():
    """Get statistics for current matches"""
    global processed_matches, current_filters, current_filtered_matches
    
    # Determine which data to use: filtered data if available, otherwise original data
    data_to_use = current_filtered_matches if current_filtered_matches else processed_matches
    data_source = "filtered" if current_filtered_matches else "original"
    
    # Log user interaction
    ui_logger.info("📊 STATISTICS API CALL - User requested Section 3 statistics")
    ui_logger.info(f"📊 Using {data_source} data: {len(data_to_use) if data_to_use else 0} matches")
    ui_logger.info(f"📊 Current filters applied: {json.dumps(current_filters, indent=2)}")
    
    # Log data processing details
    data_logger.info("📊 SECTION 3 STATISTICS CALCULATION STARTED")
    data_logger.info(f"📊 Using {data_source} data: {len(data_to_use) if data_to_use else 0} matches")
    data_logger.info(f"📊 Original data: {len(processed_matches) if processed_matches else 0} matches")
    data_logger.info(f"📊 Filtered data: {len(current_filtered_matches) if current_filtered_matches else 0} matches")
    data_logger.info(f"📊 Filters applied: {json.dumps(current_filters, indent=2)}")
    
    print(f"🔍 Statistics endpoint called - using {data_source} data: {len(data_to_use) if data_to_use else 0} matches")
    
    if not data_to_use:
        logger.info("📊 No processed matches - returning default values")
        print("🔍 No processed matches - returning default values")
        # Return empty statistics instead of error to prevent frontend crashes
        default_stats = {
            'mean': 0,
            'median': 0,
            'mode': 0,
            'count': 0,
            'std_dev': 0,
            'match_mode': 0,
            'match_count': 0,
            'pair_mode': 0
        }
        logger.info(f"📊 Returning default statistics: {default_stats}")
        return jsonify(default_stats)
    
    # Use the appropriate data source (already determined above)
    filtered_matches = data_to_use
    print(f"🔍 Using {data_source} data for statistics: {len(filtered_matches)} matches")
    
    # Calculate pair speeds (average speed per pair) and individual match speeds
    pair_speeds = []
    all_match_speeds = []
    
    # Group matches by pair_index to calculate pair speeds
    pair_groups = {}
    for match in filtered_matches:
        pair_idx = match['pair_index']
        if pair_idx not in pair_groups:
            pair_groups[pair_idx] = []
        pair_groups[pair_idx].append(match)
        all_match_speeds.append(match['speed'])
    
    # Calculate average speed for each pair
    for pair_idx, matches in pair_groups.items():
        if matches:
            avg_speed = sum(match['speed'] for match in matches) / len(matches)
            pair_speeds.append(avg_speed)
    
    # Use individual match speeds for consistency with histogram (Section 5)
    # This ensures Section 3 and Section 5 show the same median values
    # ALWAYS use individual keypoint speeds, never pair averages
    stats = calculate_statistics(all_match_speeds)
    
    # Fix: Set the correct count for pairs vs matches
    stats['count'] = len(pair_speeds)  # Number of pairs
    stats['match_count'] = len(all_match_speeds)  # Number of individual matches
    
    logger.info(f"📊 Statistics calculated - pairs: {len(pair_speeds)}, matches: {len(all_match_speeds)}")
    logger.info(f"📊 Raw statistics: {stats}")
    
    # Calculate pair_mode (most common pair speed at one decimal place)
    # If all values are unique, use the one closest to the mean
    print(f"🔍 Pair mode calculation debug:")
    print(f"🔍   pair_speeds count: {len(pair_speeds)}")
    print(f"🔍   pair_speeds values: {pair_speeds[:5] if pair_speeds else 'None'}")
    
    if pair_speeds:
        from collections import Counter
        pair_speeds_rounded = [round(speed, 1) for speed in pair_speeds]
        pair_counter = Counter(pair_speeds_rounded)
        most_common = pair_counter.most_common(1)
        
        # Check if the most common value appears more than once
        if most_common and most_common[0][1] > 1:
            # There is a truly most common value
            stats['pair_mode'] = most_common[0][0]
            print(f"🔍   Found most common value: {stats['pair_mode']} (appears {most_common[0][1]} times)")
        else:
            # All values are unique, find the one closest to the mean
            pair_mean = sum(pair_speeds) / len(pair_speeds)
            closest_to_mean = min(pair_speeds_rounded, key=lambda x: abs(x - pair_mean))
            stats['pair_mode'] = closest_to_mean
            print(f"🔍   All values unique, using closest to mean: {stats['pair_mode']} (mean: {pair_mean:.2f})")
        
        print(f"🔍   pair_speeds_rounded: {pair_speeds_rounded}")
        print(f"🔍   pair_counter: {dict(pair_counter)}")
        print(f"🔍   most_common: {most_common}")
        print(f"🔍   final pair_mode: {stats['pair_mode']}")
    else:
        stats['pair_mode'] = None
        print(f"🔍   No pair_speeds available - setting pair_mode to None")
    
    # Debug: Print the actual values being calculated
    print(f"🔍 Section 3 Statistics Debug:")
    print(f"🔍   Total matches used: {len(all_match_speeds)}")
    print(f"🔍   Mean: {stats['mean']:.3f} km/s")
    print(f"🔍   Median: {stats['median']:.3f} km/s")
    print(f"🔍   Speed range: {min(all_match_speeds):.3f} - {max(all_match_speeds):.3f} km/s")
    
    # Log detailed statistics calculation results
    data_logger.info("📊 SECTION 3 STATISTICS CALCULATION COMPLETED")
    data_logger.info(f"📊 Final statistics returned: {json.dumps(stats, indent=2)}")
    data_logger.info(f"📊 Speed range: {min(all_match_speeds):.3f} - {max(all_match_speeds):.3f} km/s")
    data_logger.info(f"📊 Total matches used: {len(all_match_speeds)}")
    data_logger.info(f"📊 Total pairs: {len(pair_speeds)}")
    
    logger.info(f"📊 FINAL STATISTICS RETURNED: {stats}")
    logger.info(f"📊 Speed range: {min(all_match_speeds):.3f} - {max(all_match_speeds):.3f} km/s")
    
    return jsonify(stats)

@app.route('/api/algorithm-comparison', methods=['POST'])
def run_algorithm_comparison_api():
    """Run multiple algorithm comparisons on the currently loaded images"""
    global processed_matches
    
    if not processed_matches:
        return jsonify({'error': 'No data processed'})
    
    try:
        data = request.json
        algorithms = data.get('algorithms', ['github_cv', 'gps'])  # Default to both methods
        max_features = data.get('max_features', MAX_FEATURES)
        
        # Extract unique image paths from processed matches
        image_paths = []
        seen_paths = set()
        
        for match in processed_matches:
            img1_path = match.get('image1_path')
            img2_path = match.get('image2_path')
            
            if img1_path and img1_path not in seen_paths:
                image_paths.append(img1_path)
                seen_paths.add(img1_path)
            
            if img2_path and img2_path not in seen_paths:
                image_paths.append(img2_path)
                seen_paths.add(img2_path)
        
        # Sort image paths to maintain order
        image_paths.sort()
        
        print(f"🔄 Running algorithm comparisons on {len(image_paths)} images...")
        print(f"📋 Algorithms requested: {algorithms}")
        
        results = {}
        
        # Run GitHub Computer Vision comparison
        if 'github_cv' in algorithms:
            print("🔄 Running GitHub Computer Vision comparison...")
            github_results = run_github_comparison(image_paths, max_features)
            if github_results['success']:
                github_results['method'] = 'GitHub Computer Vision (diyasmenon/astropi)'
                github_results['description'] = 'Basic ORB feature detection with simple outlier removal'
                results['github_cv'] = github_results
                print(f"✅ GitHub CV completed: {github_results['average_speed']:.4f} km/s")
            else:
                results['github_cv'] = {
                    'success': False,
                    'error': github_results.get('error', 'Unknown error'),
                    'method': 'GitHub Computer Vision (diyasmenon/astropi)'
                }
                print(f"❌ GitHub CV failed: {github_results.get('error', 'Unknown error')}")
        
        # Run GPS-based comparison
        if 'gps' in algorithms:
            print("🔄 Running GPS-based comparison...")
            gps_results = run_gps_comparison(image_paths)
            if gps_results['success']:
                gps_results['method'] = 'GPS-based (Cossack42/AstroPI)'
                gps_results['description'] = 'GPS coordinates with haversine formula and ISS altitude correction'
                results['gps'] = gps_results
                print(f"✅ GPS comparison completed: {gps_results['average_speed']:.4f} km/s")
            else:
                results['gps'] = {
                    'success': False,
                    'error': gps_results.get('error', 'Unknown error'),
                    'method': 'GPS-based (Cossack42/AstroPI)'
                }
                print(f"❌ GPS comparison failed: {gps_results.get('error', 'Unknown error')}")
        
        # Run cchan083/AstroPi SIFT comparison
        if 'cchan083' in algorithms:
            print("🔄 Running cchan083/AstroPi SIFT comparison...")
            cchan083_results = run_cchan083_comparison(image_paths, max_features)
            if cchan083_results['success']:
                cchan083_results['method'] = 'SIFT-based (cchan083/AstroPi)'
                cchan083_results['description'] = 'SIFT feature detection with grayscale processing'
                results['cchan083'] = cchan083_results
                print(f"✅ cchan083 comparison completed: {cchan083_results['average_speed']:.4f} km/s")
            else:
                results['cchan083'] = {
                    'success': False,
                    'error': cchan083_results.get('error', 'Unknown error'),
                    'method': 'SIFT-based (cchan083/AstroPi)'
                }
                print(f"❌ cchan083 comparison failed: {cchan083_results.get('error', 'Unknown error')}")
        
        return jsonify({
            'success': True,
            'results': results,
            'total_images': len(image_paths),
            'algorithms_run': list(results.keys())
        })
            
    except Exception as e:
        print(f"❌ Error running algorithm comparisons: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/apply-filters', methods=['POST'])
def apply_filters():
    """Apply filters to the processed matches"""
    global processed_matches, current_filters
    
    try:
        data = request.json
        current_filters = data
        
        # Log user interaction
        ui_logger.info("🔄 APPLY FILTERS API CALL - User applied filters")
        ui_logger.info(f"🔄 Filters received: {json.dumps(data, indent=2)}")
        ui_logger.info(f"🔄 Current processed_matches count: {len(processed_matches) if processed_matches else 0}")
        
        # Log data processing
        data_logger.info("🔄 FILTER APPLICATION STARTED")
        data_logger.info(f"🔄 Input filters: {json.dumps(data, indent=2)}")
        data_logger.info(f"🔄 Input data: {len(processed_matches) if processed_matches else 0} matches")
        
        logger.info(f"🔄 APPLY FILTERS API CALL - received filters: {data}")
        logger.info(f"🔄 Current processed_matches count: {len(processed_matches) if processed_matches else 0}")
        
        # Store original count before filtering
        original_count = len(processed_matches)
        
        # Apply filters to processed_matches (ORIGINAL DATA - never overwritten)
        data_logger.info(f"🔄 PRESERVING ORIGINAL DATA: {len(processed_matches)} matches (never overwritten)")
        data_logger.info(f"🔄 ABOUT TO APPLY FILTERS: current_filters = {current_filters}")
        filtered_matches = apply_match_filters(processed_matches, current_filters)
        data_logger.info(f"🔄 FILTERED DATA: {len(filtered_matches)} matches (temporary, not stored)")
        data_logger.info(f"🔄 FILTERING COMPLETE: {len(filtered_matches)}/{len(processed_matches)} matches remaining")
        
        # Apply custom GSD if enabled
        if current_filters.get('enable_custom_gsd', False):
            custom_gsd = current_filters.get('custom_gsd', 12648)
            print(f"🛰️ Applying custom GSD: {custom_gsd} cm/pixel")
            logger.info(f"🛰️ APPLYING CUSTOM GSD: {custom_gsd} cm/pixel to {len(filtered_matches)} matches")
            
            # Recalculate speeds with custom GSD
            gsd_recalculated_count = 0
            for match in filtered_matches:
                if 'pixel_distance' in match and 'time_difference' in match:
                    # Recalculate speed using custom GSD
                    old_speed = match['speed']
                    new_speed = calculate_speed_in_kmps(match['pixel_distance'], custom_gsd, match['time_difference'])
                    match['speed'] = new_speed
                    match['gsd_used'] = custom_gsd
                    if 'original_speed' not in match:
                        match['original_speed'] = old_speed
                    gsd_recalculated_count += 1
            
            logger.info(f"🛰️ GSD RECALCULATION COMPLETE: {gsd_recalculated_count} matches recalculated")
        
        # Store filtered results separately - NEVER overwrite original data
        # The original processed_matches must always remain unchanged
        data_logger.info(f"🔄 PRESERVING ORIGINAL DATA: processed_matches count = {len(processed_matches)} (NEVER CHANGED)")
        data_logger.info(f"🔄 FILTERED DATA READY: filtered_matches count = {len(filtered_matches)} (for display only)")
        
        # Store current filtered data for API endpoints to use
        # This will be used by /api/statistics and /api/plot-data
        global current_filtered_matches
        
        # If no filters are applied, clear the filtered data to use original data
        if not current_filters or all(not v for v in current_filters.values() if isinstance(v, bool)):
            current_filtered_matches = []
            data_logger.info(f"🔄 NO FILTERS APPLIED: cleared current_filtered_matches (will use original data)")
        else:
            current_filtered_matches = filtered_matches
            data_logger.info(f"🔄 FILTERS APPLIED: stored current_filtered_matches = {len(current_filtered_matches)} matches")
        
        # Calculate statistics for filtered matches
        # Calculate pair speeds (average speed per pair) and collect all match speeds
        pair_speeds = []
        all_match_speeds = []
        
        # Group filtered matches by pair_index to calculate pair speeds
        pair_groups = {}
        for match in filtered_matches:
            pair_idx = match['pair_index']
            if pair_idx not in pair_groups:
                pair_groups[pair_idx] = []
            pair_groups[pair_idx].append(match)
            all_match_speeds.append(match['speed'])
        
        # Calculate average speed for each pair
        for pair_idx, matches in pair_groups.items():
            if matches:
                avg_speed = sum(match['speed'] for match in matches) / len(matches)
                pair_speeds.append(avg_speed)
        
        # Use the same calculation method as /api/statistics endpoint
        stats = calculate_statistics(all_match_speeds)
        
        # Fix: Set the correct count for pairs vs matches (same as /api/statistics)
        stats['count'] = len(pair_speeds)  # Number of pairs
        stats['match_count'] = len(all_match_speeds)  # Number of individual matches
        
        # Calculate pair_mode (most common pair speed at one decimal place)
        # If all values are unique, use the one closest to the mean
        if pair_speeds:
            from collections import Counter
            pair_speeds_rounded = [round(speed, 1) for speed in pair_speeds]
            pair_counter = Counter(pair_speeds_rounded)
            most_common = pair_counter.most_common(1)
            
            # Check if the most common value appears more than once
            if most_common and most_common[0][1] > 1:
                # There is a truly most common value
                stats['pair_mode'] = most_common[0][0]
            else:
                # All values are unique, find the one closest to the mean
                pair_mean = sum(pair_speeds) / len(pair_speeds)
                closest_to_mean = min(pair_speeds_rounded, key=lambda x: abs(x - pair_mean))
                stats['pair_mode'] = closest_to_mean
        else:
            stats['pair_mode'] = None
        
        result = {
            'success': True,
            'filtered_count': len(filtered_matches),
            'original_count': original_count,
            'statistics': stats
        }
        
        # Log filter application results
        data_logger.info("🔄 FILTER APPLICATION COMPLETED")
        data_logger.info(f"🔄 Filtered results: {len(filtered_matches)}/{original_count} matches remaining")
        data_logger.info(f"🔄 Final statistics: {json.dumps(stats, indent=2)}")
        
        logger.info(f"🔄 FILTERS APPLIED SUCCESSFULLY: {len(filtered_matches)}/{original_count} matches remaining")
        logger.info(f"🔄 FILTERED STATISTICS: {stats}")
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export-csv', methods=['GET'])
def export_csv():
    """Export raw speed calculation data to CSV format
    
    Exports only the records that pass all applied filters (the data used for calculations).
    If no filters are applied, exports all original data.
    """
    global processed_matches, current_filtered_matches
    
    try:
        # Export filtered data if filters are applied, otherwise export all original data
        data_to_export = current_filtered_matches if current_filtered_matches else processed_matches
        
        if not data_to_export:
            return jsonify({'error': 'No data available to export. Please load data first.'}), 400
        
        # Log export request
        data_source = 'filtered' if current_filtered_matches else 'original'
        ui_logger.info("📥 CSV EXPORT REQUEST - User requested CSV export")
        ui_logger.info(f"📥 Exporting {len(data_to_export)} records ({data_source} data - records used for calculations)")
        
        # Create CSV content
        import csv
        from io import StringIO
        
        output = StringIO()
        
        # Determine CSV columns based on available fields in the first match
        if data_to_export:
            # Get all possible fields from all matches
            all_fields = set()
            for match in data_to_export:
                all_fields.update(match.keys())
            
            # Define column order (prioritize important fields)
            priority_fields = ['pair_index', 'pair_num', 'keypoint_idx', 'speed', 'pixel_distance', 'distance', 
                             'time_difference', 'time_diff', 'gsd_used', 'image1', 'image2', 'image1_path', 
                             'image2_path', 'brightness', 'contrast', 'cloudiness', 'matches', 'match_distance']
            
            # Order columns: priority fields first, then remaining fields alphabetically
            columns = []
            for field in priority_fields:
                if field in all_fields:
                    columns.append(field)
            for field in sorted(all_fields):
                if field not in columns:
                    columns.append(field)
            
            # Flatten nested structures (like image1_properties, image2_properties) into separate columns
            flattened_columns = []
            nested_fields_to_flatten = ['image1_properties', 'image2_properties']
            
            # First, collect all regular columns (excluding nested dict fields)
            for col in columns:
                if col not in nested_fields_to_flatten:
                    flattened_columns.append(col)
            
            # Then, add flattened columns for nested dict fields
            for nested_field in nested_fields_to_flatten:
                if nested_field in all_fields:
                    # Find all unique sub-keys in this nested field across all matches
                    sub_keys = set()
                    for match in data_to_export:
                        nested_value = match.get(nested_field)
                        if isinstance(nested_value, dict):
                            sub_keys.update(nested_value.keys())
                    
                    # Add flattened columns for each sub-key
                    for sub_key in sorted(sub_keys):
                        flattened_col_name = f"{nested_field}_{sub_key}"
                        flattened_columns.append(flattened_col_name)
            
            writer = csv.DictWriter(output, fieldnames=flattened_columns, extrasaction='ignore')
            writer.writeheader()
            
            # Write data rows
            for match in data_to_export:
                row = {}
                for key in flattened_columns:
                    # Check if this is a flattened nested field (e.g., image1_properties_brightness)
                    if key.startswith('image1_properties_') or key.startswith('image2_properties_'):
                        # Extract parent field and sub-key
                        if key.startswith('image1_properties_'):
                            parent_key = 'image1_properties'
                            sub_key = key.replace('image1_properties_', '')
                        else:  # image2_properties_
                            parent_key = 'image2_properties'
                            sub_key = key.replace('image2_properties_', '')
                        
                        # Get value from nested dict
                        parent_value = match.get(parent_key)
                        if isinstance(parent_value, dict):
                            row[key] = parent_value.get(sub_key, '')
                        else:
                            row[key] = ''
                    else:
                        # Regular field (not nested)
                        value = match.get(key)
                        if value is None:
                            row[key] = ''
                        elif isinstance(value, dict):
                            # Skip dict fields - they should be flattened
                            row[key] = ''
                        elif isinstance(value, list):
                            # Convert lists to comma-separated string
                            row[key] = ','.join(str(v) for v in value)
                        elif isinstance(value, tuple):
                            # Convert tuples to comma-separated string
                            row[key] = ','.join(str(v) for v in value)
                        else:
                            row[key] = value
                writer.writerow(row)
        
        csv_content = output.getvalue()
        output.close()
        
        # Create response with CSV content
        from flask import Response
        response = Response(
            csv_content,
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename=iss_speed_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            }
        )
        
        logger.info(f"📥 CSV EXPORT SUCCESS: {len(data_to_export)} records exported")
        return response
        
    except Exception as e:
        logger.error(f"📥 CSV EXPORT ERROR: {str(e)}")
        return jsonify({'error': f'Failed to export CSV: {str(e)}'}), 500

@app.route('/api/requirements', methods=['POST'])
def get_requirements():
    """Generate requirements document for replicating the current logic with user-selected parameters"""
    try:
        data = request.json
        requirements = {
            'title': 'ISS Speed Calculation Requirements',
            'description': 'Parameters required to replicate the ISS speed calculation logic with current settings',
            'parameters': {}
        }
        
        # Algorithm Configuration
        requirements['parameters']['algorithm_configuration'] = {
            'algorithm': data.get('algorithm', 'ORB'),
            'use_flann': data.get('use_flann', False),
            'max_features': data.get('max_features', 1000),
            'contrast_enhancement': data.get('contrast_enhancement', 'clahe'),
            'use_ransac_homography': data.get('use_ransac_homography', False),
            'ransac_threshold': data.get('ransac_threshold', 5.0),
            'ransac_min_matches': data.get('ransac_min_matches', 10)
        }
        
        # Image Range
        requirements['parameters']['image_range'] = {
            'start_idx': data.get('start_idx', 0),
            'end_idx': data.get('end_idx', 0)
        }
        
        # Filters
        filters = {}
        if data.get('enable_keypoint_percentile', False):
            filters['keypoint_percentile'] = {
                'enabled': True,
                'bottom_percentile': data.get('keypoint_percentile_bottom', 5),
                'top_percentile': data.get('keypoint_percentile_top', 5)
            }
        
        if data.get('enable_percentile', False):
            filters['minimum_matches'] = {
                'enabled': True,
                'minimum_matches': data.get('pair_percentile', 10)
            }
        
        if data.get('enable_std_dev', False):
            filters['standard_deviation'] = {
                'enabled': True,
                'multiplier': data.get('std_dev_multiplier', 2.0)
            }
        
        if data.get('enable_mad', False):
            filters['mad'] = {
                'enabled': True,
                'multiplier': data.get('mad_multiplier', 3.0)
            }
        
        if data.get('enable_cloudiness', False):
            filters['cloudiness'] = {
                'enabled': True,
                'include_partly_cloudy': data.get('include_partly_cloudy', True),
                'include_mostly_cloudy': data.get('include_mostly_cloudy', True),
                'clear_brightness_min': data.get('clear_brightness_min', 120),
                'clear_contrast_min': data.get('clear_contrast_min', 55),
                'cloudy_brightness_max': data.get('cloudy_brightness_max', 60),
                'cloudy_contrast_max': data.get('cloudy_contrast_max', 40)
            }
        
        if data.get('enable_custom_gsd', False):
            filters['custom_gsd'] = {
                'enabled': True,
                'gsd_value': data.get('custom_gsd', 12648)
            }
        
        requirements['parameters']['filters'] = filters
        
        # Format as readable text
        formatted_text = format_requirements_text(requirements)
        
        ui_logger.info("📋 REQUIREMENTS REQUEST - User requested requirements document")
        return jsonify({
            'success': True,
            'requirements': requirements,
            'formatted_text': formatted_text
        })
        
    except Exception as e:
        logger.error(f"📋 REQUIREMENTS ERROR: {str(e)}")
        return jsonify({'error': f'Failed to generate requirements: {str(e)}'}), 500

def format_requirements_text(requirements):
    """Format requirements as AI-agent-friendly implementation guide"""
    lines = []
    lines.append("=" * 80)
    lines.append("ISS SPEED CALCULATION - CONFIGURATION SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    
    # Algorithm Configuration
    algo_config = requirements['parameters']['algorithm_configuration']
    img_range = requirements['parameters']['image_range']
    filters = requirements['parameters']['filters']
    
    # Calculate GSD value
    custom_gsd_enabled = 'custom_gsd' in filters
    if custom_gsd_enabled:
        gsd_value = filters['custom_gsd']['gsd_value']
    else:
        gsd_value = 12648
    
    use_flann = algo_config['use_flann']
    use_ransac = algo_config['use_ransac_homography']
    enhancement = algo_config['contrast_enhancement']
    
    # CONFIGURATION SUMMARY ONLY
    lines.append("Image Capture:")
    lines.append("  - Source: Raspberry Pi Camera (picamera)")
    lines.append("  - Capture Interval: Configurable (e.g., 14 seconds)")
    lines.append("  - Program Duration: 9 minutes 30 seconds (570 seconds)")
    lines.append("  - Processing: Real-time, pairs processed as captured")
    lines.append("")
    lines.append("Algorithm: {algorithm}".format(algorithm=algo_config['algorithm']))
    lines.append("FLANN Matching: {use_flann}".format(use_flann=use_flann))
    lines.append("Max Features: {max_features}".format(max_features=algo_config['max_features']))
    lines.append("Contrast Enhancement: {enhancement}".format(enhancement=enhancement))
    lines.append("RANSAC/Homography: {use_ransac}".format(use_ransac=use_ransac))
    if use_ransac:
        lines.append("  RANSAC Threshold: {threshold}".format(threshold=algo_config['ransac_threshold']))
        lines.append("  RANSAC Min Matches: {min_matches}".format(min_matches=algo_config['ransac_min_matches']))
    lines.append("GSD: {gsd} cm/pixel".format(gsd=gsd_value))
    lines.append("")
    lines.append("Data Structure:")
    lines.append("  - All keypoints from all image pairs are stored in a single list in memory")
    lines.append("  - Each keypoint/match contains: speed, pixel_distance, time_difference, gsd_used, pair_index")
    lines.append("  - Final speed calculation: Use the MEAN (average) of all filtered match speeds")
    lines.append("")
    lines.append("Output:")
    lines.append("  - Result File: result.txt")
    lines.append("  - Contains: Mean speed only")
    lines.append("")
    lines.append("Enabled Filters:")
    if filters:
        # Filter out cloudiness filter if it doesn't actually filter anything
        effective_filters = {}
        for filter_name, filter_data in filters.items():
            if filter_name == 'cloudiness':
                # Only include if it actually filters (not if both partly and mostly are included)
                include_partly = filter_data.get('include_partly_cloudy', True)
                include_mostly = filter_data.get('include_mostly_cloudy', True)
                if not (include_partly and include_mostly):
                    effective_filters[filter_name] = filter_data
            else:
                effective_filters[filter_name] = filter_data
        
        if effective_filters:
            for filter_name, filter_data in effective_filters.items():
                lines.append(f"  - {filter_name}: {filter_data}")
        else:
            lines.append("  - None")
    else:
        lines.append("  - None")
    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REQUIREMENTS")
    lines.append("=" * 80)
    
    return "\n".join(lines)

@app.route('/api/processing-status')
def get_processing_status():
    """Get current processing status for progress bar"""
    global processing_status
    # Return a copy to avoid any thread safety issues
    status_copy = {
        'progress': processing_status.get('progress', 0),
        'current_pair': processing_status.get('current_pair', 0),
        'total_pairs': processing_status.get('total_pairs', 0),
        'status': processing_status.get('status', 'idle')
    }
    # Include error if present
    if 'error' in processing_status:
        status_copy['error'] = processing_status['error']
    return jsonify(status_copy)

@app.route('/api/plot-data')
def get_plot_data():
    """Get plot data for visualization - same as original dashboard"""
    global processed_matches, current_filters, current_filtered_matches
    
    # Determine which data to use: filtered data if available, otherwise original data
    data_to_use = current_filtered_matches if current_filtered_matches else processed_matches
    data_source = "filtered" if current_filtered_matches else "original"
    
    # Log user interaction
    ui_logger.info("📈 PLOT DATA API CALL - User requested Section 5 & 6 graph data")
    ui_logger.info(f"📈 Using {data_source} data: {len(data_to_use) if data_to_use else 0} matches")
    ui_logger.info(f"📈 Current filters applied: {json.dumps(current_filters, indent=2)}")
    
    # Log data processing
    data_logger.info("📈 SECTION 5 & 6 GRAPH DATA GENERATION STARTED")
    data_logger.info(f"📈 Using {data_source} data: {len(data_to_use) if data_to_use else 0} matches")
    data_logger.info(f"📈 Original data: {len(processed_matches) if processed_matches else 0} matches")
    data_logger.info(f"📈 Filtered data: {len(current_filtered_matches) if current_filtered_matches else 0} matches")
    data_logger.info(f"📈 Filters applied: {json.dumps(current_filters, indent=2)}")
    
    logger.info(f"📈 PLOT DATA API CALL - using {data_source} data: {len(data_to_use) if data_to_use else 0}")
    logger.info(f"📈 Current filters: {current_filters}")
    
    print(f"🔍 Plot data request - using {data_source} data: {len(data_to_use) if data_to_use else 0}")
    print(f"🔍 Current filters: {current_filters}")
    if current_filters.get('enable_ml_classification', False):
        print(f"🔍 ML classification enabled: True")
    
    if not data_to_use:
        print("❌ No processed matches available")
        return jsonify({'error': 'No data processed'})
    
    # Debug: Check the structure of data_to_use
    if data_to_use:
        sample_match = data_to_use[0]
        print(f"🔍 Sample match structure: {list(sample_match.keys())}")
        print(f"🔍 Sample match speed: {sample_match.get('speed', 'NO SPEED')}")
        print(f"🔍 Sample match pair_index: {sample_match.get('pair_index', 'NO PAIR_INDEX')}")
        if current_filters.get('enable_ml_classification', False):
            print(f"🔍 Sample match ml_classification: {sample_match.get('ml_classification', 'NO ML CLASSIFICATION')}")
        
        # Count ML classifications (only if ML is enabled)
        if current_filters.get('enable_ml_classification', False):
            ml_classifications = [m.get('ml_classification') for m in data_to_use if m.get('ml_classification') is not None]
            print(f"🔍 ML classifications found: {len(ml_classifications)} out of {len(data_to_use)} matches")
            if ml_classifications:
                from collections import Counter
                ml_counts = Counter(ml_classifications)
                print(f"🔍 ML classification counts: {dict(ml_counts)}")
    
    # Use the appropriate data source (already determined above)
    filtered_matches = data_to_use
    print(f"🔍 Using {data_source} data for plot generation: {len(filtered_matches) if filtered_matches else 0}")
    
    if not filtered_matches:
        print("❌ No data available for plotting")
        return jsonify({'error': 'No data available for plotting'})
    
    # Extract speeds
    speeds = [match['speed'] for match in filtered_matches]
    
    # Group by pairs for pair analysis
    pair_data = {}
    for match in filtered_matches:
        pair_idx = match['pair_index']
        if pair_idx not in pair_data:
            pair_data[pair_idx] = []
        pair_data[pair_idx].append(match['speed'])
    
    # Calculate pair averages
    pair_averages = []
    pair_numbers = []
    pair_colors = []
    
    for pair_idx, pair_speeds in pair_data.items():
        avg_speed = sum(pair_speeds) / len(pair_speeds)
        pair_averages.append(avg_speed)
        pair_numbers.append(pair_idx + 1)  # Convert to 1-based indexing
        
        # Determine color based on classification type (ML and cloudiness are mutually exclusive)
        sample_match = next((m for m in filtered_matches if m['pair_index'] == pair_idx), None)
        if sample_match:
            ml_enabled = current_filters.get('enable_ml_classification', False)
            cloudiness_enabled = current_filters.get('enable_cloudiness', False)
            
            # Enforce mutual exclusivity: ML takes priority if both are enabled
            if ml_enabled:
                # Use ML classification ONLY (never cloudiness)
                ml_class = sample_match.get('ml_classification')
                
                if ml_class is not None:
                    print(f"🎨 Using ML classification for pair {pair_idx}: {ml_class}")
                    if ml_class == 'Good':
                        pair_colors.append('green')  # Good
                    elif ml_class == 'Not_Good':
                        pair_colors.append('red')    # Not_Good
                    else:
                        pair_colors.append('gray')   # Unknown
                else:
                    # ML enabled but classification not available - use placeholder
                    print(f"⚠️ ML enabled but classification missing for pair {pair_idx}, using gray")
                    pair_colors.append('gray')  # Placeholder when ML data is missing
            
            elif cloudiness_enabled:
                # Use cloudiness classification
                if sample_match.get('image1_properties') and sample_match.get('image2_properties'):
                    img1_props = sample_match['image1_properties']
                    img2_props = sample_match['image2_properties']
                    avg_brightness = (img1_props.get('brightness', 0) + img2_props.get('brightness', 0)) / 2
                    avg_contrast = (img1_props.get('contrast', 0) + img2_props.get('contrast', 0)) / 2
                    
                    # Classify cloudiness using current filter thresholds
                    clear_brightness_min = current_filters.get('clear_brightness_min', 120)
                    clear_contrast_min = current_filters.get('clear_contrast_min', 55)
                    cloudy_brightness_max = current_filters.get('cloudy_brightness_max', 60)
                    cloudy_contrast_max = current_filters.get('cloudy_contrast_max', 40)
                    
                    if avg_brightness >= clear_brightness_min and avg_contrast >= clear_contrast_min:
                        pair_colors.append('green')  # clear
                    elif avg_brightness <= cloudy_brightness_max or avg_contrast <= cloudy_contrast_max:
                        pair_colors.append('red')    # mostly cloudy
                    else:
                        pair_colors.append('orange') # partly cloudy
                else:
                    pair_colors.append('gray')  # unknown
            else:
                # Neither ML nor cloudiness enabled - use gray/default
                pair_colors.append('gray')
        else:
            pair_colors.append('gray')  # unknown
    
    # Create plot data structure (same as original dashboard)
    plot_data = {
        'histogram': {
            'speeds': speeds,
            'mean': float(np.mean(speeds)),
            'median': float(np.median(speeds)),
            'std': float(np.std(speeds))
        },
        'boxplot': {
            'speeds': speeds
        },
        'pairs': {
            'pairs': pair_numbers,
            'original_pairs': pair_numbers,  # Same as pairs for v2
            'means': pair_averages,
            'medians': [np.median([match['speed'] for match in filtered_matches if match['pair_index'] + 1 == pair_num]) for pair_num in pair_numbers],
            'stds': [np.std([match['speed'] for match in filtered_matches if match['pair_index'] + 1 == pair_num]) for pair_num in pair_numbers],
            'colors': pair_colors
        }
    }
    
    print(f"✅ Plot data created - speeds: {len(speeds)}, pairs: {len(pair_numbers)}")
    print(f"📊 Speed range: {min(speeds):.3f} - {max(speeds):.3f} km/s")
    
    # Debug: Print the actual values being calculated for histogram
    print(f"🔍 Section 5 Histogram Debug:")
    print(f"🔍   Total speeds used: {len(speeds)}")
    print(f"🔍   Mean: {plot_data['histogram']['mean']:.3f} km/s")
    print(f"🔍   Median: {plot_data['histogram']['median']:.3f} km/s")
    print(f"🔍   Speed range: {min(speeds):.3f} - {max(speeds):.3f} km/s")
    
    # Log detailed plot data generation results
    data_logger.info("📈 SECTION 5 & 6 GRAPH DATA GENERATION COMPLETED")
    data_logger.info(f"📈 Final plot data: {json.dumps(plot_data, indent=2)}")
    data_logger.info(f"📈 Pairs data: {len(plot_data.get('pairs', {}).get('pairs', []))} pairs")
    data_logger.info(f"📈 Histogram data: {len(plot_data.get('histogram', {}).get('speeds', []))} speeds")
    data_logger.info(f"📈 Speed range: {min(speeds):.3f} - {max(speeds):.3f} km/s")
    
    logger.info(f"📈 PLOT DATA GENERATED - pairs: {len(plot_data.get('pairs', {}).get('pairs', []))}, histogram: {len(plot_data.get('histogram', {}).get('speeds', []))}")
    logger.info(f"📈 PLOT DATA SUMMARY: {plot_data}")
    
    return jsonify(plot_data)

@app.route('/api/pair/<int:pair_num>')
def get_pair_images(pair_num):
    """Get images for a specific pair with keypoint matches - same as original dashboard"""
    global processed_matches
    
    if not processed_matches:
        return jsonify({'error': 'No data processed'})
    
    # Find matches for this pair
    pair_matches = [match for match in processed_matches if match['pair_index'] + 1 == pair_num]
    
    if not pair_matches:
        return jsonify({'error': 'Pair not found'})
    
    # Get sample match to extract pair information
    sample_match = pair_matches[0]
    
    try:
        # Load images
        image1_path = sample_match['image1_path']
        image2_path = sample_match['image2_path']
        
        image1_cv = cv2.imread(image1_path, 0)
        image2_cv = cv2.imread(image2_path, 0)
        
        if image1_cv is None or image2_cv is None:
            return jsonify({'error': 'Could not load images'})
        
        # Recalculate keypoints and matches for visualization
        if sample_match['algorithm'] == 'ORB':
            detector = cv2.ORB_create(nfeatures=MAX_FEATURES)
        else:  # SIFT
            detector = cv2.SIFT_create(nfeatures=MAX_FEATURES)
        
        kp1, des1 = detector.detectAndCompute(image1_cv, None)
        kp2, des2 = detector.detectAndCompute(image2_cv, None)
        
        if des1 is None or des2 is None:
            return jsonify({'error': 'Could not detect features'})
        
        # Match descriptors
        if sample_match['use_flann']:
            if sample_match['algorithm'] == 'ORB':
                FLANN_INDEX_LSH = 6
                index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
                search_params = dict(checks=50)
            else:  # SIFT
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
            
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            
            # Apply ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            matches = good_matches
        else:
            if sample_match['algorithm'] == 'ORB':
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            else:  # SIFT
                matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = matcher.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
        
        # Apply RANSAC/Homography filtering if enabled
        if sample_match.get('use_ransac_homography', False) and len(matches) >= sample_match.get('ransac_min_matches', 10):
            matches = apply_homography_filtering(kp1, kp2, matches, 
                                               sample_match.get('ransac_threshold', 5.0), 
                                               sample_match.get('ransac_min_matches', 10))
        
        # Create matches visualization
        match_img = cv2.drawMatches(image1_cv, kp1, image2_cv, kp2, matches[:100], None, 
                                   matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), 
                                   matchesMask=None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # Make the lines thicker
        height, width = match_img.shape[:2]
        img_width = width // 2
        
        for match in matches[:100]:
            pt1 = (int(kp1[match.queryIdx].pt[0]), int(kp1[match.queryIdx].pt[1]))
            pt2 = (int(kp2[match.trainIdx].pt[0] + img_width), int(kp2[match.trainIdx].pt[1]))
            cv2.line(match_img, pt1, pt2, (0, 255, 0), 3)
        
        # Resize for web display
        height, width = match_img.shape[:2]
        if width > 1600:
            scale = 1600 / width
            new_width = 1600
            new_height = int(height * scale)
            match_img = cv2.resize(match_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', match_img)
        matches_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Get individual images as base64
        def image_to_base64(image_path):
            try:
                with open(image_path, 'rb') as f:
                    return base64.b64encode(f.read()).decode('utf-8')
            except:
                return None
        
        image1_b64 = image_to_base64(image1_path)
        image2_b64 = image_to_base64(image2_path)
        
        if not image1_b64 or not image2_b64:
            return jsonify({'error': 'Could not load individual images'})
        
        # Calculate statistics for this pair
        pair_speeds = [match['speed'] for match in pair_matches]
        avg_speed = np.mean(pair_speeds)
        std_speed = np.std(pair_speeds)
        
        # Get image properties
        img1_props = sample_match.get('image1_properties', {})
        img2_props = sample_match.get('image2_properties', {})
        
        # Calculate time difference
        time_diff = sample_match.get('time_difference', 0)
        
        # Determine cloudiness
        avg_brightness = (img1_props.get('brightness', 0) + img2_props.get('brightness', 0)) / 2
        avg_contrast = (img1_props.get('contrast', 0) + img2_props.get('contrast', 0)) / 2
        
        if avg_brightness >= 120 and avg_contrast >= 55:
            cloudiness = 'clear'
        elif avg_brightness <= 60 or avg_contrast <= 56:
            cloudiness = 'mostly cloudy'
        else:
            cloudiness = 'partly cloudy'
        
        # Calculate accuracy
        accuracy = 100 * (1 - abs(avg_speed - 7.66) / 7.66)
        
        return jsonify({
            'pair_num': pair_num,
            'image1': os.path.basename(image1_path),
            'image2': os.path.basename(image2_path),
            'image1_data': image1_b64,
            'image2_data': image2_b64,
            'matches_data': matches_b64,
            'avg_speed': float(avg_speed),
            'std_speed': float(std_speed),
            'keypoint_count': len(pair_matches),
            'time_diff': time_diff,
            'brightness': avg_brightness,
            'contrast': avg_contrast,
            'cloudiness': cloudiness,
            'accuracy': float(accuracy),
            'total_matches': len(matches),
            'displayed_matches': min(100, len(matches))
        })
        
    except Exception as e:
        return jsonify({'error': f'Error creating keypoint visualization: {str(e)}'})

def process_image_pair(image1_path, image2_path, algorithm, use_flann, use_ransac_homography=False, ransac_threshold=5.0, ransac_min_matches=10, max_features=MAX_FEATURES):
    """Process a single image pair and return matches with speeds"""
    try:
        # Load images with contrast enhancement for better feature detection
        img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            return []
        
        # Enhance contrast for better feature detection (will be set by the API call)
        # Default to CLAHE if not specified
        enhancement_method = getattr(process_image_pair, 'enhancement_method', 'clahe')
        if enhancement_method != 'none':
            img1 = enhance_image_contrast(img1, method=enhancement_method, clip_limit=3.0)
            img2 = enhance_image_contrast(img2, method=enhancement_method, clip_limit=3.0)
        
        # Get time difference
        time_diff = get_time_difference(image1_path, image2_path)
        if time_diff <= 0:
            return []
        
        # GPS speed calculation removed - not allowed to use GPS location data
        
        # Detect features based on algorithm
        if algorithm == 'ORB':
            detector = cv2.ORB_create(nfeatures=max_features)
            kp1, desc1 = detector.detectAndCompute(img1, None)
            kp2, desc2 = detector.detectAndCompute(img2, None)
            
            if desc1 is None or desc2 is None:
                return []
            
            # Match features
            if use_flann:
                # FLANN for ORB (using LSH)
                FLANN_INDEX_LSH = 6
                index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(desc1, desc2, k=2)
                
                # Apply Lowe's ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)
                matches = good_matches
            else:
                # Brute force for ORB
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = matcher.match(desc1, desc2)
                matches = sorted(matches, key=lambda x: x.distance)
        
        else:  # SIFT
            detector = cv2.SIFT_create(nfeatures=max_features)
            kp1, desc1 = detector.detectAndCompute(img1, None)
            kp2, desc2 = detector.detectAndCompute(img2, None)
            
            if desc1 is None or desc2 is None:
                return []
            
            # Match features
            if use_flann:
                # FLANN for SIFT
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(desc1, desc2, k=2)
                
                # Apply Lowe's ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)
                matches = good_matches
            else:
                # Brute force for SIFT
                matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
                matches = matcher.match(desc1, desc2)
                matches = sorted(matches, key=lambda x: x.distance)
        
        # Apply RANSAC/Homography filtering if requested
        original_match_count = len(matches)
        if use_ransac_homography and len(matches) >= ransac_min_matches:
            print(f"🔧 Applying RANSAC/Homography filtering: {len(matches)} matches, threshold={ransac_threshold}, min_matches={ransac_min_matches}")
            matches = apply_homography_filtering(kp1, kp2, matches, ransac_threshold, ransac_min_matches)
            print(f"🔧 Filtering result: {original_match_count} → {len(matches)} matches")
        elif use_ransac_homography:
            print(f"⚠️ RANSAC/Homography filtering skipped: only {len(matches)} matches (need {ransac_min_matches})")
        else:
            print(f"ℹ️ RANSAC/Homography filtering disabled: {len(matches)} matches")
        
        # Calculate speeds for each match
        match_results = []
        for match in matches:
            # Get coordinates
            pt1 = kp1[match.queryIdx].pt
            pt2 = kp2[match.trainIdx].pt
            
            # Calculate pixel distance
            pixel_distance = math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
            
            # Calculate speed (using default GSD for now)
            gsd = 12648  # Default GSD
            speed = calculate_speed_in_kmps(pixel_distance, gsd, time_diff)
            
            match_results.append({
                'speed': speed,
                'pixel_distance': pixel_distance,
                'match_distance': match.distance,
                'pt1': pt1,
                'pt2': pt2
            })
        
        return match_results
        
    except Exception as e:
        print(f"Error processing image pair: {e}")
        return []


def apply_homography_filtering(kp1, kp2, matches, threshold=5.0, min_matches=10):
    """Apply homography-based outlier filtering using RANSAC"""
    if len(matches) < min_matches:
        print(f"⚠️ Not enough matches for homography: {len(matches)} < {min_matches}")
        return matches
    
    try:
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        print(f"🔍 Computing homography with {len(matches)} matches, threshold={threshold}")
        
        # Find homography with RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, threshold)
        
        if H is not None and mask is not None:
            # Count inliers
            inlier_count = np.sum(mask)
            inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]
            print(f"🔍 Homography filtering: {len(matches)} → {len(inlier_matches)} matches ({inlier_count} inliers)")
            return inlier_matches
        else:
            print(f"⚠️ Homography filtering failed (H={H}, mask={mask}), returning original {len(matches)} matches")
            return matches
            
    except Exception as e:
        print(f"❌ Error in homography filtering: {e}")
        return matches

def apply_match_filters(matches, filters):
    """Apply filters to matches - same filters as original dashboard (excluding GPS)"""
    filtered = matches.copy()
    
    # Enforce mutual exclusivity: ML classification and cloudiness cannot both be enabled
    ml_enabled = filters.get('enable_ml_classification', False)
    cloudiness_enabled = filters.get('enable_cloudiness', False)
    
    if ml_enabled and cloudiness_enabled:
        # ML takes priority - disable cloudiness if both are enabled
        print("⚠️ Both ML classification and cloudiness filters are enabled. ML takes priority (cloudiness disabled).")
        filters = filters.copy()  # Don't modify the original filters dict
        filters['enable_cloudiness'] = False
        cloudiness_enabled = False
    
    # Keypoint percentile filter (remove bottom X% and top Y% by speed)
    if filters.get('enable_keypoint_percentile', False) and len(filtered) > 0:
        keypoint_percentile_bottom = filters.get('keypoint_percentile_bottom', 5)
        keypoint_percentile_top = filters.get('keypoint_percentile_top', 5)
        speeds = [m['speed'] for m in filtered]
        
        # Calculate thresholds for bottom and top percentiles separately
        bottom_threshold = np.percentile(speeds, keypoint_percentile_bottom)
        top_threshold = np.percentile(speeds, 100 - keypoint_percentile_top)
        
        # Keep only matches within the percentile range
        filtered = [m for m in filtered if bottom_threshold <= m['speed'] <= top_threshold]
    
    # Minimum matches filter (filter pairs by minimum keypoint count)
    if filters.get('enable_percentile', False) and len(filtered) > 0:
        min_matches = filters.get('pair_percentile', 10)  # Now represents minimum matches, not percentile
        
        # Count keypoints per pair
        pair_counts = {}
        for m in filtered:
            pair_num = m['pair_index']
            pair_counts[pair_num] = pair_counts.get(pair_num, 0) + 1
        
        if pair_counts:
            # Use minimum match count instead of percentile
            pairs_to_keep = [pair_num for pair_num, count in pair_counts.items() if count >= min_matches]
            
            # Safety check: ensure we keep at least 3 pairs or 50% of pairs, whichever is smaller
            min_pairs_to_keep = min(3, max(1, len(pair_counts) // 2))
            if len(pairs_to_keep) < min_pairs_to_keep and len(pair_counts) >= min_pairs_to_keep:
                sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
                pairs_to_keep = [pair_num for pair_num, _ in sorted_pairs[:min_pairs_to_keep]]
            
            # Filter matches to only include pairs above threshold
            filtered = [m for m in filtered if m['pair_index'] in pairs_to_keep]
    
    # Standard deviation filter (outlier removal)
    if filters.get('enable_std_dev', False) and len(filtered) > 0:
        std_dev_multiplier = filters.get('std_dev_multiplier', 2.0)
        speeds = [m['speed'] for m in filtered]
        mean_speed = np.mean(speeds)
        std_speed = np.std(speeds)
        
        # Calculate bounds
        lower_bound = mean_speed - (std_dev_multiplier * std_speed)
        upper_bound = mean_speed + (std_dev_multiplier * std_speed)
        
        # Filter matches within bounds
        filtered = [m for m in filtered if lower_bound <= m['speed'] <= upper_bound]
    
    # MAD (Median Absolute Deviation) filter (robust outlier removal)
    if filters.get('enable_mad', False) and len(filtered) > 0:
        mad_multiplier = filters.get('mad_multiplier', 3.0)
        speeds = [m['speed'] for m in filtered]
        median_speed = np.median(speeds)
        
        # Calculate MAD: median of absolute deviations from median
        absolute_deviations = [abs(speed - median_speed) for speed in speeds]
        mad = np.median(absolute_deviations)
        
        # Calculate bounds
        lower_bound = median_speed - (mad_multiplier * mad)
        upper_bound = median_speed + (mad_multiplier * mad)
        
        # Filter matches within bounds
        filtered = [m for m in filtered if lower_bound <= m['speed'] <= upper_bound]
    
    # Cloudiness filter (based on image properties)
    if filters.get('enable_cloudiness', False) and len(filtered) > 0:
        clear_brightness_min = filters.get('clear_brightness_min', 120)
        clear_contrast_min = filters.get('clear_contrast_min', 50)
        cloudy_brightness_max = filters.get('cloudy_brightness_max', 60)
        cloudy_contrast_max = filters.get('cloudy_contrast_max', 15)
        include_partly_cloudy = filters.get('include_partly_cloudy', True)
        include_mostly_cloudy = filters.get('include_mostly_cloudy', True)
        
        print(f"🔍 Cloudiness filter enabled with extreme thresholds:")
        print(f"   Clear: brightness >= {clear_brightness_min}, contrast >= {clear_contrast_min}")
        print(f"   Cloudy: brightness <= {cloudy_brightness_max}, contrast <= {cloudy_contrast_max}")
        print(f"   Include partly cloudy: {include_partly_cloudy}")
        print(f"   Include mostly cloudy: {include_mostly_cloudy}")
        print(f"   Input matches: {len(filtered)}")
        
        cloudiness_filtered = []
        for m in filtered:
            img1_props = m.get('image1_properties', {})
            img2_props = m.get('image2_properties', {})
            
            if img1_props and img2_props:
                # Determine cloudiness based on average properties
                avg_brightness = (img1_props.get('brightness', 0) + img2_props.get('brightness', 0)) / 2
                avg_contrast = (img1_props.get('contrast', 0) + img2_props.get('contrast', 0)) / 2
                
                # Classify cloudiness
                if avg_brightness >= clear_brightness_min and avg_contrast >= clear_contrast_min:
                    cloudiness = 'clear'
                elif avg_brightness <= cloudy_brightness_max or avg_contrast <= cloudy_contrast_max:
                    cloudiness = 'mostly_cloudy'
                else:
                    cloudiness = 'partly_cloudy'
                
                # Debug: Show classification for first few matches
                if len(cloudiness_filtered) < 3:
                    print(f"     Match {len(cloudiness_filtered)+1}: brightness={avg_brightness:.1f}, contrast={avg_contrast:.1f} → {cloudiness}")
                
                # Include based on cloudiness settings
                if cloudiness == 'clear' or \
                   (cloudiness == 'partly_cloudy' and include_partly_cloudy) or \
                   (cloudiness == 'mostly_cloudy' and include_mostly_cloudy):
                    cloudiness_filtered.append(m)
            else:
                # If no image properties, include the match
                cloudiness_filtered.append(m)
        
        filtered = cloudiness_filtered
        print(f"   Output matches after cloudiness filter: {len(filtered)}")
        
        # Debug: Show some sample brightness/contrast values
        if len(filtered) > 0:
            sample_match = filtered[0]
            img1_props = sample_match.get('image1_properties', {})
            img2_props = sample_match.get('image2_properties', {})
            if img1_props and img2_props:
                avg_brightness = (img1_props.get('brightness', 0) + img2_props.get('brightness', 0)) / 2
                avg_contrast = (img1_props.get('contrast', 0) + img2_props.get('contrast', 0)) / 2
                print(f"   Sample match: brightness={avg_brightness:.1f}, contrast={avg_contrast:.1f}")
        else:
            print(f"   ⚠️ No matches passed the cloudiness filter!")
            
        # Debug: Show original data properties before filtering
        if len(filtered) < len(processed_matches):
            print(f"   Original data sample (before filtering):")
            for i, match in enumerate(processed_matches[:3]):  # Show first 3 matches
                img1_props = match.get('image1_properties', {})
                img2_props = match.get('image2_properties', {})
                if img1_props and img2_props:
                    avg_brightness = (img1_props.get('brightness', 0) + img2_props.get('brightness', 0)) / 2
                    avg_contrast = (img1_props.get('contrast', 0) + img2_props.get('contrast', 0)) / 2
                    print(f"     Match {i+1}: brightness={avg_brightness:.1f}, contrast={avg_contrast:.1f}")
                else:
                    print(f"     Match {i+1}: No image properties available")
    
    # ML Classification filter
    if filters.get('enable_ml_classification', False) and len(filtered) > 0:
        include_good = filters.get('include_good', True)
        include_not_good = filters.get('include_not_good', True)
        
        # Check if model is loaded before starting
        if not ml_model_loaded:
            print("🔄 ML model not loaded, attempting to load now...")
            load_ml_model()
        
        if ml_interpreter is None or not ml_labels:
            print("⚠️ ML model not available - cannot classify images")
            print(f"   ml_interpreter: {ml_interpreter is not None}")
            print(f"   ml_labels: {len(ml_labels) if ml_labels else 0} classes")
            print("")
            print("📋 Troubleshooting ML model loading:")
            if os.path.exists("model_edgetpu.tflite"):
                print("   • EdgeTPU model found but requires 'pycoral' library")
                print("   • Option 1: Install pycoral: pip install pycoral")
                print("   • Option 2: Export a regular TensorFlow Lite model from Teachable Machine")
                print("              (not EdgeTPU) and save as 'model_unquant.tflite'")
            elif os.path.exists("model_unquant.tflite"):
                print("   • Regular model found but failed to load - check error messages above")
            else:
                print("   • No model files found!")
                print("   • Please add 'model_unquant.tflite' and 'labels.txt' to the project directory")
            print("")
            return filtered  # Return original filtered data if ML model unavailable
        else:
            print(f"✅ ML model ready - interpreter available, {len(ml_labels)} classes: {list(ml_labels.values())}")
        
        print(f"🔄 Running ML classification on {len(filtered)} filtered matches...")
        ml_filtered = []
        
        # Track which images have already been classified to avoid infinite loops
        classified_images = {}
        
        for i, m in enumerate(filtered):
            if i % 10 == 0:  # Progress update every 10 matches
                print(f"📊 ML Progress: {i}/{len(filtered)} matches processed")
            # Run ML classification if not already done
            if m.get('ml_classification') is None:
                try:
                    image1_path = m.get('image1_path')
                    image2_path = m.get('image2_path')

                    if image1_path and image2_path:
                        # Check if we've already classified these images
                        if image1_path not in classified_images:
                            ml_class1, ml_conf1 = classify_image_with_ml(image1_path)
                            classified_images[image1_path] = (ml_class1, ml_conf1)
                        else:
                            ml_class1, ml_conf1 = classified_images[image1_path]
                        
                        if image2_path not in classified_images:
                            ml_class2, ml_conf2 = classify_image_with_ml(image2_path)
                            classified_images[image2_path] = (ml_class2, ml_conf2)
                        else:
                            ml_class2, ml_conf2 = classified_images[image2_path]

                        # Store individual classifications
                        m['ml_classification1'] = ml_class1
                        m['ml_confidence1'] = ml_conf1
                        m['ml_classification2'] = ml_class2
                        m['ml_confidence2'] = ml_conf2

                        # Determine pair-level ML classification
                        # Debug logging
                        if i < 3:  # Log first 3 matches for debugging
                            print(f"🔍 Debug ML classification for match {i+1}:")
                            print(f"   Image1: {os.path.basename(image1_path)} → {ml_class1} (conf: {ml_conf1:.3f})")
                            print(f"   Image2: {os.path.basename(image2_path)} → {ml_class2} (conf: {ml_conf2:.3f})")
                        
                        if ml_class1 == ml_class2 and ml_class1 is not None:
                            m['ml_classification'] = ml_class1
                            m['ml_confidence'] = (ml_conf1 + ml_conf2) / 2
                            if i < 3:
                                print(f"   → Pair classification: {ml_class1} (both agree)")
                        elif ml_class1 is not None and ml_class2 is not None:
                            # Images have different classifications - use higher confidence
                            if ml_conf1 >= ml_conf2:
                                m['ml_classification'] = ml_class1
                                m['ml_confidence'] = ml_conf1
                            else:
                                m['ml_classification'] = ml_class2
                                m['ml_confidence'] = ml_conf2
                            if i < 3:
                                print(f"   → Pair classification: {m['ml_classification']} (using higher confidence)")
                        elif ml_class1 is not None:
                            # Only image1 classified - use it
                            m['ml_classification'] = ml_class1
                            m['ml_confidence'] = ml_conf1
                            if i < 3:
                                print(f"   → Pair classification: {ml_class1} (image1 only)")
                        elif ml_class2 is not None:
                            # Only image2 classified - use it
                            m['ml_classification'] = ml_class2
                            m['ml_confidence'] = ml_conf2
                            if i < 3:
                                print(f"   → Pair classification: {ml_class2} (image2 only)")
                        else:
                            # Both returned None - classification failed
                            m['ml_classification'] = None
                            m['ml_confidence'] = 0.0
                            if i < 3:
                                print(f"   → Pair classification: None (both images failed)")
                    else:
                        m['ml_classification'] = None
                        m['ml_confidence'] = 0.0

                except Exception as e:
                    print(f"⚠️ ML classification failed for match {i}: {e}")
                    m['ml_classification'] = None
                    m['ml_confidence'] = 0.0
            
            # Apply ML filter
            ml_class = m.get('ml_classification')
            if ml_class == 'Good' and include_good:
                ml_filtered.append(m)
            elif ml_class == 'Not_Good' and include_not_good:
                ml_filtered.append(m)
            elif ml_class is None:
                # Include matches with no ML classification
                ml_filtered.append(m)
        
        print(f"✅ ML classification completed. {len(ml_filtered)} matches after ML filtering.")
        filtered = ml_filtered
    
    # Homography filtering (applied to individual matches, not the aggregated list)
    # Note: This filter needs to be applied during the initial processing phase
    # since it requires access to the original keypoints and matches
    # For now, we'll add a placeholder that could be implemented in the future
    
    return filtered

@app.route('/api/folders')
def get_folders():
    """Get list of available photos folders (project folder and uploaded folders)"""
    photos_folders = []
    
    # Get the project directory (where the script is located)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check current directory for 'photos' folder
    if os.path.exists('photos') and os.path.isdir('photos'):
        photos_folders.append('photos')
    
    # Only check the project directory for folders starting with 'photos-'
    try:
        for item in os.listdir(project_dir):
            item_path = os.path.join(project_dir, item)
            if os.path.isdir(item_path) and item.startswith('photos-'):
                # Use relative path from current working directory
                rel_path = os.path.relpath(item_path, os.getcwd())
                if rel_path not in photos_folders:
                    photos_folders.append(rel_path)
    except OSError:
        # If we can't read the directory, just return what we have
        pass
    
    # Check for uploaded photo folders
    try:
        upload_folder = 'uploaded_photos'
        if os.path.exists(upload_folder):
            for item in os.listdir(upload_folder):
                item_path = os.path.join(upload_folder, item)
                if os.path.isdir(item_path):
                    rel_path = os.path.relpath(item_path, os.getcwd())
                    if rel_path not in photos_folders:
                        photos_folders.append(rel_path)
    except OSError:
        pass
    
    # Sort and add image counts
    photos_folders = sorted(photos_folders)
    folder_data = []
    for folder in photos_folders:
        try:
            # Count various image formats
            image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif']
            image_count = 0
            for ext in image_extensions:
                image_count += len([f for f in os.listdir(folder) if f.lower().endswith(ext)])
            folder_data.append({'name': folder, 'count': image_count})
        except:
            folder_data.append({'name': folder, 'count': 0})
    
    return jsonify(folder_data)

@app.route('/api/environment')
def get_environment():
    """Get environment information for frontend"""
    return jsonify({
        'is_railway': is_railway_deployment(),
        'environment': 'railway' if is_railway_deployment() else 'local',
        'supports_folder_selection': not is_railway_deployment()
    })

@app.route('/api/delete-folder', methods=['DELETE'])
def delete_folder():
    """Delete an uploaded folder"""
    try:
        data = request.get_json()
        folder_path = data.get('folder_path')
        
        if not folder_path:
            return jsonify({'error': 'No folder path provided'}), 400
        
        # Security check: only allow deletion of folders in uploaded_photos directory
        if not folder_path.startswith('uploaded_photos/'):
            return jsonify({'error': 'Can only delete uploaded folders'}), 403
        
        # Check if folder exists
        if not os.path.exists(folder_path):
            return jsonify({'error': 'Folder not found'}), 404
        
        # Count files before deletion
        file_count = 0
        try:
            for file in os.listdir(folder_path):
                if os.path.isfile(os.path.join(folder_path, file)):
                    file_count += 1
        except OSError:
            pass
        
        # Delete the folder and all its contents
        import shutil
        shutil.rmtree(folder_path)
        
        return jsonify({
            'success': True,
            'message': f'Successfully deleted folder with {file_count} files',
            'deleted_folder': folder_path,
            'files_deleted': file_count
        })
        
    except Exception as e:
        return jsonify({'error': f'Delete failed: {str(e)}'}), 500

@app.route('/api/upload-folder', methods=['POST'])
def upload_folder():
    """Handle folder uploads from File System Access API"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        folder_name = request.form.get('folder_name', 'selected_folder')
        original_folder_name = request.form.get('original_folder_name', 'Unknown Folder')
        
        if not files or all(file.filename == '' for file in files):
            return jsonify({'error': 'No files selected'}), 400
        
        # Create upload directory
        upload_folder = 'uploaded_photos'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        
        # Create a unique folder path
        folder_path = os.path.join(upload_folder, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        uploaded_files = []
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}
        
        for file in files:
            if file and file.filename:
                # Check file extension
                filename = file.filename.lower()
                if any(filename.endswith(ext) for ext in allowed_extensions):
                    # Save file
                    file_path = os.path.join(folder_path, file.filename)
                    file.save(file_path)
                    uploaded_files.append(file.filename)
        
        if not uploaded_files:
            return jsonify({'error': 'No valid image files uploaded'}), 400
        
        # Return success response
        return jsonify({
            'success': True,
            'message': f'Successfully uploaded {len(uploaded_files)} files',
            'folder_path': folder_path,
            'uploaded_files': uploaded_files,
            'folder_name': folder_name
        })
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/process', methods=['POST'])
def process_folder():
    """Process selected folder with parameters"""
    data = request.json
    photos_dir = data.get('folder')
    pair_percentile = float(data.get('pair_percentile', 10))  # Now represents minimum matches, not percentile
    clear_brightness_min = float(data.get('clear_brightness_min', 120))
    clear_contrast_min = float(data.get('clear_contrast_min', 50))
    cloudy_brightness_max = float(data.get('cloudy_brightness_max', 60))
    cloudy_contrast_max = float(data.get('cloudy_contrast_max', 15))
    std_dev_multiplier = float(data.get('std_dev_multiplier', 2.0))
    mad_multiplier = float(data.get('mad_multiplier', 3.0))
    keypoint_percentile_bottom = float(data.get('keypoint_percentile_bottom', 5))
    keypoint_percentile_top = float(data.get('keypoint_percentile_top', 5))
    match_quality_threshold = float(data.get('match_quality_threshold', 30))
    start_pair = int(data.get('start_pair', 1))
    end_pair = int(data.get('end_pair', 41))
    filter_order = data.get('filter_order', ['sequence', 'gps_consistency', 'keypoint_percentile', 'percentile', 'std_dev', 'mad', 'cloudiness'])
    
    # Process in background thread
    def process_thread():
        try:
            print(f"🔄 Starting background processing for {photos_dir}")
            result = process_data(photos_dir, pair_percentile, 
                        clear_brightness_min, clear_contrast_min, cloudy_brightness_max, cloudy_contrast_max,
                        std_dev_multiplier, keypoint_percentile_bottom, 
                        match_quality_threshold, start_pair, end_pair)
            if result:
                print(f"✅ Background processing completed successfully for {photos_dir}")
            else:
                print(f"❌ Background processing failed for {photos_dir}")
                global_data['processing'] = False
                global_data['processed'] = False
        except Exception as e:
            print(f"❌ Exception in background processing: {e}")
            import traceback
            traceback.print_exc()
            global_data['processing'] = False
            global_data['processed'] = False
    
    thread = threading.Thread(target=process_thread)
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'processing'})

@app.route('/api/status')
def get_status():
    """Get processing status with progress information"""
    if global_data['processed']:
        return jsonify({
            'processed': True,
            'keypoint_count': len(global_data['raw_keypoints']),
            'pair_count': len(global_data['pair_results']),
            'progress': 100,
            'details': 'Complete!'
        })
    else:
        # Simulate progress based on current processing state
        progress = global_data.get('progress', 0)
        details = global_data.get('details', 'Processing...')
        return jsonify({
            'processed': False,
            'keypoint_count': 0,
            'pair_count': 0,
            'progress': progress,
            'details': details
        })



# @app.route('/api/reprocess', methods=['POST'])  # Removed - using dynamic filtering instead
def reprocess_data():
    """Re-process data with new parameters"""
    if not global_data['processed']:
        return jsonify({'error': 'No data processed'})
    
    try:
        params = request.get_json()
        
        # Apply new parameters to the existing data
        stationary_threshold = params.get('stationary_threshold', 0.1)
        pair_percentile = params.get('pair_percentile', 10)  # Now represents minimum matches, not percentile
        clear_brightness_min = params.get('clear_brightness_min', 120)
        clear_contrast_min = params.get('clear_contrast_min', 50)
        cloudy_brightness_max = params.get('cloudy_brightness_max', 60)
        cloudy_contrast_max = params.get('cloudy_contrast_max', 15)
        
        # Filter keypoints based on new stationary threshold using RAW unfiltered data
        raw_keypoints = global_data.get('raw_keypoints', global_data['all_keypoints'])
        filtered_keypoints = [kp for kp in raw_keypoints if kp['speed'] > stationary_threshold]
        print(f"Debug: After stationary filter: {len(filtered_keypoints)} keypoints (from {len(raw_keypoints)} raw keypoints)")
        
        # Re-analyze pair characteristics with new thresholds
        pair_characteristics = {}
        photos_dir = global_data.get('photos_dir', '')
        
        if photos_dir and os.path.exists(photos_dir):
            image_files = sorted([f for f in os.listdir(photos_dir) if f.endswith('.jpg')])
            total_pairs = len(image_files) - 1  # Load ALL possible pairs
            
            for i in range(total_pairs):
                pair_num = i + 1
                image1_path = os.path.join(photos_dir, image_files[i])
                image2_path = os.path.join(photos_dir, image_files[i + 1])
                
                # Analyze image characteristics with new thresholds
                img1_chars = analyze_image_characteristics(image1_path, clear_brightness_min, clear_contrast_min, 
                                                         cloudy_brightness_max, cloudy_contrast_max)
                img2_chars = analyze_image_characteristics(image2_path, clear_brightness_min, clear_contrast_min, 
                                                         cloudy_brightness_max, cloudy_contrast_max)
                
                avg_brightness = (img1_chars['brightness'] + img2_chars['brightness']) / 2
                avg_contrast = (img1_chars['contrast'] + img2_chars['contrast']) / 2
                
                if img1_chars['cloudiness'] == 'clear' and img2_chars['cloudiness'] == 'clear':
                    overall_cloudiness = 'clear'
                elif img1_chars['cloudiness'] == 'mostly cloudy' or img2_chars['cloudiness'] == 'mostly cloudy':
                    overall_cloudiness = 'mostly cloudy'
                else:
                    overall_cloudiness = 'partly cloudy'
                
                pair_characteristics[pair_num] = {
                    'brightness': avg_brightness,
                    'contrast': avg_contrast,
                    'cloudiness': overall_cloudiness,
                    'image1': os.path.basename(image1_path),
                    'image2': os.path.basename(image2_path)
                }
        
        print(f"Debug: Re-analyzed pair characteristics for {len(pair_characteristics)} pairs with new thresholds")
        
        # Apply percentile filtering to pairs
        pair_counts = {}
        for kp in filtered_keypoints:
            pair_num = kp['pair_num']
            pair_counts[pair_num] = pair_counts.get(pair_num, 0) + 1
        
        if pair_counts:
            keypoint_counts = list(pair_counts.values())
            # pair_percentile now means "minimum matches per pair" (e.g., 10 means keep pairs with at least 10 matches)
            min_matches = pair_percentile
            pairs_to_keep = [pair_num for pair_num, count in pair_counts.items() if count >= min_matches]
            
            # Safety check: ensure we keep at least 3 pairs or 50% of pairs, whichever is smaller
            min_pairs_to_keep = min(3, max(1, len(pair_counts) // 2))
            if len(pairs_to_keep) < min_pairs_to_keep and len(pair_counts) >= min_pairs_to_keep:
                # Sort pairs by keypoint count and keep top pairs
                sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
                pairs_to_keep = [pair_num for pair_num, count in sorted_pairs[:min_pairs_to_keep]]
                print(f"Debug: Safety check activated - keeping top {min_pairs_to_keep} pairs instead")
            
            print(f"Debug: Re-processing - min_matches={min_matches}, pairs_before={len(pair_counts)}, pairs_after={len(pairs_to_keep)}")
            
            # Filter keypoints to only include pairs above threshold
            final_keypoints = [kp for kp in filtered_keypoints if kp['pair_num'] in pairs_to_keep]
        else:
            final_keypoints = filtered_keypoints
            print(f"Debug: No pair counts available, using all {len(final_keypoints)} keypoints")
        
        if not final_keypoints:
            # Return empty data structure instead of error
            return jsonify({
                'histogram': {
                    'speeds': [],
                    'mean': 0,
                    'median': 0,
                    'std': 0
                },
                'boxplot': {
                    'speeds': []
                },
                'pairs': {
                    'pairs': [],
                    'means': [],
                    'stds': [],
                    'colors': []
                },
                'cumulative': {
                    'speeds': []
                }
            })
        
        # Recalculate statistics
        speeds = np.array([kp['speed'] for kp in final_keypoints])
        
        # Use existing pair results but filter out pairs with no keypoints
        pair_results = []
        for result in global_data['pair_results']:
            pair_num = result['pair_num']
            pair_keypoints = [kp for kp in final_keypoints if kp['pair_num'] == pair_num]
            
            if pair_keypoints:  # Only include pairs that have keypoints after filtering
                pair_speeds = [kp['speed'] for kp in pair_keypoints]
                pair_results.append({
                    'pair_num': pair_num,
                    'image1': result['image1'],
                    'image2': result['image2'],
                    'image1_path': result['image1_path'],
                    'image2_path': result['image2_path'],
                    'avg_speed': np.mean(pair_speeds),
                    'median_speed': np.median(pair_speeds),
                    'std_speed': np.std(pair_speeds),
                    'keypoint_count': len(pair_keypoints),
                    'time_diff': result['time_diff']
                })
        
        # Create new plot data
        plot_data = {
            'histogram': {
                'speeds': speeds.tolist(),
                'mean': float(np.mean(speeds)),
                'median': float(np.median(speeds)),
                'std': float(np.std(speeds))
            },
            'boxplot': {
                'speeds': speeds.tolist()
            },
            'pairs': {
                'pairs': [r['pair_num'] for r in pair_results],
                'means': [r['avg_speed'] for r in pair_results],
                'stds': [r['std_speed'] for r in pair_results],
                'colors': [get_current_cloudiness_color(r['pair_num'], pair_characteristics, current_filters) for r in pair_results]
            },
            'cumulative': {
                'speeds': speeds.tolist()
            }
        }
        
        plot_data['gps_enabled'] = global_data.get('gps_enabled', False)
        return jsonify(plot_data)
        
    except Exception as e:
        return jsonify({'error': f'Error re-processing data: {str(e)}'})

@app.route('/api/clear', methods=['POST'])
def clear_data():
    """Clear all existing data"""
    try:
        global_data.update({
            'raw_keypoints': [],
            'pair_results': [],
            'pair_characteristics': {},
            'photos_dir': None,
            'processed': False,
            'progress': 0,
            'details': 'Ready to process new data'
        })
        return jsonify({'status': 'cleared'})
    except Exception as e:
        return jsonify({'error': f'Error clearing data: {str(e)}'})

@app.route('/api/clear-cache/<path:folder_name>', methods=['POST'])
def clear_cache_endpoint(folder_name):
    """Clear cache for a specific folder"""
    try:
        # Find the full path to the folder
        base_dir = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.join(base_dir, folder_name)
        
        if not os.path.exists(folder_path):
            return jsonify({'error': 'Folder not found'}), 404
        
        success = clear_cache(folder_path)
        if success:
            return jsonify({'status': 'cache_cleared'})
        else:
            return jsonify({'status': 'no_cache_found'})
    except Exception as e:
        return jsonify({'error': f'Error clearing cache: {str(e)}'}), 500

@app.route('/api/clear-v2-cache', methods=['POST'])
def clear_v2_cache():
    """Clear all v2 cache files and v1 cache files"""
    try:
        cleared_count = 0
        
        # Clear v2 cache files
        if os.path.exists(CACHE_DIR):
            cache_files = [f for f in os.listdir(CACHE_DIR) if f.startswith('v2_cache_')]
            for cache_file in cache_files:
                cache_path = os.path.join(CACHE_DIR, cache_file)
                os.remove(cache_path)
                cleared_count += 1
        
        # Clear v1 cache files in photos directories
        for item in os.listdir('.'):
            if item.startswith('photos-') and os.path.isdir(item):
                photos_dir = item
                cache_file = os.path.join(photos_dir, f'.{photos_dir}_keypoints_cache.pkl')
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                    cleared_count += 1
                    print(f"🗑️ Cleared v1 cache: {cache_file}")
        
        # Clear in-memory data
        global processed_matches, cache_cleared_by_user
        processed_matches = []
        cache_cleared_by_user = True  # Set flag to prevent auto-loading
        
        return jsonify({
            'success': True, 
            'message': f'Cleared {cleared_count} cache files (v1 and v2) and in-memory data'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cache-status')
def cache_status():
    """Get cache status information"""
    try:
        cache_files = [f for f in os.listdir(CACHE_DIR) if f.startswith('v2_cache_')]
        cache_info = []
        
        for cache_file in cache_files:
            cache_path = os.path.join(CACHE_DIR, cache_file)
            cache_size = os.path.getsize(cache_path)
            cache_time = os.path.getmtime(cache_path)
            
            cache_info.append({
                'file': cache_file,
                'size_mb': round(cache_size / (1024 * 1024), 2),
                'created': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cache_time))
            })
        
        return jsonify({
            'success': True,
            'cache_files': cache_info,
            'total_files': len(cache_files)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/images/<path:folder_name>')
def get_images(folder_name):
    """Get list of images in a folder"""
    try:
        # Handle both relative and absolute paths
        if not os.path.exists(folder_name) or not os.path.isdir(folder_name):
            return jsonify({'error': f'Folder not found: {folder_name}'})
        
        images = sorted([f for f in os.listdir(folder_name) if f.endswith('.jpg')])
        print(f"Debug: Found {len(images)} images in folder: {folder_name}")
        return jsonify(images)
        
    except Exception as e:
        print(f"Debug: Error loading images from {folder_name}: {e}")
        return jsonify({'error': f'Error loading images: {str(e)}'})

@app.route('/api/analyze-reference', methods=['POST'])
def analyze_reference_image():
    """Analyze a reference image and return its characteristics"""
    try:
        data = request.get_json()
        image_path = data.get('image_path')
        image_type = data.get('type', 'unknown')
        
        if not image_path or not os.path.exists(image_path):
            return jsonify({'error': 'Image not found'})
        
        # Load and analyze the image
        image = cv2.imread(image_path, 0)  # Load as grayscale
        if image is None:
            return jsonify({'error': 'Could not load image'})
        
        # Calculate brightness and contrast
        brightness = np.mean(image)
        contrast = np.std(image)
        
        # Convert image to base64 for preview
        _, buffer = cv2.imencode('.jpg', image)
        image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'brightness': float(brightness),
            'contrast': float(contrast),
            'image_data': image_b64,
            'type': image_type
        })
        
    except Exception as e:
        return jsonify({'error': f'Error analyzing image: {str(e)}'})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create the HTML template
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ISS Speed Analysis Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        .controls {
            padding: 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
        }
        .control-row {
            display: flex;
            align-items: center;
            margin: 15px 0;
            gap: 20px;
            flex-wrap: wrap;
        }
        .control-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .control-group label {
            font-weight: bold;
            color: #495057;
            min-width: 120px;
            text-align: right;
        }
        .control-group input[type="range"] {
            width: 150px;
            margin: 0 10px;
        }
        .control-group input[type="number"] {
            width: 80px;
            padding: 5px 8px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            font-size: 14px;
        }
        .control-group select {
            padding: 8px 12px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            font-size: 14px;
            min-width: 200px;
        }
        .value-display {
            font-weight: bold;
            color: #007bff;
            min-width: 50px;
            text-align: center;
        }
        .btn {
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            margin: 10px 5px;
        }
        .btn:hover {
            background: #0056b3;
        }
        .btn:disabled {
            background: #6c757d;
            cursor: not-allowed;
        }
        .btn-select {
            background: #28a745;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            margin: 5px 0;
            min-width: 200px;
            text-align: left;
        }
        .btn-select:hover {
            background: #218838;
        }
        .gallery-content {
            max-width: 90vw;
            max-height: 90vh;
            width: 90vw;
            height: 90vh;
        }
        .gallery-container {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
            padding: 20px;
            max-height: calc(90vh - 100px);
            overflow-y: auto;
        }
        .gallery-item {
            border: 2px solid #dee2e6;
            border-radius: 8px;
            padding: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: center;
        }
        .gallery-item:hover {
            border-color: #007bff;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .gallery-item.selected {
            border-color: #28a745;
            background-color: #d4edda;
        }
        .gallery-item img {
            width: 100%;
            height: 150px;
            object-fit: cover;
            border-radius: 4px;
        }
        .gallery-item .image-name {
            margin-top: 8px;
            font-size: 12px;
            color: #495057;
            word-break: break-all;
        }
        .gallery-item .image-stats {
            margin-top: 5px;
            font-size: 11px;
            color: #6c757d;
        }
        .status {
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
            display: none;
        }
        .status.success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .status.warning {
            background: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
        }
        .gps-status {
            padding: 8px 12px;
            margin: 5px 0;
            border-radius: 4px;
            background: #e8f5e8;
            color: #155724;
            border: 1px solid #c3e6cb;
            font-weight: bold;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .status.info {
            background: #d1ecf1;
            color: #0c5460;
            border: 1px solid #bee5eb;
        }
        .image-preview {
            margin-top: 10px;
            max-width: 150px;
            max-height: 100px;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            overflow: hidden;
        }
        .image-preview img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .reference-info {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            border: 1px solid #dee2e6;
        }
        .reference-info h4 {
            margin: 0 0 10px 0;
            color: #495057;
        }
        .reference-info p {
            margin: 0;
            color: #6c757d;
            font-size: 0.9em;
            line-height: 1.4;
        }
        .plots {
            padding: 20px;
        }
        .plot-container {
            margin: 20px 0;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            overflow: hidden;
        }
        .plot-title {
            background: #e9ecef;
            padding: 10px 15px;
            margin: 0;
            font-size: 16px;
            font-weight: bold;
        }
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.8);
        }
        .modal-content {
            background-color: white;
            margin: 5% auto;
            padding: 20px;
            border-radius: 8px;
            width: 90%;
            max-width: 1200px;
            max-height: 80vh;
            overflow-y: auto;
        }
        .close {
            color: #aaa;
            float: right;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }
        .close:hover {
            color: black;
        }
        .image-container {
            display: flex;
            gap: 20px;
            margin: 20px 0;
        }
        .image-box {
            flex: 1;
            text-align: center;
        }
        .image-box img {
            max-width: 100%;
            height: auto;
            border: 1px solid #dee2e6;
            border-radius: 4px;
        }
        .pair-info {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            margin: 20px 0;
        }
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
        }
        .info-item {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #dee2e6;
        }
        .info-label {
            font-weight: bold;
            color: #495057;
        }
        .info-value {
            color: #6c757d;
        }
        .keypoint-matches {
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 4px;
            border: 1px solid #dee2e6;
        }
        .keypoint-matches h3 {
            margin: 0 0 10px 0;
            color: #495057;
        }
        .keypoint-matches p {
            margin: 0 0 15px 0;
            color: #6c757d;
            font-style: italic;
        }
        .speedometer-container {
            margin: 20px 0;
            display: flex;
            justify-content: space-around;
            gap: 15px;
        }
        .speedometer {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            flex: 1;
            min-width: 200px;
        }
        .speedometer h3 {
            margin: 0 0 15px 0;
            font-size: 1.1em;
            font-weight: 300;
        }
        .count-display {
            font-size: 0.9em;
            opacity: 0.8;
            margin-top: 8px;
        }
        .count-unit {
            font-size: 0.8em;
        }
        .mode-display {
            font-size: 0.9em;
            opacity: 0.8;
            margin-top: 8px;
        }
        .mode-selector select {
            width: 100%;
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 6px;
            background: white;
            font-size: 0.9em;
        }
        .speed-display {
            margin: 15px 0;
        }
        .speed-display span:first-child {
            font-size: 2.5em;
            font-weight: bold;
            display: block;
        }
        .speed-unit {
            font-size: 1.2em;
            opacity: 0.8;
        }
        .accuracy-display {
            margin: 15px 0;
        }
        .accuracy-display #currentAccuracy {
            font-size: 2em;
            font-weight: bold;
            display: block;
        }
        .accuracy-unit {
            font-size: 1em;
            opacity: 0.8;
        }
        .progress-container {
            margin: 20px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }
        .progress-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .progress-header h3 {
            margin: 0;
            color: #495057;
        }
        .progress-header span {
            font-weight: bold;
            color: #007bff;
            font-size: 1.1em;
        }
        .progress-bar {
            width: 100%;
            height: 20px;
            background: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 10px;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #007bff, #0056b3);
            width: 0%;
            transition: width 0.3s ease;
            border-radius: 10px;
        }
        .progress-details {
            font-size: 0.9em;
            color: #6c757d;
            text-align: center;
        }
        
        /* New Filter Design Styles */
        .filter-row {
            margin: 15px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #e9ecef;
            cursor: move;
            transition: all 0.2s ease;
        }
        
        .filter-row:hover {
            background: #e9ecef;
            border-color: #007bff;
        }
        
        .filter-row.dragging {
            opacity: 0.5;
            transform: rotate(2deg);
        }
        
        .filter-row.drag-over {
            border-color: #28a745;
            background: #d4edda;
        }
        
        .drag-handle {
            width: 20px;
            height: 20px;
            margin-right: 10px;
            cursor: grab;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #6c757d;
        }
        
        .drag-handle:active {
            cursor: grabbing;
        }
        
        .drag-handle::before {
            content: "⋮⋮";
            font-size: 12px;
            line-height: 1;
        }
        
        .filter-control {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .toggle-container {
            flex-shrink: 0;
        }
        
        .toggle {
            display: none;
        }
        
        .toggle-label {
            display: block;
            width: 50px;
            height: 25px;
            background: #ccc;
            border-radius: 25px;
            position: relative;
            cursor: pointer;
            transition: background 0.3s;
        }
        
        .toggle-label:before {
            content: '';
            position: absolute;
            width: 21px;
            height: 21px;
            border-radius: 50%;
            background: white;
            top: 2px;
            left: 2px;
            transition: transform 0.3s;
        }
        
        .toggle:checked + .toggle-label {
            background: #007bff;
        }
        
        .toggle:checked + .toggle-label:before {
            transform: translateX(25px);
        }
        
        .filter-content {
            flex: 1;
            min-width: 0;
        }
        
        .filter-content label {
            display: block;
            font-size: 0.9em;
            font-weight: 600;
            color: #495057;
            margin-bottom: 8px;
        }
        
        .filter-description {
            flex: 1;
            font-size: 0.8em;
            color: #6c757d;
            font-style: italic;
            text-align: right;
            max-width: 200px;
        }
        
        .filter-parameter {
            margin-top: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .filter-parameter label {
            font-size: 0.8em;
            color: #495057;
            font-weight: 500;
        }
        
        .filter-parameter input[type="number"] {
            padding: 4px 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 0.8em;
        }
        
        .range-separator {
            margin: 0 10px;
            color: #6c757d;
            font-weight: 500;
        }
        
        .checkbox-container {
            display: flex;
            gap: 20px;
            margin-top: 8px;
        }
        
        .checkbox-label {
            display: flex;
            align-items: center;
            cursor: pointer;
            font-size: 0.85em;
        }
        
        .checkbox-label input[type="checkbox"] {
            display: none;
        }
        
        .checkmark {
            width: 18px;
            height: 18px;
            border: 2px solid #ccc;
            border-radius: 3px;
            margin-right: 8px;
            position: relative;
            transition: all 0.3s;
        }
        
        .checkbox-label input[type="checkbox"]:checked + .checkmark {
            background: #007bff;
            border-color: #007bff;
        }
        
        .checkbox-label input[type="checkbox"]:checked + .checkmark:after {
            content: '';
            position: absolute;
            left: 5px;
            top: 2px;
            width: 4px;
            height: 8px;
            border: solid white;
            border-width: 0 2px 2px 0;
            transform: rotate(45deg);
        }
        
        .checkbox-label input[type="checkbox"]:disabled + .checkmark {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .slider-container input[type="range"]:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .value-display {
            font-size: 0.85em;
            font-weight: 600;
            color: #007bff;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 ISS Speed Analysis Dashboard</h1>
            <p>Interactive analysis of ISS speed from image pairs</p>
        </div>
        
        <div class="controls">
            <!-- Folder Selection Row -->
            <div class="control-row">
                <div class="control-group">
                    <label for="folderSelect">Photos Folder:</label>
                    <select id="folderSelect">
                        <option value="">Select a folder...</option>
                    </select>
                </div>
            </div>
            
            <!-- Sequence Range Filter (Always First) -->
            <div class="filter-row" data-filter-order="1">
                <div class="drag-handle" title="Drag to reorder (Sequence filter is always first)"></div>
                <div class="filter-control">
                    <div class="toggle-container">
                        <input type="checkbox" id="enableSequence" class="toggle" checked>
                        <label for="enableSequence" class="toggle-label"></label>
                    </div>
                    <div class="filter-content">
                        <label for="startPair">Image Sequence Range</label>
                        <div class="slider-container">
                            <input type="range" id="startPair" min="1" max="100" value="1" step="1">
                            <span class="value-display" id="startPairValue">1</span>
                            <span class="range-separator">to</span>
                            <input type="range" id="endPair" min="1" max="100" value="41" step="1">
                            <span class="value-display" id="endPairValue">41</span>
                        </div>
                    </div>
                    <div class="filter-description">
                        Analyze only keypoints from pairs within this range
                    </div>
                </div>
            </div>
            
            <!-- Speed Calculation Mode -->
            <div class="filter-row" data-filter-order="0">
                <div class="drag-handle" title="Speed calculation method"></div>
                <div class="filter-control">
                    <div class="filter-content">
                        <label for="speedMode">Speed Calculation Mode</label>
                        <div class="mode-selector">
                            <select id="speedMode">
                                <option value="gps_only">GPS-Only (Most Accurate)</option>
                                <option value="constant_gsd">Constant GSD (ESA Default)</option>
                                <option value="dynamic_gsd" selected>Dynamic GSD (GPS + Keypoints)</option>
                            </select>
                        </div>
                    </div>
                    <div class="filter-description">
                        Choose how to calculate ISS speed from the data
                    </div>
                </div>
            </div>
            
            <!-- GPS Consistency Filter -->
            <div class="filter-row" data-filter-order="1">
                <div class="drag-handle" title="Drag to reorder"></div>
                <div class="filter-control">
                    <div class="toggle-container">
                        <input type="checkbox" id="enableGpsConsistency" class="toggle">
                        <label for="enableGpsConsistency" class="toggle-label"></label>
                    </div>
                    <div class="filter-content">
                        <label for="gpsToleranceConstant">GPS Consistency Filter</label>
                        <div class="filter-parameter" id="gpsConsistencyParam" style="display: none;">
                            <label for="gpsToleranceConstant">Tolerance (km/s):</label>
                            <input type="number" id="gpsToleranceConstant" value="0.783" min="0" max="5" step="0.001" style="width: 80px;">
                        </div>
                    </div>
                    <div class="filter-description">
                        Remove keypoints with Constant GSD speeds outside GPS-only speed range (GPS data required)
                    </div>
                </div>
            </div>
            
            <!-- Keypoint Percentile Filter -->
            <div class="filter-row" data-filter-order="3">
                <div class="drag-handle" title="Drag to reorder"></div>
                <div class="filter-control">
                    <div class="toggle-container">
                        <input type="checkbox" id="enableKeypointPercentile" class="toggle">
                        <label for="enableKeypointPercentile" class="toggle-label"></label>
                    </div>
                    <div class="filter-content">
                        <label for="keypointPercentile">Keypoint Percentile Filter</label>
                        <div class="slider-container">
                            <input type="range" id="keypointPercentile" min="0" max="50" value="5" step="1" disabled>
                            <span class="value-display" id="keypointPercentileValue">5</span>
                        </div>
                    </div>
                    <div class="filter-description">
                        Remove bottom and top X% of keypoints by speed (outlier removal)
                    </div>
                </div>
            </div>
            
            <!-- Pair Percentile Filter -->
            <div class="filter-row" data-filter-order="4">
                <div class="drag-handle" title="Drag to reorder"></div>
                <div class="filter-control">
                    <div class="toggle-container">
                        <input type="checkbox" id="enablePercentile" class="toggle">
                        <label for="enablePercentile" class="toggle-label"></label>
                    </div>
                    <div class="filter-content">
                        <label for="pairPercentile">Pair Percentile Filter</label>
                        <div class="slider-container">
                            <input type="range" id="pairPercentile" min="0" max="50" value="30" step="5" disabled>
                            <span class="value-display" id="pairPercentileValue">30</span>
                        </div>
                    </div>
                    <div class="filter-description">
                        Remove bottom X% of pairs with fewest keypoints
                    </div>
                </div>
            </div>
            
            <!-- Standard Deviation Filter -->
            <div class="filter-row" data-filter-order="5">
                <div class="drag-handle" title="Drag to reorder"></div>
                <div class="filter-control">
                    <div class="toggle-container">
                        <input type="checkbox" id="enableStdDev" class="toggle">
                        <label for="enableStdDev" class="toggle-label"></label>
                    </div>
                    <div class="filter-content">
                        <label for="stdDevMultiplier">Standard Deviation Filter</label>
                        <div class="slider-container">
                            <input type="range" id="stdDevMultiplier" min="0" max="3" value="2" step="0.1" disabled>
                            <span class="value-display" id="stdDevMultiplierValue">2.0</span>
                        </div>
                    </div>
                    <div class="filter-description">
                        Remove outliers beyond X standard deviations from mean
                    </div>
                </div>
            </div>
            
            <!-- Cloudiness Filter -->
            <div class="filter-row" data-filter-order="6">
                <div class="drag-handle" title="Drag to reorder"></div>
                <div class="filter-control">
                    <div class="toggle-container">
                        <input type="checkbox" id="enableCloudiness" class="toggle">
                        <label for="enableCloudiness" class="toggle-label"></label>
                    </div>
                    <div class="filter-content">
                        <label>Cloudiness Filter</label>
                        <div class="checkbox-container">
                            <label class="checkbox-label">
                                <input type="checkbox" id="includePartlyCloudy" disabled>
                                <span class="checkmark"></span>
                                Include Partly Cloudy
                            </label>
                            <label class="checkbox-label">
                                <input type="checkbox" id="includeMostlyCloudy" disabled>
                                <span class="checkmark"></span>
                                Include Mostly Cloudy
                            </label>
                        </div>
                    </div>
                    <div class="filter-description">
                        Select which cloudiness categories to include in analysis
                    </div>
                </div>
            </div>
            
            <!-- Image Classification Thresholds -->
            <div style="background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 15px; margin: 15px 0;">
                <h4 style="margin: 0 0 15px 0; color: #495057;">Image Classification Thresholds</h4>
                <div style="display: flex; gap: 20px; margin-bottom: 10px;">
                    <div style="flex: 1;">
                        <label style="display: block; margin-bottom: 5px; font-size: 0.9em; color: #6c757d;">Clear Min Brightness:</label>
                        <input type="range" id="clearBrightnessMin" min="50" max="200" value="120" step="5" style="width: 100%;">
                        <span id="clearBrightnessMinValue" style="font-size: 0.8em; color: #495057;">120</span>
                    </div>
                    <div style="flex: 1;">
                        <label style="display: block; margin-bottom: 5px; font-size: 0.9em; color: #6c757d;">Clear Min Contrast:</label>
                        <input type="range" id="clearContrastMin" min="10" max="100" value="50" step="5" style="width: 100%;">
                        <span id="clearContrastMinValue" style="font-size: 0.8em; color: #495057;">50</span>
                    </div>
                </div>
                <div style="display: flex; gap: 20px;">
                    <div style="flex: 1;">
                        <label style="display: block; margin-bottom: 5px; font-size: 0.9em; color: #6c757d;">Cloudy Max Brightness:</label>
                        <input type="range" id="cloudyBrightnessMax" min="30" max="150" value="60" step="5" style="width: 100%;">
                        <span id="cloudyBrightnessMaxValue" style="font-size: 0.8em; color: #495057;">60</span>
                    </div>
                    <div style="flex: 1;">
                        <label style="display: block; margin-bottom: 5px; font-size: 0.9em; color: #6c757d;">Cloudy Max Contrast:</label>
                        <input type="range" id="cloudyContrastMax" min="5" max="80" value="15" step="5" style="width: 100%;">
                        <span id="cloudyContrastMaxValue" style="font-size: 0.8em; color: #495057;">15</span>
                    </div>
                </div>
            </div>
            
            <!-- Action Buttons Row -->
            <div class="control-row">
                <button class="btn" id="loadBtn" onclick="loadData()">Load Data</button>
                <button class="btn" id="defaultBtn" onclick="resetToDefaults()">Reset to Defaults</button>
                <button class="btn" id="clearCacheBtn" onclick="clearCache()" style="background-color: #ff6b6b;">Clear Cache</button>
            </div>
        </div>
        
        <div id="status" class="status"></div>
        <div id="gpsStatus" class="gps-status" style="display: none;">
            <span id="gpsIcon">🛰️</span>
            <span id="gpsText">GPS Enhanced Mode</span>
        </div>
        
        <div id="progressContainer" class="progress-container" style="display: none;">
            <div class="progress-header">
                <h3>Processing Data...</h3>
                <span id="progressText">0%</span>
            </div>
            <div class="progress-bar">
                <div id="progressBar" class="progress-fill"></div>
            </div>
            <div id="progressDetails" class="progress-details"></div>
        </div>
        
        <div class="speedometer-container">
            <div class="speedometer">
                <h3>Estimated ISS Speed</h3>
                <div class="speed-display">
                    <span id="currentSpeed">--</span>
                    <span class="speed-unit">km/s</span>
                </div>
                <div class="mode-display">
                    <span id="currentMode">--</span>
                </div>
            </div>
        </div>
        
        <div class="plots">
            <div class="plot-container">
                <h3 class="plot-title">Speed Distribution Histogram</h3>
                <div id="histogram"></div>
            </div>
            
            <div class="plot-container">
                <h3 class="plot-title">Average Speed per Image Pair (Click legend to toggle categories, click points to view images)</h3>
                <div id="pairs"></div>
            </div>
        </div>
    </div>
    
    <!-- Modal for displaying pair images -->
    <div id="imageModal" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeModal()">&times;</span>
            <h2 id="modalTitle">Pair Details</h2>
            <div id="modalContent"></div>
        </div>
    </div>
    
    
    <script>
        let plotData = null;
        let updatingSequenceRange = false;
        let userModifiedEndPair = false;
        let draggedElement = null;
        
        function setupDragAndDrop() {
            const filterRows = document.querySelectorAll('.filter-row');
            
            filterRows.forEach(row => {
                // Skip the sequence filter (always first)
                if (row.dataset.filterOrder === '1') {
                    return;
                }
                
                row.draggable = true;
                
                row.addEventListener('dragstart', function(e) {
                    draggedElement = this;
                    this.classList.add('dragging');
                    e.dataTransfer.effectAllowed = 'move';
                });
                
                row.addEventListener('dragend', function(e) {
                    this.classList.remove('dragging');
                    draggedElement = null;
                });
                
                row.addEventListener('dragover', function(e) {
                    e.preventDefault();
                    e.dataTransfer.dropEffect = 'move';
                    this.classList.add('drag-over');
                });
                
                row.addEventListener('dragleave', function(e) {
                    this.classList.remove('drag-over');
                });
                
                row.addEventListener('drop', function(e) {
                    e.preventDefault();
                    this.classList.remove('drag-over');
                    
                    if (draggedElement && draggedElement !== this) {
                        // Get the container
                        const container = this.parentNode;
                        
                        // Insert the dragged element before this element
                        container.insertBefore(draggedElement, this);
                        
                        // Update the filter order attributes and get the new order
                        const newOrder = updateFilterOrder();
                        
                        // Store the filter order for sending to backend
                        window.currentFilterOrder = newOrder;
                        
                        // Show status message
                        showStatus('Filter order updated', 'success');
                    }
                });
            });
        }
        
        function updateFilterOrder() {
            const filterRows = document.querySelectorAll('.filter-row');
            const filterOrder = [];
            filterRows.forEach((row, index) => {
                row.dataset.filterOrder = index + 1;
                // Get the filter type from the row
                const filterType = getFilterTypeFromRow(row);
                if (filterType) {
                    filterOrder.push(filterType);
                }
            });
            return filterOrder;
        }
        
        function getFilterTypeFromRow(row) {
            // Map filter rows to their types based on the enable checkbox IDs
            const enableCheckbox = row.querySelector('input[type="checkbox"]');
            if (!enableCheckbox) return null;
            
            const checkboxId = enableCheckbox.id;
            switch(checkboxId) {
                case 'enableSequence': return 'sequence';
                case 'enableKeypointPercentile': return 'keypoint_percentile';
                case 'enablePercentile': return 'percentile';
                case 'enableStdDev': return 'std_dev';
                case 'enableCloudiness': return 'cloudiness';
                case 'enableGpsConsistency': return 'gps_consistency';
                default: return null;
            }
        }
        
        // Load folders on page load
        window.onload = function() {
            loadFolders();
            setupSliders();
            setupDragAndDrop();
            
            // Initialize filter order
            window.currentFilterOrder = updateFilterOrder();
            
            // Enable sequence range filter by default
            document.getElementById('enableSequence').checked = true;
            toggleFilterControls('sequence', true);
            
            // Add folder change detection
            document.getElementById('folderSelect').addEventListener('change', function() {
                if (plotData) {
                    showStatus('Folder changed - data will be cleared when you load', 'info');
                }
            });
        };
        
        function setupSliders() {
            // Setup toggle switches
            const toggles = [
                'enablePercentile', 'enableStdDev', 'enableKeypointPercentile', 'enableSequence', 'enableCloudiness'
            ];
            
            toggles.forEach(toggleId => {
                const toggle = document.getElementById(toggleId);
                toggle.addEventListener('change', function() {
                    const filterType = toggleId.replace('enable', '').toLowerCase();
                    toggleFilterControls(filterType, this.checked);
                    
                    // Update button text to indicate refresh is needed
                    if (plotData) {
                        updateLoadButton('Refresh Data');
                    }
                });
            });
            
            // Setup slider event listeners
            const sliders = [
                'pairPercentile', 'stdDevMultiplier', 'keypointPercentile', 'startPair', 'endPair',
                'clearBrightnessMin', 'clearContrastMin', 'cloudyBrightnessMax', 'cloudyContrastMax'
            ];
            
            sliders.forEach(sliderId => {
                const slider = document.getElementById(sliderId);
                const valueDisplay = document.getElementById(sliderId + 'Value');
                
                slider.addEventListener('input', function() {
                    // Format display values appropriately
                    if (sliderId === 'minSpeed' || sliderId === 'maxSpeed' || sliderId === 'stdDevMultiplier') {
                        valueDisplay.textContent = parseFloat(this.value).toFixed(1);
                    } else {
                        valueDisplay.textContent = this.value;
                    }
                    
                    // Update button text to indicate refresh is needed
                    if (plotData) {
                        updateLoadButton('Refresh Data');
                    }
                });
            });
            
            // Special handling for pair range sliders
            const startPairSlider = document.getElementById('startPair');
            const endPairSlider = document.getElementById('endPair');
            
            startPairSlider.addEventListener('input', function() {
                const startValue = parseInt(this.value);
                const endValue = parseInt(endPairSlider.value);
                if (startValue > endValue) {
                    endPairSlider.value = startValue;
                    document.getElementById('endPairValue').textContent = startValue;
                }
            });
            
            endPairSlider.addEventListener('input', function() {
                const startValue = parseInt(startPairSlider.value);
                const endValue = parseInt(this.value);
                if (endValue < startValue) {
                    startPairSlider.value = endValue;
                    document.getElementById('startPairValue').textContent = endValue;
                }
                // Mark that user has manually modified the end pair slider
                userModifiedEndPair = true;
            });
            
            // Setup cloudiness checkboxes
            const cloudinessCheckboxes = ['includePartlyCloudy', 'includeMostlyCloudy'];
            cloudinessCheckboxes.forEach(checkboxId => {
                const checkbox = document.getElementById(checkboxId);
                checkbox.addEventListener('change', function() {
                    if (plotData) {
                        updateLoadButton('Refresh Data');
                    }
                });
            });
        }
        
        function toggleFilterControls(filterType, enabled) {
            let controls = [];
            
            switch(filterType) {
                case 'percentile':
                    controls = ['pairPercentile'];
                    break;
                case 'stddev':
                    controls = ['stdDevMultiplier'];
                    break;
                case 'keypointpercentile':
                    controls = ['keypointPercentile'];
                    break;
                case 'sequence':
                    controls = ['startPair', 'endPair'];
                    break;
                case 'cloudiness':
                    controls = ['includePartlyCloudy', 'includeMostlyCloudy'];
                    break;
            }
            
            controls.forEach(controlId => {
                const control = document.getElementById(controlId);
                if (control) {
                    control.disabled = !enabled;
                }
            });
        }
        
        function updateLoadButton(text) {
            const loadBtn = document.getElementById('loadBtn');
            loadBtn.textContent = text;
        }
        
        function updateGpsConsistencyAvailability(gpsEnabled) {
            const gpsConsistencyToggle = document.getElementById('enableGpsConsistency');
            const gpsConsistencyRow = gpsConsistencyToggle.closest('.filter-row');
            const paramDiv = document.getElementById('gpsConsistencyParam');
            
            if (gpsEnabled) {
                gpsConsistencyRow.style.opacity = '1';
                gpsConsistencyToggle.disabled = false;
                gpsConsistencyRow.querySelector('.filter-description').textContent = 
                    'Remove keypoints with Constant GSD speeds outside GPS-only speed range';
                // Show parameter if toggle is checked
                if (gpsConsistencyToggle.checked) {
                    paramDiv.style.display = 'block';
                }
            } else {
                gpsConsistencyRow.style.opacity = '0.5';
                gpsConsistencyToggle.disabled = true;
                gpsConsistencyToggle.checked = false;
                paramDiv.style.display = 'none';
                gpsConsistencyRow.querySelector('.filter-description').textContent = 
                    'GPS data required - not available for this dataset';
            }
        }
        
        function updateSequenceRangeLimits(maxPairs) {
            if (updatingSequenceRange) {
                return;
            }
            
            updatingSequenceRange = true;
            
            const startPairSlider = document.getElementById('startPair');
            const endPairSlider = document.getElementById('endPair');
            
            if (!startPairSlider || !endPairSlider) {
                updatingSequenceRange = false;
                return;
            }
            
            // Update max values
            startPairSlider.max = maxPairs;
            endPairSlider.max = maxPairs;
            
            // Only set default value if it's the first time (no user interaction yet)
            if (!userModifiedEndPair && (endPairSlider.value === endPairSlider.max || endPairSlider.value > maxPairs)) {
                // Set to 41 or maxPairs, whichever is smaller
                const defaultEnd = Math.min(41, maxPairs);
                endPairSlider.value = defaultEnd;
                document.getElementById('endPairValue').textContent = defaultEnd;
            }
            
            // If user has set a value higher than the new max, cap it
            if (endPairSlider.value > maxPairs) {
                endPairSlider.value = maxPairs;
                document.getElementById('endPairValue').textContent = maxPairs;
            }
            
            // Sequence range limits updated
            
            updatingSequenceRange = false;
        }
        
        function resetToDefaults() {
            // Reset toggles - sequence range ON by default, others OFF
            const toggles = [
                'enablePercentile', 'enableStdDev', 'enableKeypointPercentile', 'enableCloudiness'
            ];
            
            toggles.forEach(toggleId => {
                const toggle = document.getElementById(toggleId);
                toggle.checked = false;
                const filterType = toggleId.replace('enable', '').toLowerCase();
                toggleFilterControls(filterType, false);
            });
            
            // Enable sequence range by default
            document.getElementById('enableSequence').checked = true;
            
            // Add speed mode change listener
            document.getElementById('speedMode').addEventListener('change', function() {
                updateSpeedometer();
                // Refresh data to update graphs with new speed mode
                if (plotData) {
                    refreshData();
                }
            });
            
            // GPS Consistency Filter toggle
            document.getElementById('enableGpsConsistency').addEventListener('change', function() {
                const paramDiv = document.getElementById('gpsConsistencyParam');
                if (this.checked) {
                    paramDiv.style.display = 'block';
                } else {
                    paramDiv.style.display = 'none';
                }
                
                // Refresh data if plotData exists
                if (plotData) {
                    refreshData();
                }
            });
            
            // GPS Tolerance Constant input
            document.getElementById('gpsToleranceConstant').addEventListener('input', function() {
                // Refresh data if plotData exists
                if (plotData) {
                    refreshData();
                }
            });
            toggleFilterControls('sequence', true);
            
            // Reset slider values to defaults
            document.getElementById('pairPercentile').value = 30;
            document.getElementById('pairPercentileValue').textContent = '30';
            
            document.getElementById('stdDevMultiplier').value = 2;
            document.getElementById('stdDevMultiplierValue').textContent = '2.0';
            
            document.getElementById('keypointPercentile').value = 5;
            document.getElementById('keypointPercentileValue').textContent = '5';
            
            document.getElementById('startPair').value = 1;
            document.getElementById('startPairValue').textContent = '1';
            
            document.getElementById('endPair').value = 41;
            document.getElementById('endPairValue').textContent = '41';
            
            // Reset cloudiness checkboxes to include both by default
            document.getElementById('includePartlyCloudy').checked = true;
            document.getElementById('includeMostlyCloudy').checked = true;
            
            // Reset threshold sliders to defaults
            document.getElementById('clearBrightnessMin').value = 120;
            document.getElementById('clearBrightnessMinValue').textContent = '120';
            
            document.getElementById('clearContrastMin').value = 50;
            document.getElementById('clearContrastMinValue').textContent = '50';
            
            document.getElementById('cloudyBrightnessMax').value = 60;
            document.getElementById('cloudyBrightnessMaxValue').textContent = '60';
            
            document.getElementById('cloudyContrastMax').value = 15;
            document.getElementById('cloudyContrastMaxValue').textContent = '15';
            
            // Reset user modification flag
            userModifiedEndPair = false;
            
            // Update button text if data is loaded
            if (plotData) {
                updateLoadButton('Refresh Data');
            }
        }
        
        async function clearCache() {
            const folder = document.getElementById('folderSelect').value;
            if (!folder) {
                showStatus('Please select a folder first', 'error');
                return;
            }
            
            if (!confirm('Are you sure you want to clear the cache for this folder? This will force re-processing of all images next time.')) {
                return;
            }
            
            try {
                showStatus('Clearing cache...', 'info');
                const response = await fetch(`/api/clear-cache/${encodeURIComponent(folder)}`, {
                    method: 'POST'
                });
                
                if (response.ok) {
                    showStatus('Cache cleared successfully', 'success');
                } else {
                    showStatus('Failed to clear cache', 'error');
                }
            } catch (error) {
                showStatus('Error clearing cache: ' + error.message, 'error');
            }
        }
        
        async function loadFolders() {
            try {
                const response = await fetch('/api/folders');
                const folders = await response.json();
                
                const select = document.getElementById('folderSelect');
                select.innerHTML = '<option value="">Select a folder...</option>';
                
                folders.forEach(folder => {
                    const option = document.createElement('option');
                    option.value = folder.name;
                    option.textContent = `${folder.name} (${folder.count} images)`;
                    select.appendChild(option);
                });
                
                
            } catch (error) {
                showStatus('Error loading folders: ' + error.message, 'error');
            }
        }
        
        
        async function loadData() {
            const folder = document.getElementById('folderSelect').value;
            if (!folder) {
                showStatus('Please select a folder', 'error');
                return;
            }
            
            const isRefresh = plotData !== null;
            
            if (!isRefresh) {
                // Clear existing data and reset UI for initial load
                await clearExistingData();
            }
            
            const params = {
                folder: folder,
                // Only include enabled filters
                enable_percentile: document.getElementById('enablePercentile').checked,
                pair_percentile: parseFloat(document.getElementById('pairPercentile').value),
                filter_order: window.currentFilterOrder || ['sequence', 'keypoint_percentile', 'percentile', 'std_dev', 'cloudiness', 'gps_consistency'],
                enable_std_dev: document.getElementById('enableStdDev').checked,
                std_dev_multiplier: parseFloat(document.getElementById('stdDevMultiplier').value),
                enable_keypoint_percentile: document.getElementById('enableKeypointPercentile').checked,
                keypoint_percentile: parseFloat(document.getElementById('keypointPercentile').value),
                enable_sequence: document.getElementById('enableSequence').checked,
                start_pair: parseInt(document.getElementById('startPair').value),
                end_pair: parseInt(document.getElementById('endPair').value),
                enable_cloudiness: document.getElementById('enableCloudiness').checked,
                include_partly_cloudy: document.getElementById('includePartlyCloudy').checked,
                include_mostly_cloudy: document.getElementById('includeMostlyCloudy').checked,
                clear_brightness_min: parseFloat(document.getElementById('clearBrightnessMin').value),
                clear_contrast_min: parseFloat(document.getElementById('clearContrastMin').value),
                cloudy_brightness_max: parseFloat(document.getElementById('cloudyBrightnessMax').value),
                cloudy_contrast_max: parseFloat(document.getElementById('cloudyContrastMax').value),
                enable_gps_consistency: document.getElementById('enableGpsConsistency').checked,
                gps_tolerance_constant: parseFloat(document.getElementById('gpsToleranceConstant').value),
                speed_mode: document.getElementById('speedMode').value
            };
            
            const actionText = isRefresh ? 'Refreshing data with new parameters...' : 'Starting data processing...';
            showStatus(actionText, 'info');
            showProgress(true);
            updateProgress(0, 'Initializing...');
            document.getElementById('loadBtn').disabled = true;
            
            // Reset polling counter for new load
            pollingAttempts = 0;
            
            try {
                const response = await fetch('/api/process', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(params)
                });
                
                if (response.ok) {
                    // Poll for completion with progress updates
                    pollStatusWithProgress();
                } else {
                    throw new Error('Failed to start processing');
                }
            } catch (error) {
                showStatus('Error: ' + error.message, 'error');
                showProgress(false);
                document.getElementById('loadBtn').disabled = false;
            }
        }
        
        async function pollStatus() {
            try {
                const response = await fetch('/api/status');
                const status = await response.json();
                
                if (status.processed) {
                    showStatus(`Data loaded: ${status.keypoint_count} keypoints from ${status.pair_count} pairs`, 'success');
                    document.getElementById('loadBtn').disabled = false;
                    await refreshData();
                } else {
                    setTimeout(pollStatus, 1000);
                }
            } catch (error) {
                showStatus('Error checking status: ' + error.message, 'error');
                document.getElementById('loadBtn').disabled = false;
            }
        }
        
        let pollingAttempts = 0;
        const maxPollingAttempts = 120; // 60 seconds max (120 * 500ms)
        
        async function pollStatusWithProgress() {
            try {
                pollingAttempts++;
                
                if (pollingAttempts > maxPollingAttempts) {
                    showStatus('Processing timeout - please try again', 'error');
                    showProgress(false);
                    document.getElementById('loadBtn').disabled = false;
                    return;
                }
                
                const response = await fetch('/api/status');
                const status = await response.json();
                
                if (status.processed) {
                    updateProgress(100, 'Complete!');
                    showStatus(`Data loaded: ${status.keypoint_count} keypoints from ${status.pair_count} pairs`, 'success');
                    showProgress(false);
                    document.getElementById('loadBtn').disabled = false;
                    updateLoadButton('Refresh Data');
                    
                    // Update sequence range limits based on actual pair count
                    setTimeout(() => {
                        updateSequenceRangeLimits(status.pair_count);
                    }, 100);
                    
                    // Load the plot data without triggering another processing cycle
                    await refreshData();
                } else {
                    // Update progress based on current status
                    const progress = status.progress || 0;
                    const details = status.details || 'Processing...';
                    updateProgress(progress, details);
                    setTimeout(pollStatusWithProgress, 500);
                }
            } catch (error) {
                showStatus('Error checking status: ' + error.message, 'error');
                showProgress(false);
                document.getElementById('loadBtn').disabled = false;
            }
        }
        
        async function refreshData() {
            try {
                console.log('Refreshing data with current parameters...');
                console.log('GPS Consistency enabled:', document.getElementById('enableGpsConsistency').checked);
                console.log('GPS Tolerance constant:', document.getElementById('gpsToleranceConstant').value);
                console.log('Current filter order:', window.currentFilterOrder);
                console.log('Default filter order:', ['sequence', 'keypoint_percentile', 'percentile', 'std_dev', 'cloudiness', 'gps_consistency']);
                
                // Get current filter parameters
                const params = new URLSearchParams({
                    enable_percentile: document.getElementById('enablePercentile').checked,
                    pair_percentile: document.getElementById('pairPercentile').value,
                    filter_order: JSON.stringify(window.currentFilterOrder || ['sequence', 'keypoint_percentile', 'percentile', 'std_dev', 'cloudiness', 'gps_consistency']),
                    enable_std_dev: document.getElementById('enableStdDev').checked,
                    std_dev_multiplier: document.getElementById('stdDevMultiplier').value,
                    enable_keypoint_percentile: document.getElementById('enableKeypointPercentile').checked,
                    keypoint_percentile: document.getElementById('keypointPercentile').value,
                    enable_sequence: document.getElementById('enableSequence').checked,
                    start_pair: document.getElementById('startPair').value,
                    end_pair: document.getElementById('endPair').value,
                    enable_cloudiness: document.getElementById('enableCloudiness').checked,
                    include_partly_cloudy: document.getElementById('includePartlyCloudy').checked,
                    include_mostly_cloudy: document.getElementById('includeMostlyCloudy').checked,
                    clear_brightness_min: document.getElementById('clearBrightnessMin').value,
                    clear_contrast_min: document.getElementById('clearContrastMin').value,
                    cloudy_brightness_max: document.getElementById('cloudyBrightnessMax').value,
                    cloudy_contrast_max: document.getElementById('cloudyContrastMax').value,
                    enable_gps_consistency: document.getElementById('enableGpsConsistency').checked,
                    gps_tolerance_constant: document.getElementById('gpsToleranceConstant').value,
                    speed_mode: document.getElementById('speedMode').value
                });
                
                console.log('Sending parameters:', params.toString());
                const response = await fetch(`/api/plot-data?${params}`);
                plotData = await response.json();
                
                console.log('Plot data loaded:', plotData);
                console.log('Histogram speeds count:', plotData.histogram?.speeds?.length);
                console.log('Pairs count:', plotData.pairs?.pairs?.length);
                
                if (plotData.error) {
                    showStatus(plotData.error, 'error');
                    return;
                }
                
                // Update GPS status if available
                if (plotData.gps_enabled !== undefined) {
                    updateGpsStatus(plotData.gps_enabled);
                }
                
                // Store individual keypoint data for filtering
                window.allKeypointsData = plotData.keypoints || [];
                
                // Update GPS consistency filter availability
                updateGpsConsistencyAvailability(plotData.gps_enabled || false);
                
                createPlots();
            } catch (error) {
                showStatus('Error loading plot data: ' + error.message, 'error');
            }
        }
        
        
        function updateSpeedometer() {
            console.log('updateSpeedometer called, plotData:', plotData);
            if (!plotData) {
                console.log('No plotData available');
                return;
            }
            
            const speedMode = document.getElementById('speedMode').value;
            console.log('Speed mode:', speedMode);
            console.log('plotData.speedometers:', plotData.speedometers);
            let speed, mode, count;
            
            if (plotData.speedometers) {
                switch(speedMode) {
                    case 'gps_only':
                        speed = plotData.speedometers.gps_only?.speed || 0;
                        mode = 'GPS-Only';
                        count = plotData.speedometers.gps_only?.count || 0;
                        break;
                    case 'constant_gsd':
                        speed = plotData.speedometers.constant_gsd?.speed || 0;
                        mode = 'Constant GSD (ESA)';
                        count = plotData.speedometers.constant_gsd?.count || 0;
                        break;
                    case 'dynamic_gsd':
                        speed = plotData.speedometers.dynamic_gsd?.speed || 0;
                        mode = 'Dynamic GSD (GPS+Keypoints)';
                        count = plotData.speedometers.dynamic_gsd?.count || 0;
                        break;
                    default:
                        speed = plotData.histogram?.mean || 0;
                        mode = 'Overall Average';
                        count = plotData.histogram?.speeds?.length || 0;
                }
            } else {
                speed = plotData.histogram?.mean || 0;
                mode = 'Overall Average';
                count = plotData.histogram?.speeds?.length || 0;
            }
            
            document.getElementById('currentSpeed').textContent = speed.toFixed(3);
            document.getElementById('currentMode').textContent = `${mode} (${count} points)`;
            console.log(`Speedometer updated: ${speed.toFixed(3)} km/s, ${mode} (${count} points)`);
        }
        
        function showEmptyPlots() {
            // Show empty state for all plots
            const emptyMessage = {
                x: [0.5],
                y: [0.5],
                mode: 'text',
                text: ['No data available<br>Try adjusting parameters'],
                textfont: {size: 16, color: '#666'},
                showlegend: false,
                hoverinfo: 'skip'
            };
            
            const emptyLayout = {
                xaxis: {visible: false},
                yaxis: {visible: false},
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                margin: {l: 0, r: 0, t: 0, b: 0}
            };
            
            // Clear plots with empty state
            Plotly.newPlot('histogram', [emptyMessage], emptyLayout);
            Plotly.newPlot('pairs', [emptyMessage], emptyLayout);
            
            // Reset speedometer
            const currentSpeedElement = document.getElementById('currentSpeed');
            const currentModeElement = document.getElementById('currentMode');
            if (currentSpeedElement) currentSpeedElement.textContent = '--';
            if (currentModeElement) currentModeElement.textContent = '--';
        }
        
        function createPlots() {
            console.log('createPlots called, plotData:', plotData);
            if (!plotData) {
                console.log('No plotData available in createPlots');
                return;
            }
            
            // Update speedometer
            updateSpeedometer();
            
            // Check if we have data
            const hasData = plotData.histogram.speeds.length > 0;
            
            if (!hasData) {
                // Show empty state for all plots
                showEmptyPlots();
                return;
            }
            
            // Histogram
            const speeds = plotData.histogram.speeds;
            const minSpeed = Math.min(...speeds);
            const maxSpeed = Math.max(...speeds);
            const speedRange = maxSpeed - minSpeed;
            const binSize = Math.max(0.1, speedRange / 30); // At least 0.1 km/s bins, max 30 bins
            
            Plotly.newPlot('histogram', [{
                x: speeds,
                type: 'histogram',
                xbins: {
                    start: minSpeed - binSize/2,
                    end: maxSpeed + binSize/2,
                    size: binSize
                },
                marker: {color: 'skyblue', line: {color: 'black', width: 1}},
                name: 'Speed Distribution'
            }], {
                title: 'Speed Distribution of All Keypoints',
                xaxis: {
                    title: 'Speed (km/s)',
                    range: [minSpeed - speedRange * 0.05, maxSpeed + speedRange * 0.05]
                },
                yaxis: {title: 'Number of Keypoints'},
                shapes: [
                    {type: 'line', x0: plotData.histogram.mean, x1: plotData.histogram.mean, y0: 0, y1: 1, yref: 'paper', line: {color: 'red', width: 2, dash: 'dash'}},
                    {type: 'line', x0: 7.66, x1: 7.66, y0: 0, y1: 1, yref: 'paper', line: {color: 'orange', width: 3}}
                ],
                annotations: [
                    {x: plotData.histogram.mean, y: 0.9, yref: 'paper', text: `Mean: ${plotData.histogram.mean.toFixed(3)} km/s`, showarrow: false, xanchor: 'left'},
                    {x: 7.66, y: 0.8, yref: 'paper', text: 'Target: 7.66 km/s', showarrow: false, xanchor: 'left'}
                ]
            });
            
            // Pairs plot (interactive) - Group by color to avoid legend repetition
            const colorGroups = {
                'green': {x: [], y: [], error_y: [], customdata: []},
                'orange': {x: [], y: [], error_y: [], customdata: []},
                'red': {x: [], y: [], error_y: [], customdata: []}
            };
            
            // Group data by color (only if we have pairs data)
            if (plotData.pairs.pairs && plotData.pairs.pairs.length > 0) {
                for (let i = 0; i < plotData.pairs.pairs.length; i++) {
                    const color = plotData.pairs.colors[i];
                    if (colorGroups[color]) {
                        colorGroups[color].x.push(plotData.pairs.pairs[i]);
                        colorGroups[color].y.push(plotData.pairs.means[i]);
                        colorGroups[color].error_y.push(plotData.pairs.stds[i]);
                        colorGroups[color].customdata.push(plotData.pairs.original_pairs[i]);
                    }
                }
            }
            
            // Create traces for each color group
            const pairTraces = [];
            const colorMap = {green: 'Clear', orange: 'Partly Cloudy', red: 'Mostly Cloudy'};
            
            Object.keys(colorGroups).forEach(color => {
                const group = colorGroups[color];
                if (group.x.length > 0) {
                    pairTraces.push({
                        x: group.x,
                        y: group.y,
                        error_y: {type: 'data', array: group.error_y},
                        mode: 'markers',
                        marker: {color: color, size: 10},
                        name: colorMap[color],
                        customdata: group.customdata,
                        hovertemplate: 'Pair: %{customdata}<br>Speed: %{y:.3f} km/s<br>Std: %{error_y.array[0]:.3f}<br>Click to view images<extra></extra>'
                    });
                }
            });
            
            Plotly.newPlot('pairs', pairTraces, {
                title: 'Average Speed per Image Pair',
                xaxis: {title: 'Image Pair Number'},
                yaxis: {title: 'Average Speed (km/s)'},
                shapes: [
                    {type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 7.66, y1: 7.66, line: {color: 'blue', width: 3}}
                ],
                annotations: [
                    {x: 0.5, y: 7.66, xref: 'paper', text: 'Target: 7.66 km/s', showarrow: false, yanchor: 'bottom'}
                ]
            });
            
            // Add click event to pairs plot
            document.getElementById('pairs').on('plotly_click', function(data) {
                console.log('Plot click event:', data);
                const pairNum = data.points[0].customdata;
                console.log('Clicked pair number:', pairNum);
                showPairImages(pairNum);
            });
            
            // Add legend click functionality for filtering
            document.getElementById('pairs').on('plotly_legendclick', function(data) {
                const legendItem = data.curveNumber;
                const traceName = data.fullData[legendItem].name;
                
                // Toggle visibility of the clicked legend item
                const update = {
                    visible: data.fullData[legendItem].visible === 'legendonly' ? true : 'legendonly'
                };
                
                Plotly.restyle('pairs', update, [legendItem]);
                
                // Recalculate and update other graphs based on visible data
                updateFilteredGraphs();
            });
        }
        
        async function showPairImages(pairNum) {
            console.log('showPairImages called with pairNum:', pairNum);
            try {
                const response = await fetch(`/api/pair/${pairNum}`);
                console.log('API response status:', response.status);
                const data = await response.json();
                console.log('API response data:', data);
                
                if (data.error) {
                    showStatus(data.error, 'error');
                    return;
                }
                
                document.getElementById('modalTitle').textContent = `Pair ${pairNum} Details`;
                
                const content = `
                    <div class="pair-info">
                        <h3>Pair Information</h3>
                        <div class="info-grid">
                            <div class="info-item">
                                <span class="info-label">Average Speed (Dynamic GSD):</span>
                                <span class="info-value">${data.avg_speed.toFixed(3)} km/s</span>
                            </div>
                            <div class="info-item">
                                <span class="info-label">Standard Deviation:</span>
                                <span class="info-value">${data.std_speed.toFixed(3)} km/s</span>
                            </div>
                            <div class="info-item">
                                <span class="info-label">Keypoints:</span>
                                <span class="info-value">${data.keypoint_count}</span>
                            </div>
                            <div class="info-item">
                                <span class="info-label">Time Difference:</span>
                                <span class="info-value">${data.time_diff} seconds</span>
                            </div>
                            <div class="info-item">
                                <span class="info-label">Image Quality:</span>
                                <span class="info-value">${data.cloudiness}</span>
                            </div>
                            <div class="info-item">
                                <span class="info-label">Brightness:</span>
                                <span class="info-value">${data.brightness.toFixed(1)}</span>
                            </div>
                            <div class="info-item">
                                <span class="info-label">Contrast:</span>
                                <span class="info-value">${data.contrast.toFixed(1)}</span>
                            </div>
                            <div class="info-item">
                                <span class="info-label">Accuracy:</span>
                                <span class="info-value">${data.accuracy.toFixed(1)}%</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="speed-methods">
                        <h3>Speed Calculation Methods</h3>
                        <div class="info-grid">
                            <div class="info-item">
                                <span class="info-label">GPS-Only Speed:</span>
                                <span class="info-value">${data.gps_only_avg ? data.gps_only_avg.toFixed(3) : 'N/A'} km/s (${data.gps_only_count} points)</span>
                            </div>
                            <div class="info-item">
                                <span class="info-label">Constant GSD Speed:</span>
                                <span class="info-value">${data.constant_gsd_avg ? data.constant_gsd_avg.toFixed(3) : 'N/A'} km/s (${data.constant_gsd_count} points)</span>
                            </div>
                            <div class="info-item">
                                <span class="info-label">Dynamic GSD Speed:</span>
                                <span class="info-value">${data.dynamic_gsd_avg ? data.dynamic_gsd_avg.toFixed(3) : 'N/A'} km/s (${data.dynamic_gsd_count} points)</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="gps-metadata">
                        <h3>GPS Metadata</h3>
                        <div class="info-grid">
                            <div class="info-item">
                                <span class="info-label">Image 1 GPS:</span>
                                <span class="info-value">${data.gps1 ? `${data.gps1[0].toFixed(6)}°N, ${data.gps1[1].toFixed(6)}°E` : 'Not available'}</span>
                            </div>
                            <div class="info-item">
                                <span class="info-label">Image 2 GPS:</span>
                                <span class="info-value">${data.gps2 ? `${data.gps2[0].toFixed(6)}°N, ${data.gps2[1].toFixed(6)}°E` : 'Not available'}</span>
                            </div>
                            <div class="info-item">
                                <span class="info-label">GPS Distance:</span>
                                <span class="info-value">${data.gps_distance_km ? data.gps_distance_km.toFixed(3) + ' km' : 'Not available'}</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="image-container">
                        <div class="image-box">
                            <h4>${data.image1}</h4>
                            <img src="data:image/jpeg;base64,${data.image1_data}" alt="Image 1">
                        </div>
                        <div class="image-box">
                            <h4>${data.image2}</h4>
                            <img src="data:image/jpeg;base64,${data.image2_data}" alt="Image 2">
                        </div>
                    </div>
                    
                    <div class="keypoint-matches">
                        <h3>Keypoint Matches (${data.displayed_matches} of ${data.total_matches} shown)</h3>
                        <p>Lines connect matching keypoints between the two images. This shows how features moved between the images.</p>
                        <img src="data:image/jpeg;base64,${data.matches_data}" alt="Keypoint Matches" style="max-width: 100%; height: auto; border: 1px solid #dee2e6; border-radius: 4px;">
                    </div>
                `;
                
                document.getElementById('modalContent').innerHTML = content;
                document.getElementById('imageModal').style.display = 'block';
                console.log('Modal displayed for pair:', pairNum);
                
            } catch (error) {
                showStatus('Error loading pair images: ' + error.message, 'error');
            }
        }
        
        function closeModal() {
            document.getElementById('imageModal').style.display = 'none';
        }
        
        function updateFilteredGraphs() {
            if (!plotData || !window.allKeypointsData) return;
            
            // Get the current state of the pairs plot
            const pairsDiv = document.getElementById('pairs');
            const pairsData = pairsDiv.data;
            
            // Collect all visible pair numbers
            const visiblePairs = new Set();
            
            // Get all pair numbers that are currently visible
            for (let i = 0; i < pairsData.length; i++) {
                const trace = pairsData[i];
                if (trace.visible !== 'legendonly' && trace.visible !== false) {
                    // This trace is visible, add its data
                    for (let j = 0; j < trace.x.length; j++) {
                        const pairNum = trace.x[j];
                        visiblePairs.add(pairNum);
                    }
                }
            }
            
            // Filter keypoints based on visible pairs
            const filteredKeypoints = window.allKeypointsData.filter(kp => visiblePairs.has(kp.pair_num));
            const filteredSpeeds = filteredKeypoints.map(kp => kp.speed);
            
            if (filteredSpeeds.length === 0) {
                showEmptyPlots();
                return;
            }
            
            // Update histogram with filtered data
            const meanSpeed = filteredSpeeds.reduce((a, b) => a + b, 0) / filteredSpeeds.length;
            const sortedSpeeds = [...filteredSpeeds].sort((a, b) => a - b);
            const medianSpeed = sortedSpeeds[Math.floor(sortedSpeeds.length / 2)];
            const stdSpeed = Math.sqrt(filteredSpeeds.reduce((sq, n) => sq + Math.pow(n - meanSpeed, 2), 0) / filteredSpeeds.length);
            
            Plotly.newPlot('histogram', [{
                x: filteredSpeeds,
                type: 'histogram',
                nbinsx: 50,
                marker: {color: 'skyblue', line: {color: 'black', width: 1}},
                name: 'Speed Distribution'
            }], {
                title: `Speed Distribution (${filteredSpeeds.length} keypoints)`,
                xaxis: {title: 'Speed (km/s)'},
                yaxis: {title: 'Number of Keypoints'},
                shapes: [
                    {type: 'line', x0: meanSpeed, x1: meanSpeed, y0: 0, y1: 1, yref: 'paper', line: {color: 'red', width: 2, dash: 'dash'}},
                    {type: 'line', x0: 7.66, x1: 7.66, y0: 0, y1: 1, yref: 'paper', line: {color: 'orange', width: 3}}
                ],
                annotations: [
                    {x: meanSpeed, y: 0.9, yref: 'paper', text: `Mean: ${meanSpeed.toFixed(3)} km/s`, showarrow: false, xanchor: 'left'},
                    {x: 7.66, y: 0.8, yref: 'paper', text: 'Target: 7.66 km/s', showarrow: false, xanchor: 'left'}
                ]
            });
            
            // Update speedometer - this will be handled by updateSpeedometer() function
            // which gets called after plot data is loaded
            
            // Show status
            showStatus(`Filtered to ${filteredSpeeds.length} keypoints from ${visiblePairs.size} pairs. Mean speed: ${meanSpeed.toFixed(3)} km/s`, 'info');
        }
        
        
        function showStatus(message, type) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = `status ${type}`;
            status.style.display = 'block';
            
            if (type === 'success' || type === 'error') {
                setTimeout(() => {
                    status.style.display = 'none';
                }, 5000);
            }
        }
        
        function updateGpsStatus(gpsEnabled) {
            const gpsStatus = document.getElementById('gpsStatus');
            const gpsText = document.getElementById('gpsText');
            
            if (gpsEnabled) {
                gpsText.textContent = 'GPS Enhanced Mode - Dynamic GSD Calculation';
                gpsStatus.style.display = 'flex';
            } else {
                gpsStatus.style.display = 'none';
            }
        }
        
        function showProgress(show = true) {
            const progressContainer = document.getElementById('progressContainer');
            progressContainer.style.display = show ? 'block' : 'none';
        }
        
        function updateProgress(percent, details = '') {
            const progressBar = document.getElementById('progressBar');
            const progressText = document.getElementById('progressText');
            const progressDetails = document.getElementById('progressDetails');
            
            progressBar.style.width = percent + '%';
            progressText.textContent = Math.round(percent) + '%';
            progressDetails.textContent = details;
        }
        
        async function clearExistingData() {
            try {
                // Clear backend data
                await fetch('/api/clear', { method: 'POST' });
                
                // Clear frontend data
                plotData = null;
                
                // Reset speedometer
                const currentSpeedElement = document.getElementById('currentSpeed');
                const currentModeElement = document.getElementById('currentMode');
                if (currentSpeedElement) currentSpeedElement.textContent = '--';
                if (currentModeElement) currentModeElement.textContent = '--';
                
                // Clear plots
                const plotIds = ['histogram', 'pairs'];
                plotIds.forEach(id => {
                    const element = document.getElementById(id);
                    if (element) {
                        element.innerHTML = '';
                    }
                });
                
                // Hide progress bar
                showProgress(false);
                
                showStatus('Previous data cleared', 'info');
                
            } catch (error) {
                console.log('Error clearing data:', error);
                // Continue anyway - this is not critical
            }
        }
        
        // Close modal when clicking outside
        window.onclick = function(event) {
            const modal = document.getElementById('imageModal');
            if (event.target === modal) {
                closeModal();
            }
        }
    </script>
</body>
</html>'''
    
    with open('templates/dashboard.html', 'w') as f:
        f.write(html_template)
    
    def convert_v1_to_v2_cache(cache_data):
        """Convert v1 cache format (keypoints) to v2 format (matches)"""
        try:
            raw_keypoints = cache_data.get('raw_keypoints', [])
            pair_characteristics = cache_data.get('pair_characteristics', {})
            
            if not raw_keypoints:
                print("❌ No keypoints found in v1 cache")
                return None
            
            print(f"🔄 Converting {len(raw_keypoints)} keypoints to matches...")
            
            # Convert keypoints to matches format
            matches = []
            for i, kp in enumerate(raw_keypoints):
                # Create a match object in v2 format
                match = {
                    'speed': kp.get('speed', 0),
                    'pixel_distance': kp.get('pixel_distance', 0),
                    'match_distance': kp.get('match_distance', 0),
                    'pt1': kp.get('pt1', (0, 0)),
                    'pt2': kp.get('pt2', (0, 0)),
                    'pair_index': kp.get('pair_num', 0),
                    'image1_name': kp.get('image1', ''),
                    'image2_name': kp.get('image2', ''),
                    'image1_path': kp.get('image1_path', ''),
                    'image2_path': kp.get('image2_path', ''),
                    'algorithm': 'ORB',  # Default algorithm
                    'use_flann': False,
                    'use_ransac_homography': False,
                    'ransac_threshold': 5.0,
                    'ransac_min_matches': 10,
                    'time_difference': kp.get('time_diff', 0),
                    'image1_properties': {},
                    'image2_properties': {},
                    'cloudiness': 'unknown',
                    'ml_classification1': None,
                    'ml_confidence1': None,
                    'ml_classification2': None,
                    'ml_confidence2': None,
                    'ml_classification': None,
                    'ml_confidence': None
                }
                matches.append(match)
            
            print(f"✅ Successfully converted {len(matches)} keypoints to matches")
            return matches
            
        except Exception as e:
            print(f"❌ Error converting v1 cache: {e}")
            return None

    def load_most_recent_cache():
        """Automatically load the most recent cache file on startup"""
        global processed_matches, cache_cleared_by_user
        
        try:
            # Don't auto-load if user has cleared the cache
            if cache_cleared_by_user:
                print("🚫 Skipping auto-load: cache was cleared by user")
                return False
                
            # Look for cache files in the cache directory first
            cache_dir = 'cache'
            cache_files = []
            
            if os.path.exists(cache_dir):
                cache_files.extend([os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith('.pkl')])
            
            # Also look for cache files in photos directories
            for item in os.listdir('.'):
                if item.startswith('photos-') and os.path.isdir(item):
                    photos_dir = item
                    cache_file = os.path.join(photos_dir, f'.{photos_dir}_keypoints_cache.pkl')
                    if os.path.exists(cache_file):
                        cache_files.append(cache_file)
            
            if cache_files:
                # Sort by modification time to get the most recent
                cache_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                most_recent_cache = cache_files[0]
                
                print(f"🔄 Auto-loading most recent cache: {os.path.basename(most_recent_cache)}")
                with open(most_recent_cache, 'rb') as f:
                    cache_data = pickle.load(f)
                
                # Handle different cache formats
                if isinstance(cache_data, list):
                    # New v2 format - direct list of matches
                    processed_matches = cache_data
                    print(f"✅ Loaded {len(processed_matches)} matches from v2 cache")
                elif isinstance(cache_data, dict) and 'raw_keypoints' in cache_data:
                    # Old v1 format - convert keypoints to matches
                    print(f"📊 Found v1 cache with {len(cache_data['raw_keypoints'])} keypoints")
                    print("🔄 Converting v1 cache to v2 format...")
                    processed_matches = convert_v1_to_v2_cache(cache_data)
                    if processed_matches:
                        print(f"✅ Converted and loaded {len(processed_matches)} matches from v1 cache")
                    else:
                        print("❌ Failed to convert v1 cache")
                        return False
                else:
                    print(f"⚠️ Unknown cache format: {type(cache_data)}")
                    return False
                
                return True
        except Exception as e:
            print(f"⚠️ Could not auto-load cache: {e}")
        
        return False
    
    print("🚀 Starting ISS Speed Analysis Dashboard...")
    print("🔧 Initializing Flask application...")
    
    # Get port from environment variable (Railway) or use default
    port = int(os.environ.get('PORT', 5003))
    debug_mode = os.environ.get('FLASK_ENV', 'development') == 'development'
    
    print(f"📊 Port: {port}")
    print(f"🔧 Debug mode: {debug_mode}")
    print(f"📊 Open your browser and go to: http://localhost:{port}")
    print("🔄 The dashboard will automatically detect photos-* folders in the project directory")
    print("✅ Flask app ready to start...")
    
    # Load cache in background thread to avoid blocking startup
    def load_cache_background():
        try:
            print("🔄 Loading cache in background...")
            load_most_recent_cache()
            print("✅ Cache loading completed")
        except Exception as e:
            print(f"⚠️ Background cache loading failed: {e}")
    
    # Start cache loading in background
    cache_thread = threading.Thread(target=load_cache_background, daemon=True)
    cache_thread.start()
    
    # Use environment port for Railway deployment
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
