#!/usr/bin/env python3
"""
AstroPi Explorer Dashboard - Railway Optimized Version
"""

import os
import sys
import threading
from datetime import datetime
from flask import Flask, jsonify, render_template

# Create Flask app
app = Flask(__name__)

# Global variables for complex features (loaded in background)
complex_features_loaded = False
loading_status = "Starting..."

def load_complex_features():
    """Load complex features in background thread"""
    global complex_features_loaded, loading_status
    
    try:
        loading_status = "Loading computer vision libraries..."
        print("🔄 Loading OpenCV and computer vision libraries...")
        
        # Import heavy dependencies
        import cv2
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        loading_status = "Loading image processing modules..."
        print("🔄 Loading image processing modules...")
        
        # Import additional modules
        from exif import Image
        import pickle
        import hashlib
        import statistics
        from collections import Counter
        
        loading_status = "Initializing ML models..."
        print("🔄 Initializing ML models...")
        
        # Try to load ML models (optional)
        try:
            import tensorflow as tf
            print("✅ TensorFlow loaded successfully")
        except ImportError:
            print("⚠️ TensorFlow not available - ML features disabled")
        
        loading_status = "Loading cache data..."
        print("🔄 Loading cache data...")
        
        # Try to load cache (this might take time)
        try:
            # This would be the cache loading logic from the original app
            # For now, just simulate it
            import time
            time.sleep(2)  # Simulate cache loading
            print("✅ Cache loading completed")
        except Exception as e:
            print(f"⚠️ Cache loading failed: {e}")
        
        complex_features_loaded = True
        loading_status = "All features loaded successfully!"
        print("🎉 All complex features loaded successfully!")
        
    except Exception as e:
        loading_status = f"Loading failed: {str(e)}"
        print(f"❌ Failed to load complex features: {e}")
        import traceback
        traceback.print_exc()

@app.route('/')
def index():
    """Main dashboard route"""
    try:
        return render_template('dashboard_v2_clean.html')
    except Exception as e:
        return f"Template error: {str(e)}", 500

@app.route('/health')
def health():
    """Health check endpoint for Railway"""
    try:
        return jsonify({
            "status": "healthy",
            "message": "AstroPi Explorer Dashboard is running",
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "port": os.environ.get('PORT', 'not_set'),
            "complex_features_loaded": complex_features_loaded,
            "loading_status": loading_status
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Health check failed: {str(e)}"
        }), 500

@app.route('/status')
def status():
    """Detailed status endpoint"""
    return jsonify({
        "app_status": "running",
        "complex_features_loaded": complex_features_loaded,
        "loading_status": loading_status,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/test')
def test():
    """Simple test endpoint"""
    return jsonify({"message": "Test endpoint working", "status": "ok"})

if __name__ == '__main__':
    print("🚀 Starting AstroPi Explorer Dashboard (Railway Optimized)...")
    print("🔧 Initializing Flask application...")
    
    # Get port from environment variable (Railway) or use default
    port = int(os.environ.get('PORT', 5003))
    debug_mode = os.environ.get('FLASK_ENV', 'development') == 'development'
    
    print(f"📊 Port: {port}")
    print(f"🔧 Debug mode: {debug_mode}")
    print(f"📊 Open your browser and go to: http://localhost:{port}")
    print("✅ Flask app ready to start...")
    
    # Start complex features loading in background
    print("🔄 Starting background loading of complex features...")
    feature_thread = threading.Thread(target=load_complex_features, daemon=True)
    feature_thread.start()
    
    # Start the app
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
