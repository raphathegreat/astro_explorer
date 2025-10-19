#!/usr/bin/env python3
"""
Minimal Flask app for Railway deployment testing
"""

import os
import sys
from datetime import datetime
from flask import Flask, jsonify, render_template

# Create Flask app
app = Flask(__name__)

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
            "port": os.environ.get('PORT', 'not_set')
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Health check failed: {str(e)}"
        }), 500

@app.route('/test')
def test():
    """Simple test endpoint"""
    return jsonify({"message": "Test endpoint working", "status": "ok"})

if __name__ == '__main__':
    print("🚀 Starting minimal AstroPi Explorer Dashboard...")
    print("🔧 Initializing Flask application...")
    
    # Get port from environment variable (Railway) or use default
    port = int(os.environ.get('PORT', 5003))
    debug_mode = os.environ.get('FLASK_ENV', 'development') == 'development'
    
    print(f"📊 Port: {port}")
    print(f"🔧 Debug mode: {debug_mode}")
    print(f"📊 Open your browser and go to: http://localhost:{port}")
    print("✅ Flask app ready to start...")
    
    # Start the app
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
