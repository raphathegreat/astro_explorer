#!/usr/bin/env python3
"""
Ultra-simple Flask app for Railway testing
"""

import os
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return """
    <html>
    <head><title>AstroPi Explorer Dashboard</title></head>
    <body>
        <h1>🚀 AstroPi Explorer Dashboard</h1>
        <p>✅ App is running successfully!</p>
        <p>🎯 This is a test deployment to verify Railway is working.</p>
        <p>📊 <a href="/health">Health Check</a></p>
        <p>🔍 <a href="/status">Status</a></p>
    </body>
    </html>
    """

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "message": "AstroPi Explorer Dashboard is running",
        "port": os.environ.get('PORT', 'not_set')
    })

@app.route('/status')
def status():
    return jsonify({
        "app": "AstroPi Explorer Dashboard",
        "status": "running",
        "version": "test-1.0"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5003))
    print(f"🚀 Starting simple AstroPi app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
