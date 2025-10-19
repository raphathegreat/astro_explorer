#!/usr/bin/env python3
"""
Ultra-simple Flask app for Railway testing
"""

import os
from flask import Flask, jsonify, Response, request

app = Flask(__name__)

# Force all routes to be handled by Flask
@app.before_request
def before_request():
    print(f"🔍 Request to: {request.path}")
    return None

@app.route('/')
def home():
    return jsonify({
        "message": "🚀 AstroPi Explorer Dashboard is running!",
        "status": "success",
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "test": "/test"
        },
        "note": "This is a test deployment to verify Railway is working."
    })

@app.route('/test')
def test():
    return jsonify({"message": "Test endpoint working", "status": "ok"})

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
        "version": "test-1.0",
        "port": os.environ.get('PORT', 'not_set')
    })

# Catch-all route to handle any unmatched paths
@app.route('/<path:path>')
def catch_all(path):
    return jsonify({
        "message": f"Route /{path} not found",
        "available_routes": ["/", "/health", "/status", "/test"],
        "note": "This request was handled by Flask"
    }), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5003))
    print(f"🚀 Starting simple AstroPi app on port {port}")
    print(f"🔧 Environment: {os.environ.get('FLASK_ENV', 'production')}")
    app.run(host='0.0.0.0', port=port, debug=False)
