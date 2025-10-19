#!/usr/bin/env python3
"""
Ultra-minimal test app for Railway
"""

import os
from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "AstroPi Dashboard is running!"

@app.route('/health')
def health():
    return "OK"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5003))
    print(f"Starting on port {port}")
    app.run(host='0.0.0.0', port=port)
