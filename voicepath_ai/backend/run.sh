#!/bin/bash
find . -name "*.pyc" -delete 2>/dev/null
find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null
exec python3 -B app.py
