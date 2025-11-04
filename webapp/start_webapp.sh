#!/bin/bash
# Start the Trash Detection Web Application

cd "$(dirname "$0")"

echo "=========================================="
echo "Trash Detection Web Application"
echo "=========================================="
echo ""
echo "Starting Flask server..."
echo "Access the web interface at: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Activate venv and run
../venv/bin/python app.py
