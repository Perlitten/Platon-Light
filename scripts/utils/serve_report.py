#!/usr/bin/env python
"""
Simple HTTP server to serve a specific HTML file.
"""

import http.server
import socketserver
import os
from pathlib import Path

# Configuration
PORT = 8080
REPORT_DIR = Path(__file__).parent / "results"
REPORT_FILES = list(REPORT_DIR.glob("*_report_*.html"))

if not REPORT_FILES:
    print("No report files found!")
    exit(1)

# Sort by modification time (most recent first)
REPORT_FILES.sort(key=lambda x: x.stat().st_mtime, reverse=True)
REPORT_FILE = REPORT_FILES[0]

print(f"Serving report: {REPORT_FILE}")

# Change to the directory containing the report
os.chdir(REPORT_FILE.parent)


class MyHttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Redirect all requests to the report file
        self.path = REPORT_FILE.name
        return http.server.SimpleHTTPRequestHandler.do_GET(self)


# Create an object of the above class
handler_object = MyHttpRequestHandler

# Create a server
my_server = socketserver.TCPServer(("", PORT), handler_object)

# Start the server
print(f"Server started at http://localhost:{PORT}")
print("Press Ctrl+C to stop the server")
my_server.serve_forever()
