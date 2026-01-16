"""
File: config.py
Description: Defaults for Grobid

@author Derek Garcia
"""
import os

# grobid server details
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8070

MAX_GROBID_REQUESTS = os.cpu_count()
MAX_CONCURRENT_DOWNLOADS = 10
