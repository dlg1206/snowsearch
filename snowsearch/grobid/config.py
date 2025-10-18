"""
File: config.py
Description: Defaults for Grobid

@author Derek Garcia
"""

# grobid server details
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8070
# download details
MAX_CONCURRENT_DOWNLOADS = 10  # todo add config option
MAX_PDF_COUNT = 100  # max pdfs allowed to be downloaded at a time
PDF_MAGIC = b'%PDF-'  # header bytes of pdf
MAX_RETRIES = 3
KILOBYTE = 1024

DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:116.0) Gecko/20100101 Firefox/116.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "TE": "Trailers"
}
