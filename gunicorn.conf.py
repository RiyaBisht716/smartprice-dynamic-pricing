import multiprocessing
import os

# Render sets PORT environment variable automatically
port = os.getenv("PORT", "10000")
bind = f"0.0.0.0:{port}"

# Optimize workers for Render's hardware (e.g. 512MB RAM on free tier)
# 2 workers are generally safe for lightweight web services, 
# or use 1 worker with threads to keep memory usage minimal.
workers = 1
threads = 4

# Preload app to save memory and speed up worker boot time
preload_app = True

# Timeout to prevent hanging workers
timeout = 120

# Access/Error logs to stdout/stderr for Render
accesslog = '-'
errorlog = '-'
