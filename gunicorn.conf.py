workers = 1
timeout = 300  # Increased timeout for frame processing
worker_class = 'sync'
max_requests = 100  # Helps prevent memory leaks
max_requests_jitter = 20
preload_app = True
worker_connections = 1000
preload_app = True  # This helps with memory usage during startup