workers = 1
timeout = 120
worker_class = 'sync'  # Changed from 'gevent' to 'sync'
worker_connections = 1000
preload_app = True  # This helps with memory usage during startup