import http.server
import socketserver
import os
import sys

PORT = 8081
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

if __name__ == "__main__":
    os.chdir(DIRECTORY)
    
    # Check for Colab
    try:
        from google.colab import output
        colab_mode = True
    except ImportError:
        colab_mode = False

    print(f"Starting server on port {PORT}...")
    print(f"Serving directory: {DIRECTORY}")
    
    if colab_mode:
        print("Google Colab detected. Exposing secure tunnel link:")
        output.serve_kernel_port(PORT)
        
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
            sys.exit(0)
