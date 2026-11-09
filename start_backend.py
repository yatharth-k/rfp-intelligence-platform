#!/usr/bin/env python3
"""
Backend startup script with port-conflict handling.
Checks port 8000, identifies existing processes, and manages Uvicorn instances.
"""

import sys
import os
import socket
import subprocess
import signal
import time
import psutil
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
API_SERVER_FILE = PROJECT_DIR / "api_server.py"
DEFAULT_PORT = 8000
ALTERNATE_PORTS = [8001, 8002, 8003, 8080, 8081]

def is_port_in_use(port: int) -> bool:
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def get_process_on_port(port: int) -> dict | None:
    """Get process info for the process listening on a port."""
    for conn in psutil.net_connections(kind='inet'):
        if conn.laddr.port == port and conn.status == 'LISTEN':
            try:
                proc = psutil.Process(conn.pid)
                return {
                    'pid': conn.pid,
                    'name': proc.name(),
                    'cmdline': ' '.join(proc.cmdline()),
                    'cwd': proc.cwd() if hasattr(proc, 'cwd') else None,
                }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    return None

def is_our_backend(proc_info: dict) -> bool:
    """Check if the process is our API server instance."""
    cmdline = proc_info.get('cmdline', '').lower()
    cwd = proc_info.get('cwd', '').lower()
    return (
        'api_server.py' in cmdline or 
        'uvicorn' in cmdline or
        'fastapi' in cmdline or
        PROJECT_DIR.name.lower() in cwd
    )

def kill_process(pid: int, timeout: float = 5.0) -> bool:
    """Gracefully kill a process."""
    try:
        proc = psutil.Process(pid)
        proc.terminate()
        proc.wait(timeout=timeout)
        return True
    except (psutil.NoSuchProcess, psutil.TimeoutExpired):
        try:
            proc = psutil.Process(pid)
            proc.kill()
            proc.wait(timeout=2.0)
            return True
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            return False
    except psutil.AccessDenied:
        return False

def find_available_port(start: int = DEFAULT_PORT) -> int:
    """Find an available port starting from start."""
    for port in [start] + ALTERNATE_PORTS:
        if not is_port_in_use(port):
            return port
    raise RuntimeError(f"No available ports in range {start}-{max(ALTERNATE_PORTS)}")

def start_uvicorn(port: int, host: str = "0.0.0.0") -> subprocess.Popen:
    """Start Uvicorn server."""
    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_DIR)
    cmd = [
        sys.executable, "-m", "uvicorn",
        "api_server:app",
        "--host", host,
        "--port", str(port),
        "--reload",
        "--log-level", "info",
    ]
    return subprocess.Popen(cmd, cwd=PROJECT_DIR, env=env)

def main():
    """Main entry point with port-conflict handling."""
    print(f"[Backend] Checking port {DEFAULT_PORT}...")
    
    if not is_port_in_use(DEFAULT_PORT):
        print(f"[Backend] Port {DEFAULT_PORT} is free. Starting server...")
        port = DEFAULT_PORT
    else:
        proc_info = get_process_on_port(DEFAULT_PORT)
        if proc_info:
            print(f"[Backend] Port {DEFAULT_PORT} occupied by PID {proc_info['pid']}: {proc_info['cmdline'][:100]}")
            
            if is_our_backend(proc_info):
                print(f"[Backend] Detected our existing backend instance (PID {proc_info['pid']}). Reusing it.")
                # Test if it's responsive
                try:
                    import requests
                    resp = requests.get(f"http://127.0.0.1:{DEFAULT_PORT}/health", timeout=2)
                    if resp.status_code == 200:
                        print(f"[Backend] Existing instance is healthy. Using port {DEFAULT_PORT}.")
                        return 0  # Success - server already running
                    else:
                        print(f"[Backend] Instance unhealthy (status {resp.status_code}). Restarting...")
                except Exception as e:
                    print(f"[Backend] Instance not responding ({e}). Restarting...")
                
                # Kill and restart
                print(f"[Backend] Stopping existing instance (PID {proc_info['pid']})...")
                if kill_process(proc_info['pid']):
                    print(f"[Backend] Stopped. Waiting for port release...")
                    time.sleep(1)
                else:
                    print(f"[Backend] Failed to stop. Finding alternate port...")
            else:
                print(f"[Backend] Port occupied by another application. Finding alternate port...")
        else:
            print(f"[Backend] Port {DEFAULT_PORT} in use but process not identifiable. Finding alternate port...")
        
        # Find alternate port
        port = find_available_port(DEFAULT_PORT + 1)
        print(f"[Backend] Using alternate port {port}")
    
    # Start server
    print(f"[Backend] Starting Uvicorn on port {port}...")
    proc = start_uvicorn(port)
    
    # Wait for server to be ready
    for _ in range(30):
        time.sleep(0.5)
        if is_port_in_use(port):
            try:
                import requests
                resp = requests.get(f"http://127.0.0.1:{port}/health", timeout=1)
                if resp.status_code == 200:
                    print(f"[Backend] Server ready at http://127.0.0.1:{port}")
                    # Write port to file for other scripts
                    (PROJECT_DIR / ".backend_port").write_text(str(port))
                    break
            except Exception:
                pass
    else:
        print(f"[Backend] ERROR: Server failed to start on port {port}")
        proc.terminate()
        return 1
    
    # Keep running
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n[Backend] Shutting down...")
        proc.terminate()
        proc.wait(timeout=5)
    return 0

if __name__ == "__main__":
    sys.exit(main())
