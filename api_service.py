import atexit
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests


_PROJECT_DIR = Path(__file__).resolve().parent
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
_API_PROCESS = None
_API_LOCK = threading.Lock()


def _api_is_healthy(api_base):
    try:
        response = requests.get(f"{api_base}/health", timeout=1)
        response.raise_for_status()
        return response.json().get("status") == "healthy"
    except (requests.RequestException, ValueError, AttributeError):
        return False


def ensure_local_api(api_base, startup_timeout=45):
    """Start the local FastAPI service when the requested local endpoint is down."""
    global _API_PROCESS

    parsed_url = urlsplit(api_base)
    if (
        parsed_url.scheme != "http"
        or parsed_url.hostname not in _LOCAL_HOSTS
        or parsed_url.path not in ("", "/")
    ):
        return

    port = parsed_url.port or 80
    with _API_LOCK:
        if _api_is_healthy(api_base):
            return

        if _API_PROCESS is None or _API_PROCESS.poll() is not None:
            _API_PROCESS = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "main:app",
                    "--host",
                    parsed_url.hostname,
                    "--port",
                    str(port),
                ],
                cwd=str(_PROJECT_DIR),
            )

        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            if _api_is_healthy(api_base):
                return

            exit_code = _API_PROCESS.poll()
            if exit_code is not None:
                _API_PROCESS = None
                raise RuntimeError(
                    f"API server exited with code {exit_code}; "
                    "check the Streamlit terminal for startup errors."
                )
            time.sleep(0.25)

        _stop_api()
        raise RuntimeError(
            f"API server did not become healthy within {startup_timeout} seconds; "
            "check the Streamlit terminal for startup errors."
        )


def _stop_api():
    global _API_PROCESS

    process = _API_PROCESS
    _API_PROCESS = None
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


atexit.register(_stop_api)
