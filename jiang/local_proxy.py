"""Enforce local model and generation settings on Claude's Anthropic requests."""
import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests


def constrain(payload, model, options, think):
    data = dict(payload)
    data["model"] = model
    data["temperature"] = options["temperature"]
    data["max_tokens"] = options["num_predict"]
    data["thinking"] = {"type": "enabled", "budget_tokens": 2048} if think else {"type": "disabled"}
    data.pop("output_config", None)  # A client effort preset must not override explicit thinking.
    return data


@contextmanager
def local_gateway(agent, log):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            route = self.path.split("?", 1)[0]
            if route not in ("/v1/messages", "/v1/messages/count_tokens"):
                self.send_error(404)
                return
            try:
                data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                if route == "/v1/messages":
                    data = constrain(data, agent.served_model, agent.options, agent.think)
                else:
                    data["model"] = agent.served_model
                with log.open("a") as trace:
                    trace.write(json.dumps({"time": time.time(), "route": route,
                        "model": data["model"], "thinking": data.get("thinking"),
                        "temperature": data.get("temperature"), "max_tokens": data.get("max_tokens")}) + "\n")
                with requests.Session() as session:
                    session.trust_env = False
                    with session.post(agent.endpoint + route, json=data, stream=True,
                        timeout=(10, 600), allow_redirects=False) as response:
                        if 300 <= response.status_code < 400:
                            raise ValueError("Local inference redirect rejected")
                        self.send_response(response.status_code)
                        self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
                        self.send_header("Connection", "close")
                        self.end_headers()
                        for chunk in response.iter_content(chunk_size=1024):
                            self.wfile.write(chunk)
                            self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except (ValueError, requests.RequestException) as exc:
                self.send_error(502, str(exc))

    # The parent agent validates its upstream endpoint as loopback-only.
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
