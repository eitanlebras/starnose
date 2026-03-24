"""MITM proxy for intercepting HTTPS traffic from Claude Code and other agents.

Works by handling HTTP CONNECT tunnels. For api.anthropic.com, performs TLS
interception using a locally-generated CA cert. All other hosts are tunneled
transparently without interception.
"""

from __future__ import annotations

import json
import logging
import socket
import ssl
import threading
import time

import httpx

from starnose.certs import create_server_cert, CA_CERT_PATH
from starnose.db import Database

logger = logging.getLogger("starnose.mitm")

# Only MITM these hosts — everything else tunnels through untouched
INTERCEPT_HOSTS = {"api.anthropic.com"}


class MITMProxy:
    """Threading-based MITM proxy server."""

    def __init__(self, db: Database, run_id: str, port: int):
        self.db = db
        self.run_id = run_id
        self.port = port
        self._server_sock: socket.socket | None = None
        self._running = False

    def start(self) -> None:
        """Start the proxy (blocking). Run in a daemon thread."""
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind(("127.0.0.1", self.port))
        self._server_sock.listen(16)
        self._server_sock.settimeout(1.0)
        self._running = True

        while self._running:
            try:
                client_sock, _ = self._server_sock.accept()
                threading.Thread(
                    target=self._handle_client,
                    args=(client_sock,),
                    daemon=True,
                ).start()
            except socket.timeout:
                continue
            except OSError:
                break

    def stop(self) -> None:
        self._running = False
        if self._server_sock:
            try:
                self._server_sock.close()
            except OSError:
                pass

    # ── Connection dispatch ──────────────────────────────────────────────

    def _handle_client(self, client_sock: socket.socket) -> None:
        try:
            data = self._recv_until(client_sock, b"\r\n\r\n")
            if not data:
                client_sock.close()
                return

            first_line = data.split(b"\r\n")[0].decode("utf-8", errors="replace")

            if first_line.startswith("CONNECT "):
                target = first_line.split()[1]
                host = target.split(":")[0]
                port = int(target.split(":")[1]) if ":" in target else 443

                if host in INTERCEPT_HOSTS:
                    self._handle_mitm(client_sock, host, port)
                else:
                    self._handle_tunnel(client_sock, host, port)
            else:
                client_sock.close()
        except Exception as e:
            logger.debug("Client handler error: %s", e)
            try:
                client_sock.close()
            except OSError:
                pass

    # ── MITM interception ────────────────────────────────────────────────

    def _handle_mitm(self, client_sock: socket.socket, host: str, port: int) -> None:
        """Intercept TLS, record API calls, forward to real upstream."""
        # Acknowledge the CONNECT
        client_sock.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")

        # Generate a cert for this host signed by our CA
        cert_path, key_path = create_server_cert(host)

        # TLS handshake — we pose as the target host
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(str(cert_path), str(key_path))
        ctx.set_alpn_protocols(["http/1.1"])

        try:
            tls_sock = ctx.wrap_socket(client_sock, server_side=True)
        except ssl.SSLError as e:
            logger.debug("TLS handshake failed: %s", e)
            try:
                client_sock.close()
            except OSError:
                pass
            return

        # Handle one or more HTTP requests on this connection (keep-alive)
        try:
            while True:
                headers_raw, body = self._read_http_request(tls_sock)
                if not headers_raw:
                    break
                self._forward_and_record(tls_sock, host, headers_raw, body)
        except (ConnectionError, ssl.SSLError, OSError):
            pass
        finally:
            try:
                tls_sock.close()
            except OSError:
                pass

    def _forward_and_record(
        self,
        tls_sock: ssl.SSLSocket,
        host: str,
        headers_raw: bytes,
        body: bytes,
    ) -> None:
        """Parse the request, forward upstream, record, and relay response."""
        header_text = headers_raw.decode("utf-8", errors="replace")
        lines = header_text.split("\r\n")
        request_line = lines[0]
        parts = request_line.split()
        method = parts[0] if parts else "POST"
        path = parts[1] if len(parts) > 1 else "/"

        # Collect headers to forward
        fwd_headers: dict[str, str] = {}
        for line in lines[1:]:
            if ": " in line:
                key, value = line.split(": ", 1)
                low = key.lower()
                if low in ("host", "content-length", "connection", "proxy-connection"):
                    continue
                fwd_headers[key] = value

        # Parse body
        try:
            request_data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            request_data = {}

        is_streaming = request_data.get("stream", False)
        model = request_data.get("model", "unknown")
        messages = self._extract_messages(request_data, path)
        params = {
            k: v
            for k, v in request_data.items()
            if k not in ("messages", "model", "stream", "system")
        }

        url = f"https://{host}{path}"
        start_ms = _now_ms()

        try:
            if is_streaming:
                self._relay_streaming(tls_sock, url, fwd_headers, body, messages, model, params, start_ms)
            else:
                self._relay_non_streaming(tls_sock, url, fwd_headers, body, messages, model, params, start_ms)
        except Exception as e:
            logger.debug("Relay error: %s", e)
            self._send_error(tls_sock, 502, str(e))

    # ── Relay helpers ────────────────────────────────────────────────────

    def _relay_non_streaming(
        self, tls_sock, url, fwd_headers, body, messages, model, params, start_ms
    ) -> None:
        with httpx.Client(verify=True, timeout=120) as client:
            resp = client.post(url, headers=fwd_headers, content=body)

        latency_ms = _now_ms() - start_ms
        content = resp.content

        # Record BEFORE sending response — ensures recording completes
        # even if the client/subprocess exits immediately after receiving it
        try:
            response_data = json.loads(content)
        except (json.JSONDecodeError, UnicodeDecodeError):
            response_data = {}

        self._record(model, params, messages, response_data, latency_ms)

        # Now send response to the intercepted client
        resp_line = f"HTTP/1.1 {resp.status_code} OK\r\n"
        hdr_lines = self._filter_response_headers(resp.headers, len(content))
        tls_sock.sendall(resp_line.encode() + hdr_lines.encode() + b"\r\n" + content)

    def _relay_streaming(
        self, tls_sock, url, fwd_headers, body, messages, model, params, start_ms
    ) -> None:
        collected: list[bytes] = []

        with httpx.Client(verify=True, timeout=300) as client:
            with client.stream("POST", url, headers=fwd_headers, content=body) as resp:
                # Response line
                tls_sock.sendall(f"HTTP/1.1 {resp.status_code} OK\r\n".encode())

                # Forward headers — drop transfer-encoding so we stream raw
                for key, value in resp.headers.multi_items():
                    low = key.lower()
                    if low in ("transfer-encoding", "content-length"):
                        continue
                    tls_sock.sendall(f"{key}: {value}\r\n".encode())
                tls_sock.sendall(b"\r\n")

                # Stream body chunks
                for chunk in resp.iter_raw():
                    tls_sock.sendall(chunk)
                    collected.append(chunk)

        latency_ms = _now_ms() - start_ms
        full_body = b"".join(collected)
        response_data = self._parse_sse(full_body)
        self._record(model, params, messages, response_data, latency_ms)

    # ── Transparent tunnel (non-intercepted hosts) ───────────────────────

    def _handle_tunnel(self, client_sock: socket.socket, host: str, port: int) -> None:
        """Tunnel traffic without interception."""
        client_sock.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")

        try:
            upstream = socket.create_connection((host, port), timeout=10)
        except OSError:
            client_sock.close()
            return

        def forward(src: socket.socket, dst: socket.socket) -> None:
            try:
                while True:
                    data = src.recv(65536)
                    if not data:
                        break
                    dst.sendall(data)
            except (OSError, ConnectionError):
                pass
            try:
                dst.shutdown(socket.SHUT_WR)
            except OSError:
                pass

        t1 = threading.Thread(target=forward, args=(client_sock, upstream), daemon=True)
        t2 = threading.Thread(target=forward, args=(upstream, client_sock), daemon=True)
        t1.start()
        t2.start()
        t1.join(timeout=600)
        t2.join(timeout=600)

        for s in (upstream, client_sock):
            try:
                s.close()
            except OSError:
                pass

    # ── Protocol helpers ─────────────────────────────────────────────────

    @staticmethod
    def _recv_until(sock: socket.socket, delimiter: bytes, timeout: float = 30) -> bytes:
        """Receive data until delimiter is found."""
        sock.settimeout(timeout)
        data = b""
        try:
            while delimiter not in data:
                chunk = sock.recv(4096)
                if not chunk:
                    return data
                data += chunk
        except socket.timeout:
            pass
        return data

    @staticmethod
    def _read_http_request(sock: ssl.SSLSocket) -> tuple[bytes | None, bytes]:
        """Read a full HTTP request (headers + body) from a TLS socket."""
        sock.settimeout(60)
        data = b""
        try:
            while b"\r\n\r\n" not in data:
                chunk = sock.recv(4096)
                if not chunk:
                    return None, b""
                data += chunk
        except socket.timeout:
            return None, b""

        header_end = data.index(b"\r\n\r\n") + 4
        headers_raw = data[:header_end]
        body_so_far = data[header_end:]

        # Parse Content-Length
        content_length = 0
        for line in headers_raw.decode("utf-8", errors="replace").split("\r\n"):
            if line.lower().startswith("content-length:"):
                content_length = int(line.split(":", 1)[1].strip())

        # Read remaining body bytes
        body = body_so_far
        while len(body) < content_length:
            remaining = content_length - len(body)
            try:
                chunk = sock.recv(min(65536, remaining))
            except socket.timeout:
                break
            if not chunk:
                break
            body += chunk

        return headers_raw, body

    @staticmethod
    def _filter_response_headers(headers: httpx.Headers, content_length: int) -> str:
        """Build response header block, replacing content-length."""
        lines = []
        for key, value in headers.multi_items():
            low = key.lower()
            if low in ("transfer-encoding", "content-length", "content-encoding"):
                continue
            lines.append(f"{key}: {value}\r\n")
        lines.append(f"Content-Length: {content_length}\r\n")
        return "".join(lines)

    @staticmethod
    def _send_error(sock: ssl.SSLSocket, status: int, message: str) -> None:
        body = json.dumps({"error": {"message": message}}).encode()
        try:
            sock.sendall(
                f"HTTP/1.1 {status} Error\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\n"
                f"\r\n".encode()
                + body
            )
        except OSError:
            pass

    @staticmethod
    def _extract_messages(data: dict, path: str) -> list[dict]:
        """Extract normalized messages from Anthropic or OpenAI request bodies."""
        messages: list[dict] = []

        if "/messages" in path:
            # Anthropic format
            system = data.get("system", "")
            if system:
                if isinstance(system, list):
                    text = "\n".join(
                        b.get("text", "") for b in system if isinstance(b, dict)
                    )
                else:
                    text = str(system)
                messages.append({"role": "system", "content": text})

            for msg in data.get("messages", []):
                content = msg.get("content", "")
                if isinstance(content, list):
                    parts = []
                    for block in content:
                        if isinstance(block, dict):
                            parts.append(block.get("text", str(block)))
                        else:
                            parts.append(str(block))
                    content = "\n".join(parts)
                messages.append({"role": msg.get("role", "user"), "content": content})

        elif "/chat/completions" in path:
            messages = data.get("messages", [])

        return messages

    @staticmethod
    def _parse_sse(body: bytes) -> dict:
        """Parse a streamed SSE response into a normalized response dict."""
        text_parts: list[str] = []
        usage: dict = {}
        stop_reason: str | None = None

        for line in body.decode("utf-8", errors="replace").split("\n"):
            if not line.startswith("data: "):
                continue
            raw = line[6:].strip()
            if raw == "[DONE]":
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue

            etype = event.get("type", "")

            # Anthropic SSE
            if etype == "content_block_delta":
                text_parts.append(event.get("delta", {}).get("text", ""))
            elif etype == "message_delta":
                stop_reason = event.get("delta", {}).get("stop_reason")
                usage.update(event.get("usage", {}))
            elif etype == "message_start":
                usage.update(event.get("message", {}).get("usage", {}))

            # OpenAI SSE
            choices = event.get("choices", [])
            if choices:
                delta_text = choices[0].get("delta", {}).get("content", "")
                if delta_text:
                    text_parts.append(delta_text)
                fr = choices[0].get("finish_reason")
                if fr:
                    stop_reason = fr
            if event.get("usage"):
                usage.update(event["usage"])

        return {
            "content": [{"type": "text", "text": "".join(text_parts)}],
            "stop_reason": stop_reason,
            "usage": usage,
        }

    def _record(self, model, params, messages, response_data, latency_ms) -> None:
        try:
            self.db.add_call(self.run_id, model, params, messages, response_data, latency_ms)
        except Exception:
            pass  # Fail open


def _now_ms() -> int:
    return int(time.time() * 1000)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_mitm_server(db: Database, run_id: str, port: int) -> None:
    """Run MITM proxy (blocking). Intended to be run in a daemon thread."""
    proxy = MITMProxy(db, run_id, port)
    proxy.start()
