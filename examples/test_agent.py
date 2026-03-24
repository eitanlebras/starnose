"""Simple test agent that makes LLM calls through a built-in mock.

No API key needed. Run with:
  STARNOSE_OPENAI_UPSTREAM=http://127.0.0.1:19876 snose run python3 examples/test_agent.py
"""

import os
import json
import socket
import http.server
import threading
import time

import httpx


# ── Start mock LLM server on a fixed port ────────────────────────────────────

MOCK_PORT = 19876


class MockLLMHandler(http.server.BaseHTTPRequestHandler):
    call_count = 0

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        MockLLMHandler.call_count += 1
        messages = body.get("messages", [])
        user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

        response = {
            "id": f"chatcmpl-mock-{MockLLMHandler.call_count}",
            "object": "chat.completion",
            "model": body.get("model", "gpt-4o"),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Mock response #{MockLLMHandler.call_count}: {user_msg[:80]}",
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": sum(len(m.get("content", "")) // 4 for m in messages),
                "completion_tokens": 20,
            },
        }

        data = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


mock_server = http.server.HTTPServer(("127.0.0.1", MOCK_PORT), MockLLMHandler)
mock_thread = threading.Thread(target=mock_server.serve_forever, daemon=True)
mock_thread.start()
time.sleep(0.1)

# ── Make calls through the starnose proxy ────────────────────────────────────

base_url = os.environ.get("OPENAI_BASE_URL", f"http://127.0.0.1:{MOCK_PORT}/v1")
api_key = os.environ.get("OPENAI_API_KEY", "test-key")

print(f"proxy: {base_url}")
print(f"run:   {os.environ.get('STARNOSE_RUN_ID', 'none')}")
print()

scenarios = [
    {
        "messages": [
            {"role": "system", "content": "You are a research assistant. Use tools to find information. Always cite your sources. Be thorough."},
            {"role": "user", "content": "What are the latest developments in quantum computing?"},
        ],
    },
    {
        "messages": [
            {"role": "system", "content": "You are a research assistant. Use tools to find information. Always cite your sources. Be thorough."},
            {"role": "user", "content": "What are the latest developments in quantum computing?"},
            {"role": "assistant", "content": "I found several key developments in quantum computing..."},
            {"role": "user", "content": "Tell me more about error correction breakthroughs."},
            {"role": "tool", "content": "Search results for 'quantum error correction 2024':\n" + "Result entry with details about quantum research. " * 50},
        ],
    },
    {
        "messages": [
            {"role": "system", "content": "You are a research assistant. Use tools to find information. Always cite your sources. Be thorough."},
            {"role": "user", "content": "What are the latest developments in quantum computing?"},
            {"role": "assistant", "content": "I found several key developments in quantum computing..."},
            {"role": "user", "content": "Tell me more about error correction breakthroughs."},
            {"role": "tool", "content": "Search results for 'quantum error correction 2024':\n" + "Result entry with details about quantum research. " * 50},
            {"role": "assistant", "content": "Based on my research, here are the key error correction breakthroughs..."},
            {"role": "user", "content": "Now write a comprehensive report. Include all raw data."},
            {"role": "tool", "content": "Full database export:\n" + "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 400},
        ],
    },
]

print(f"Making {len(scenarios)} LLM calls...\n")

for i, scenario in enumerate(scenarios, 1):
    resp = httpx.post(
        f"{base_url}/chat/completions",
        json={"model": "gpt-4o", "messages": scenario["messages"], "temperature": 0.7},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10,
    )

    data = resp.json()
    reply = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    print(f"  Call {i}: {usage.get('prompt_tokens', 0):>5} in / {usage.get('completion_tokens', 0):>3} out")
    print(f"          → {reply[:70]}")
    print()

mock_server.shutdown()
print("Done! Now try:")
print("  snose inspect")
print("  snose stats")
print("  snose optimize")
