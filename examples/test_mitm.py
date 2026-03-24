"""Test the MITM proxy by making HTTPS requests through it.

Run with: snose run --mitm python3 examples/test_mitm.py
"""

import os
import ssl

import httpx

print(f"HTTPS_PROXY: {os.environ.get('HTTPS_PROXY', 'not set')}")
print(f"Run ID: {os.environ.get('STARNOSE_RUN_ID', 'not set')}")
print()

# Build SSL context that trusts our CA
ca_path = os.environ.get("NODE_EXTRA_CA_CERTS")
if ca_path:
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.load_verify_locations(ca_path)
    # Also load default system CAs for non-intercepted hosts
    ssl_ctx.load_default_certs()
    verify = ssl_ctx
    print(f"Using CA: {ca_path}")
else:
    verify = True
    print("No custom CA set, using defaults")

print()

# Test 1: Anthropic API interception
# The request will be intercepted by the MITM proxy, forwarded to the real
# api.anthropic.com, and the response recorded. Auth will fail (no real key)
# but starnose still records the request.
print("Test: Anthropic API call through MITM proxy...")
try:
    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": "test-key-not-real",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-3-haiku-20240307",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Hello, this is a test of starnose MITM!"}
            ],
        },
        verify=verify,
        timeout=15,
    )
    print(f"  Status: {resp.status_code}")
    print(f"  (401 expected — no real API key, but request was intercepted)")
except httpx.ConnectError as e:
    print(f"  ConnectError: {e}")
    print("  (If SSL error: run 'snose setup' to trust the CA)")
except Exception as e:
    print(f"  {type(e).__name__}: {e}")

print()
print("Done! Run 'snose inspect' to see recorded calls.")
