# SPDX-License-Identifier: Apache-2.0
"""Test live HTTP chat completion against a specified target server (default: localhost:8080).

Usage:
    python tests/test_live_inference.py --server http://192.168.7.152:8080
    python tests/test_live_inference.py --host 192.168.7.152:8080
    HYDRAGEN_SERVER_URL=http://192.168.7.152:8080 python -m unittest tests/test_live_inference.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import unittest
import urllib.request


def resolve_target_url() -> str:
    """Resolve target inference server URL from CLI arguments, environment, or default to localhost:8080."""
    url = None

    # 1. Parse and extract CLI argument if present (--server, --host, --url)
    args_to_remove = []
    for i, arg in enumerate(sys.argv):
        if arg in ("--server", "--host", "--url") and i + 1 < len(sys.argv):
            url = sys.argv[i + 1]
            args_to_remove.extend([i, i + 1])
            break
        elif arg.startswith(("--server=", "--host=", "--url=")):
            url = arg.split("=", 1)[1]
            args_to_remove.append(i)
            break

    # Strip custom flags so unittest.main() doesn't fail on unknown arguments
    for idx in sorted(args_to_remove, reverse=True):
        sys.argv.pop(idx)

    # 2. Check environment variable override
    if not url:
        url = os.environ.get("HYDRAGEN_SERVER_URL") or os.environ.get("HYDRAGEN_HOST")

    # 3. Default fallback to localhost:8080
    if not url:
        url = "http://127.0.0.1:8080"

    if not url.startswith("http://") and not url.startswith("https://"):
        url = f"http://{url}"

    return url.rstrip("/")


TARGET_SERVER_URL = resolve_target_url()


class TestLiveInference(unittest.TestCase):
    def test_live_chat_completion(self):
        target = TARGET_SERVER_URL
        t0 = time.time()

        # Probe server health or props to discover active model alias
        model_name = "qwen3.8-flash-next-iq3_xxs"
        try:
            with urllib.request.urlopen(f"{target}/health", timeout=3) as resp:
                health = json.loads(resp.read().decode("utf-8"))
                model_name = health.get("model", model_name)
        except Exception:
            try:
                with urllib.request.urlopen(f"{target}/props", timeout=3) as resp:
                    props = json.loads(resp.read().decode("utf-8"))
                    model_name = props.get("model_alias", model_name)
            except Exception:
                pass

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "What is the capital of France? Reply in 5 words or less."}],
            "max_tokens": 50,
        }
        req = urllib.request.Request(
            f"{target}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                res = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            self.skipTest(f"Live server not responding at {target}: {e}")

        dt = time.time() - t0
        content = res["choices"][0]["message"].get("content", "")
        reasoning = res["choices"][0]["message"].get("reasoning_content", "")
        tokens = res.get("usage", {}).get("completion_tokens", 0)
        print(f"\n[LiveInference] Server: {target} | Model: {model_name}")
        if content:
            print(f"Content: {content.strip()}")
        if reasoning:
            print(f"Reasoning: {reasoning.strip()[:100]}...")
        print(f"Tokens: {tokens}, Elapsed: {round(dt, 2)}s, Speed: {round(tokens / max(0.01, dt), 1)} tok/s")
        self.assertTrue(len(content) > 0 or len(reasoning) > 0)


if __name__ == "__main__":
    unittest.main()
