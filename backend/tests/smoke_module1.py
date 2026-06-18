"""Quick smoke test for Module 1 — run with the server already started."""

import json
import sys
import urllib.error
import urllib.request

# UTF-8 safety on Windows
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

BASE = "http://127.0.0.1:8000"


def test(endpoint: str, method: str = "GET") -> None:
    url = f"{BASE}{endpoint}"
    req = urllib.request.Request(url, method=method)
    try:
        r = urllib.request.urlopen(req, timeout=5)
        body = json.loads(r.read())
        rid = r.headers.get("X-Request-ID", "?")
        print(f"  [PASS] [{r.status}] {endpoint}")
        print(f"     X-Request-ID: {rid}")
        print(f"     success={body.get('success')}, code={body.get('code')}")
        if body.get("data"):
            if "status" in body["data"]:
                print(f"     overall={body['data']['status']}")
                for k, v in body["data"].get("components", {}).items():
                    print(f"       {k}: {v['status']}")
    except urllib.error.HTTPError as e:
        body = json.loads(e.read())
        print(f"  [PASS] [{e.code}] {endpoint}")
        print(f"     success={body.get('success')}, message={body.get('message')}")
    except Exception as e:
        print(f"  [FAIL] {endpoint}: {e}")


if __name__ == "__main__":
    print("Module 1 Smoke Tests\n")

    test("/api/v1/health")
    test("/openapi.json")
    test("/api/v1/nonexistent")  # 404
    test("/openapi.json", method="POST")  # 405

    print("\nDone.")
