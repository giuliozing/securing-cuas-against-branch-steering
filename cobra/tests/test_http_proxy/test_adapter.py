"""Tests for `cobra.http_proxy.adapter` (no mitmproxy needed: the flow is
duck-typed by design).

Run from `cobra/`:
    python3 -m pytest tests/test_http_proxy/test_adapter.py
"""

import json
import unittest

from cobra.http_proxy.adapter import build_request_view, parse_body, request_view_from_flow


class FakeHeaders(dict):
    def get(self, key, default=None):
        return super().get(key.lower(), default)


class FakeRequest:
    def __init__(self, method, url, content_type="", raw_content=None):
        self.method = method
        self.pretty_url = url
        self.headers = FakeHeaders({"content-type": content_type})
        self.raw_content = raw_content


class FakeFlow:
    def __init__(self, request):
        self.request = request


class TestParseBody(unittest.TestCase):
    def test_empty_body(self):
        self.assertEqual(parse_body("application/json", None), {})
        self.assertEqual(parse_body("application/json", b""), {})

    def test_json_object(self):
        body = parse_body("application/json", b'{"amount": 42.99, "currency": "GBP"}')
        self.assertEqual(body, {"amount": 42.99, "currency": "GBP"})
        # JSON types survive parsing untouched (strictness depends on it).
        self.assertIsInstance(body["amount"], float)

    def test_json_non_object(self):
        self.assertEqual(parse_body("application/json", b"[1, 2]"), {"_json": [1, 2]})

    def test_json_invalid(self):
        self.assertEqual(parse_body("application/json", b"{oops"), {"_raw": "{oops"})

    def test_form_urlencoded(self):
        body = parse_body("application/x-www-form-urlencoded", b"amount=500&currency=GBP&empty=")
        self.assertEqual(body, {"amount": "500", "currency": "GBP", "empty": ""})

    def test_multipart(self):
        ct = 'multipart/form-data; boundary="B"'
        raw = (
            b'--B\r\nContent-Disposition: form-data; name="amount"\r\n\r\n500\r\n'
            b'--B\r\nContent-Disposition: form-data; name="doc"; filename="a.txt"\r\n'
            b"Content-Type: text/plain\r\n\r\nhello\r\n--B--\r\n"
        )
        body = parse_body(ct, raw)
        self.assertEqual(body["amount"], "500")
        self.assertEqual(body["doc"]["filename"], "a.txt")
        self.assertEqual(body["doc"]["content"], b"hello")

    def test_unknown_content_type(self):
        self.assertEqual(parse_body("text/plain", b"hi"), {"_raw": "hi"})


class TestBuildRequestView(unittest.TestCase):
    def test_host_lowercased_and_port_split(self):
        view = build_request_view("get", "http://Shop.Example.COM:8500/p?q=1", "", None)
        self.assertEqual(view.host, "shop.example.com")
        self.assertEqual(view.port, 8500)
        self.assertEqual(view.method, "GET")

    def test_query_multivalue_and_blank(self):
        view = build_request_view("GET", "http://h.example/p?a=1&a=2&b=", "", None)
        self.assertEqual(view.query, {"a": ["1", "2"], "b": [""]})

    def test_body_parsed(self):
        view = build_request_view(
            "POST", "http://h.example/checkout", "application/json", b'{"amount": 500}'
        )
        self.assertEqual(view.body, {"amount": 500})


class TestRequestViewFromFlow(unittest.TestCase):
    def test_duck_typed_flow(self):
        flow = FakeFlow(FakeRequest(
            "POST",
            "http://shop.example.com/checkout?src=app",
            content_type="application/json",
            raw_content=json.dumps({"amount": 500}).encode(),
        ))
        view = request_view_from_flow(flow)
        self.assertEqual(view.host, "shop.example.com")
        self.assertEqual(view.body, {"amount": 500})
        self.assertEqual(view.query, {"src": ["app"]})


if __name__ == "__main__":
    unittest.main()
