"""mitmproxy addon: route STEER-Bench's realistic hostnames to the local site.

"""

from __future__ import annotations

import sys
from pathlib import Path

from mitmproxy import ctx, http

# This file is loaded by `mitmdump -s`, i.e. as a SCRIPT by mitmproxy's own
# interpreter, which is not necessarily the one that has `steerbench` importable —
# mitmdump is deliberately treated as an external binary (`config.mitmdump`), so it
# may live in its own environment. Adding the package's source root is therefore
# correct here: the alternative is requiring the benchmark to be installed into whatever
# environment ships mitmproxy. `registry` is stdlib-only, so nothing else is needed.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


class SiteForward:
    def __init__(self) -> None:
        self.port = 8700
        self.hosts: set[str] = set()

    def load(self, loader) -> None:
        loader.add_option("site_port", int, 8700, "loopback port of the bench site")
        loader.add_option("site_hosts", str, "", "comma-separated hosts to remap (empty = all from registry)")

    def configure(self, updated) -> None:
        self.port = int(ctx.options.site_port)
        raw = (ctx.options.site_hosts or "").strip()
        if raw:
            self.hosts = {h.strip().lower() for h in raw.split(",") if h.strip()}
        else:
            from steerbench.tasks import registry as reg
            self.hosts = set(reg.all_hosts())

    def request(self, flow: http.HTTPFlow) -> None:
        if flow.response is not None:      # enforcer already produced a 403
            return
        host = (flow.request.host or "").lower()
        if host not in self.hosts:
            return
        host_header = flow.request.headers.get("Host")
        flow.request.host = "127.0.0.1"
        flow.request.port = self.port
        flow.request.scheme = "http"
        if host_header is not None:
            flow.request.headers["Host"] = host_header


addons = [SiteForward()]
