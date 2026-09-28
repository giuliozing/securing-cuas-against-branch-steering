"""Lightweight site crawler that gathers *observable* structure.

Fetches the homepage and a few same-origin linked pages, then extracts the
signals an LLM (or the deterministic fallback) needs to propose agent calls:
page titles/descriptions, forms (action/method/inputs), same-origin links,
and any pre-existing machine-readable maps (``robots.txt``, ``sitemap.xml``,
``/sitemap.json``). Purely read-only HTTP GETs against the owner-supplied site.

Optionally the crawl carries a session (``cookies``): most sites expose their
real agentic surface only to a logged-in user, and the owner running MESA on
their own site *is* logged in. See :func:`cookies_from_storage_state`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

_UA = "mesa/1.0 (+https://example.local)"
_MAX_TEXT = 4000
_REFRESH_URL_RE = re.compile(r"url\s*=\s*(.+)$", re.IGNORECASE)
_MAX_REFRESH_HOPS = 3

# Links never followed when the crawl carries a session: a plain GET on any of
# these ends the session and the rest of the crawl silently degrades to the
# logged-out surface. Matched against the URL path + query.
_SESSION_ENDING_RE = re.compile(
    r"(?:^|[/?&=])(logout|log_out|log-out|signout|sign_out|sign-out|destroy)(?:$|[/?&#.])",
    re.IGNORECASE,
)


def cookies_from_storage_state(path: str, host: str = "") -> dict[str, str]:
    """Read a Playwright ``storage_state`` JSON into a name→value cookie dict.

    This is the format produced by Playwright's ``context.storage_state()``
    (and so by WebArena's ``browser_env/auto_login.py``), which makes an
    already-authenticated browser session reusable by the crawler. When
    ``host`` is given, only cookies whose domain matches it are returned.
    """
    with open(path, "r") as fh:
        state = json.load(fh)
    host = (host or "").lower().rsplit(":", 1)[0]
    out: dict[str, str] = {}
    for c in state.get("cookies", []):
        name, value = c.get("name"), c.get("value")
        if not name or value is None:
            continue
        dom = (c.get("domain") or "").lstrip(".").lower()
        if host and dom and not (host == dom or host.endswith("." + dom)):
            continue
        out[name] = value
    return out


def _meta_refresh_target(soup: BeautifulSoup, base: str) -> str | None:
    """Return the absolute URL of an HTML ``<meta http-equiv=refresh>`` redirect.

    httpx follows HTTP-level redirects but not HTML meta-refresh landing pages
    (common on legacy apex domains), so the crawler resolves them itself.
    """
    tag = soup.find("meta", attrs={"http-equiv": re.compile(r"^\s*refresh\s*$", re.I)})
    if not tag:
        return None
    match = _REFRESH_URL_RE.search(tag.get("content") or "")
    if not match:
        return None
    return urljoin(base, match.group(1).strip().strip("'\""))


@dataclass
class FormInfo:
    action: str
    method: str
    inputs: list[str] = field(default_factory=list)


@dataclass
class PageInfo:
    url: str
    title: str = ""
    description: str = ""
    forms: list[FormInfo] = field(default_factory=list)
    links: list[str] = field(default_factory=list)


@dataclass
class CrawlEvidence:
    base_url: str
    domain: str
    pages: list[PageInfo] = field(default_factory=list)
    robots: str = ""
    sitemap_xml_urls: list[str] = field(default_factory=list)
    existing_agent_sitemap: list[dict] | None = None
    errors: list[str] = field(default_factory=list)
    authenticated: bool = False
    # Character budget for :meth:`summary`. The default keeps the historical
    # behaviour; note that it truncates from the tail, so on a wide crawl the
    # last pages never reach the LLM at all — raise it when ``max_pages`` is
    # large, or the extra pages are gathered and then silently discarded.
    text_budget: int = _MAX_TEXT * 4

    @property
    def origin(self) -> str:
        """Scheme + host + port the calls should be built on.

        ``domain`` deliberately drops the scheme and port (it is the identity
        used for same-site link filtering), so it cannot be used to build URLs:
        a site on ``http://host:8023`` would come out as ``https://host``.
        Taken from the first page's final URL so redirects are honoured.
        """
        src = self.pages[0].url if self.pages else self.base_url
        parts = urlparse(src)
        return f"{parts.scheme}://{parts.netloc}" if parts.scheme else src.rstrip("/")

    def summary(self) -> str:
        """Compact, injection-tolerant textual digest for the LLM prompt."""
        lines: list[str] = [f"SITE: {self.base_url}", f"DOMAIN: {self.domain}"]
        if self.authenticated:
            lines.append("SESSION: crawled as a logged-in user")
        lines.append("")
        for p in self.pages:
            lines.append(f"PAGE {p.url}")
            if p.title:
                lines.append(f"  title: {p.title[:160]}")
            if p.description:
                lines.append(f"  meta: {p.description[:200]}")
            for f in p.forms:
                fields = ", ".join(f.inputs[:20]) or "(no named inputs)"
                lines.append(f"  FORM {f.method} {f.action}  inputs: {fields}")
            if p.links:
                lines.append("  links: " + ", ".join(p.links[:25]))
            lines.append("")
        if self.sitemap_xml_urls:
            lines.append("SITEMAP.XML entries (sample):")
            lines += [f"  {u}" for u in self.sitemap_xml_urls[:40]]
            lines.append("")
        if self.existing_agent_sitemap:
            lines.append(f"EXISTING /sitemap.json: {len(self.existing_agent_sitemap)} entries present")
        text = "\n".join(lines)
        return text[: self.text_budget]


def _norm_domain(url: str) -> str:
    net = urlparse(url).netloc.lower()
    if ":" in net and not net.startswith("["):
        net = net.rsplit(":", 1)[0]
    return net


def crawl(
    base_url: str,
    *,
    max_pages: int = 6,
    timeout: float = 10.0,
    cookies: dict[str, str] | None = None,
    text_budget: int | None = None,
) -> CrawlEvidence:
    """Crawl ``base_url`` breadth-first (depth 1) up to ``max_pages`` pages.

    ``cookies`` (see :func:`cookies_from_storage_state`) makes the crawl run as
    a logged-in user; logout links are then skipped so the session survives.
    ``text_budget`` overrides how much of the evidence :meth:`CrawlEvidence.summary`
    passes on (see the field's note on tail truncation).
    """
    if not urlparse(base_url).scheme:
        base_url = "https://" + base_url
    domain = _norm_domain(base_url)
    ev = CrawlEvidence(base_url=base_url, domain=domain, authenticated=bool(cookies))
    if text_budget is not None:
        ev.text_budget = text_budget

    headers = {"User-Agent": _UA, "Accept": "text/html,application/xhtml+xml"}
    with httpx.Client(
        follow_redirects=True, timeout=timeout, headers=headers, cookies=cookies or None
    ) as client:
        _fetch_meta(client, base_url, domain, ev)

        queue: list[str] = [base_url]
        seen: set[str] = set()
        visited_final: set[str] = set()
        while queue and len(ev.pages) < max_pages:
            url = queue.pop(0)
            if url in seen:
                continue
            seen.add(url)
            page = _fetch_page(client, url, ev)
            if page is None:
                continue
            # Dedup by the *final* (post-redirect/refresh) URL: distinct entry
            # URLs (``/web`` vs ``/web/``, apex vs meta-refresh target) can land
            # on the same page.
            if page.url in visited_final:
                continue
            visited_final.add(page.url)
            # Canonical domain is the *final* host of the first page we land
            # on, not the input host — so redirects (e.g. apex → www) don't
            # cause every same-site link to be filtered out.
            if not ev.pages:
                ev.domain = _norm_domain(page.url)
            ev.pages.append(page)
            for link in page.links:
                if link in seen or _norm_domain(link) != ev.domain:
                    continue
                if cookies and _SESSION_ENDING_RE.search(urlparse(link).path + "?" + (urlparse(link).query or "")):
                    continue
                queue.append(link)
    return ev


def _fetch_meta(client: httpx.Client, base_url: str, domain: str, ev: CrawlEvidence) -> None:
    # robots.txt
    try:
        r = client.get(urljoin(base_url, "/robots.txt"))
        if r.status_code == 200 and "text" in r.headers.get("content-type", ""):
            ev.robots = r.text[:_MAX_TEXT]
    except httpx.HTTPError as e:
        ev.errors.append(f"robots.txt: {e}")

    # sitemap.xml — parsed with the html parser (no lxml dependency); <loc>
    # tags are simple enough that the lenient parser recovers them fine.
    try:
        r = client.get(urljoin(base_url, "/sitemap.xml"))
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            ev.sitemap_xml_urls = [
                loc.get_text(strip=True) for loc in soup.find_all("loc")
            ][:100]
    except httpx.HTTPError as e:
        ev.errors.append(f"sitemap.xml: {e}")

    # pre-existing agent sitemap
    try:
        r = client.get(urljoin(base_url, "/sitemap.json"))
        if r.status_code == 200:
            data = json.loads(r.text)
            if isinstance(data, list):
                ev.existing_agent_sitemap = data
    except (httpx.HTTPError, json.JSONDecodeError):
        pass


def _fetch_page(
    client: httpx.Client, url: str, ev: CrawlEvidence, _hops: int = 0
) -> PageInfo | None:
    try:
        r = client.get(url)
    except httpx.HTTPError as e:
        ev.errors.append(f"{url}: {e}")
        return None
    if r.status_code >= 400 or "html" not in r.headers.get("content-type", ""):
        return None

    soup = BeautifulSoup(r.text, "html.parser")
    final_url = str(r.url)

    # Follow HTML meta-refresh redirects (bounded) — legacy apex pages often
    # serve an empty shell whose only content is a refresh to the real site.
    if _hops < _MAX_REFRESH_HOPS:
        target = _meta_refresh_target(soup, final_url)
        if target and target != final_url:
            return _fetch_page(client, target, ev, _hops + 1)

    page = PageInfo(url=final_url)
    # Filter links against the page's *own* final host (post-redirect), so a
    # page reached via redirect still yields its same-site links.
    page_host = _norm_domain(final_url)
    if soup.title and soup.title.string:
        page.title = soup.title.string.strip()
    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        page.description = meta["content"].strip()

    for form in soup.find_all("form"):
        action = urljoin(final_url, form.get("action") or final_url)
        method = (form.get("method") or "GET").upper()
        inputs = []
        for tag in form.find_all(["input", "select", "textarea"]):
            name = tag.get("name")
            if name:
                inputs.append(name)
        page.forms.append(FormInfo(action=action, method=method, inputs=inputs))

    links: list[str] = []
    for a in soup.find_all("a", href=True):
        href = urljoin(final_url, a["href"]).split("#")[0]
        if _norm_domain(href) == page_host and href not in links:
            links.append(href)
    page.links = links[:60]
    return page
