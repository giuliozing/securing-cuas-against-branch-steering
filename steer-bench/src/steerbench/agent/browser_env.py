"""Playwright browser environment for the STEER-Bench live GUI loop."""

from __future__ import annotations

import base64
import time

VIEWPORT_W = 1000
VIEWPORT_H = 760


class BrowserEnv:
    def __init__(self, headless: bool = True,
                 view_w: int = VIEWPORT_W, view_h: int = VIEWPORT_H) -> None:
        self.view_w, self.view_h = view_w, view_h
        self._headless = headless
        self._pw = None
        self._browser = None
        self.page = None

    def start(self) -> "BrowserEnv":
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self._headless)
        self.page = self._browser.new_page(viewport={"width": self.view_w, "height": self.view_h})
        return self

    def goto(self, url: str) -> None:
        self.page.goto(url, wait_until="networkidle")

    def screenshot_b64(self) -> str:
        png = self.page.screenshot(type="png")
        return base64.b64encode(png).decode()

    def status_text(self) -> str:
        try:
            return (self.page.text_content("#status") or "").strip()
        except Exception:
            return ""

    def apply(self, act) -> None:
        """Execute a grounded Action on the page."""
        t = act.type
        if t in ("click", "left_double", "right_single") and act.x is not None:
            if t == "left_double":
                self.page.mouse.dblclick(act.x, act.y)
            elif t == "right_single":
                self.page.mouse.click(act.x, act.y, button="right")
            else:
                self.page.mouse.click(act.x, act.y)
        elif t == "type":
            self.page.keyboard.type(act.text)
        elif t == "hotkey":
            for key in (act.text or "").split():
                self.page.keyboard.press(key)
        elif t == "scroll" and act.x is not None:
            dy = {"up": -400, "down": 400}.get(act.direction, 0)
            dx = {"left": -400, "right": 400}.get(act.direction, 0)
            self.page.mouse.move(act.x, act.y)
            self.page.mouse.wheel(dx, dy)
        elif t == "wait":
            time.sleep(1.0)
        # finished: no-op (loop terminates on it)
        self.page.wait_for_timeout(400)  # let JS (confirm bridge) settle

    def close(self) -> None:
        try:
            if self._browser:
                self._browser.close()
            if self._pw:
                self._pw.stop()
        except Exception:
            pass
