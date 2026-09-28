"""Who may read the real wire body — the `/quote` endpoints and the harness token.

The thing being protected is the benchmark's premise. `/<theme>/quote` and
`/<theme>/__item/<i>/quote` return the body the guarded action will carry, *after* the
attack's divergence has been applied. They exist because the SITE uses them: it resolves
the authoritative body server-side and issues the action itself, which is exactly what
makes "perceived ≠ real" a fact no agent can see by looking.

They also sit on the base URL an adapter is handed as `TaskView.page_url`. Until this
was closed, an adapter — or a browser navigating one URL further along — could read the
answer and score ~0% ASR with full utility and no defence at all.

**What these tests do and do not claim.** They assert that the door is shut to a caller
without the token and open to the harness, and that every scored path configures a token.
They do NOT claim protection against a hostile in-process adapter: an adapter is Python
running in the runner's own process and can read anything the runner can, including this.
Nothing short of process isolation changes that, and asserting otherwise would be worse
than the hole. What is closed is the shortcut available *without realising it is
cheating*.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from steerbench import config
from steerbench.adapters.base import TaskView
from steerbench.harness import driver
from steerbench.site import app as site
from steerbench.tasks import registry as reg

pytestmark = pytest.mark.substrate

TOKEN = "test-harness-token"


@pytest.fixture
def guarded(monkeypatch):
    """The site with a token configured — i.e. the way every scored run brings it up."""
    monkeypatch.setattr(site, "HARNESS_TOKEN", TOKEN)
    task = next(t for t in reg.TASKS if t.on_http_axis)
    client = site.app.test_client()
    client.post("/__config", json=task.benign_config())
    return client, task


@pytest.fixture
def guarded_workflow(monkeypatch):
    monkeypatch.setattr(site, "HARNESS_TOKEN", TOKEN)
    task = next(t for t in reg.TASKS if t.on_step_axis)
    client = site.app.test_client()
    client.post("/__config", json=task.attack_config())
    return client, task


# --- the door -----------------------------------------------------------


def test_the_agent_cannot_read_the_real_body(guarded):
    client, task = guarded
    r = client.get(f"/{task.theme}/quote")
    assert r.status_code == 403, "an unauthenticated read of the answer must be refused"
    assert "not readable by the agent" in r.get_json()["error"]


def test_the_agent_cannot_read_a_rows_real_body(guarded_workflow):
    client, task = guarded_workflow
    assert client.get(f"/{task.theme}/__item/0/quote").status_code == 403


def test_a_wrong_token_is_not_a_token(guarded):
    client, task = guarded
    r = client.get(f"/{task.theme}/quote", headers={site.HARNESS_HEADER: "nope"})
    assert r.status_code == 403


def test_the_harness_still_gets_the_body(guarded):
    client, task = guarded
    r = client.get(f"/{task.theme}/quote", headers={site.HARNESS_HEADER: TOKEN})
    assert r.status_code == 200
    assert set(r.get_json()["body"]) == {f.name for f in task.action_fields}


def test_the_refusal_explains_itself(guarded):
    """A 403 with no reason gets worked around; a 403 that says why gets read. This one
    names the design and points at the fair-play section of the adapter docs."""
    client, task = guarded
    body = client.get(f"/{task.theme}/quote").get_json()
    assert "docs/adapters.md" in body["why"]
    assert "server-side" in body["why"]


# --- what the closure must NOT have broken ------------------------------


def test_the_pages_are_unaffected(guarded):
    """The token gates the ANSWER, not the environment. A page that started refusing
    would take the benchmark offline for every agent, which is a worse failure than the
    hole it closes."""
    client, task = guarded
    for path in (f"/{task.theme}/", f"/{task.theme}/record", "/__state"):
        assert client.get(path).status_code == 200, path


def test_fail_open_when_no_token_is_configured(monkeypatch):
    """Deliberate, and the reason is that a refusal nobody can switch off gets routed
    around. `steerbench site` and `render_check` bring the site up by hand for
    inspection; they configure no token and must keep working. Every SCORED path goes
    through `stack.build`, which always sets one — the test below."""
    monkeypatch.setattr(site, "HARNESS_TOKEN", "")
    task = next(t for t in reg.TASKS if t.on_http_axis)
    client = site.app.test_client()
    client.post("/__config", json=task.benign_config())
    assert client.get(f"/{task.theme}/quote").status_code == 200


# --- the token itself ----------------------------------------------------


def test_tokens_are_unguessable_and_not_shared(monkeypatch):
    a, b = config.new_harness_token(), config.new_harness_token()
    assert a != b and len(a) >= 24


def test_the_token_is_read_at_call_time(monkeypatch):
    """Frozen at import it would be the empty string forever: `stack.build()` sets it
    after this package is imported, so a cached value would leave the harness itself
    unable to reach the endpoint it is the only legitimate caller of."""
    monkeypatch.setenv(config.HARNESS_TOKEN_ENV, "later")
    assert config.harness_token() == "later"
    assert config.harness_headers() == {config.HARNESS_HEADER: "later"}


def test_the_driver_speaks_as_the_harness(monkeypatch):
    monkeypatch.setenv(config.HARNESS_TOKEN_ENV, TOKEN)
    assert driver._harness_headers() == {config.HARNESS_HEADER: TOKEN}


def test_no_token_no_header(monkeypatch):
    monkeypatch.delenv(config.HARNESS_TOKEN_ENV, raising=False)
    assert driver._harness_headers() == {}


def test_the_token_never_reaches_the_adapter(monkeypatch):
    """`TaskView` is the whole of what an adapter is told. If the token ever appeared on
    it — as a field, or inside `page_url` — the closure would be decorative."""
    monkeypatch.setenv(config.HARNESS_TOKEN_ENV, TOKEN)
    view = TaskView(tid="x", theme="t", instruction="do it", channel="gui",
                    page_url="http://127.0.0.1:8700")
    assert TOKEN not in json.dumps(view.__dict__)
    assert not any("token" in f.lower() for f in view.__dict__)


def test_the_token_is_not_rendered_on_the_page(guarded):
    client, task = guarded
    assert TOKEN not in client.get(f"/{task.theme}/").get_data(as_text=True)


# --- the scored path, end to end -----------------------------------------


@pytest.mark.slow
def test_every_scored_run_configures_a_token(monkeypatch):
    """The claim this whole file rests on, proved against a real site process rather
    than a test client: `stack.build` is how both tracks bring the environment up, so if
    it sets a token then no scored run leaves the answer readable.

    Site-only (`proxy=False`), so it needs no defence stack — the same bring-up the open
    track uses, which is the track a third-party adapter runs on."""
    from steerbench.harness import stack

    monkeypatch.delenv(config.HARNESS_TOKEN_ENV, raising=False)
    st = stack.build(proxy=False, prefix="steerbench_quotegate_")
    try:
        token = config.harness_token()
        assert token, "stack.build must configure a token before spawning the site"
        task = next(t for t in reg.TASKS if t.on_http_axis)
        config.control("POST", "/__config", task.attack_config(), port=st.site_port)

        url = f"{st.direct}/{task.theme}/quote"
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(urllib.request.Request(url), timeout=10)
        assert e.value.code == 403

        got = driver._get(url, proxy=None)          # the harness, with the token
        assert set(got["body"]) == {f.name for f in task.action_fields}
    finally:
        st.close()
