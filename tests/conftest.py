"""Offline suite safety: fail before any accidental external provider traffic."""
import socket
import pytest
import os

# Collection and deterministic SDK stubs never need a developer's real key.
os.environ.setdefault('API_KEY', 'offline-test-not-a-real-key')

# Set before app imports or any LangGraph invocation. No async telemetry in offline tests.
os.environ['ENABLE_EXTERNAL_TRACING'] = 'false'
os.environ['LANGSMITH_TRACING'] = 'false'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
os.environ['LANGCHAIN_TRACING'] = 'false'
os.environ['LANGCHAIN_CALLBACKS_BACKGROUND'] = 'false'


@pytest.fixture(autouse=True)
def deny_external_network(monkeypatch):
    original = socket.socket.connect
    def guarded(sock, address):
        host = address[0] if isinstance(address, tuple) else ""
        if host not in ("127.0.0.1", "::1", "localhost", ""):
            raise RuntimeError("External network forbidden in offline tests")
        return original(sock, address)
    monkeypatch.setattr(socket.socket, "connect", guarded)
    # Socket addresses alone miss external traffic tunnelled through localhost proxies.
    import httpx
    import requests
    from urllib.parse import urlsplit
    def allowed(url):
        host = urlsplit(str(url)).hostname
        if host not in ('127.0.0.1','::1','localhost','testserver',None):
            raise RuntimeError('External request URL forbidden in offline tests')
    send = httpx.Client.send
    async_send = httpx.AsyncClient.send
    request = requests.Session.request
    def safe_send(self, req, *args, **kw):
        allowed(req.url); return send(self, req, *args, **kw)
    async def safe_async_send(self, req, *args, **kw):
        allowed(req.url); return await async_send(self, req, *args, **kw)
    def safe_request(self, method, url, *args, **kw):
        allowed(url); return request(self, method, url, *args, **kw)
    monkeypatch.setattr(httpx.Client, 'send', safe_send)
    monkeypatch.setattr(httpx.AsyncClient, 'send', safe_async_send)
    monkeypatch.setattr(requests.Session, 'request', safe_request)
