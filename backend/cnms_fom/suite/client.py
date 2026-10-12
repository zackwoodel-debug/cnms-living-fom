"""Client for the materials-db service (``materials_db.access.http``, the ``/suite`` routes).

materials-db owns the reference data. This platform reads it over HTTP so the same code works whether the service runs on this machine or on a
remote host reached through an SSH tunnel (``ssh -L 8000:localhost:8000 <host>``); it never opens the release file itself here and never writes to it.
A refusal from the service (HTTP 422) comes back as ``SuiteRefusal`` with the service's own reason; an unreachable service is ``SuiteUnavailable``.
"""
from __future__ import annotations

import json

import httpx


class SuiteRefusal(RuntimeError):
    """The service understood the request and said no, with the reason."""


class SuiteUnavailable(RuntimeError):
    """The service could not be reached or answered with something that is not part of its contract."""


class MaterialsDbClient:
    def __init__(self, base_url: str, *, timeout: float = 30.0, transport: httpx.BaseTransport | None = None):
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, transport=transport)

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, **kw) -> httpx.Response:
        try:
            response = self._client.request(method, path, **kw)
        except httpx.HTTPError as exc:
            raise SuiteUnavailable(
                f"materials-db service not reachable at {self._client.base_url} ({type(exc).__name__}). Start it with "
                "`MATERIALS_DB_RELEASE=<release> uvicorn materials_db.access.http:app --port 8000`, or open the SSH tunnel to the host that runs it."
            ) from exc
        if response.status_code == 422:
            detail = response.json().get("detail", "refused")
            raise SuiteRefusal(detail if isinstance(detail, str) else json.dumps(detail))
        if response.status_code >= 400:
            raise SuiteUnavailable(f"materials-db service answered HTTP {response.status_code} for {method} {path}")
        return response

    def info(self) -> dict:
        return self._request("GET", "/suite/info").json()

    def materials(self, q: str | None = None, limit: int = 25) -> list[dict]:
        return self._request("GET", "/suite/materials", params={"q": q, "limit": limit}).json()

    def evidence(self, material_id: int) -> dict:
        return self._request("GET", f"/suite/materials/{material_id}/evidence").json()

    def readiness(self, selection: dict, technique: str, window: dict | None = None, conditions: dict | None = None) -> dict:
        return self._request("POST", "/suite/readiness", json=dict(selection=selection, technique=technique, window=window, conditions=conditions)).json()

    def bundle(self, request: dict) -> tuple[bytes, dict]:
        response = self._request("POST", "/suite/bundle", json=request)
        return response.content, json.loads(response.headers.get("x-bundle-summary", "{}"))

    def optical_search(self, **params) -> dict:
        return self._request("GET", "/suite/optical/search", params={k: v for k, v in params.items() if v is not None}).json()
