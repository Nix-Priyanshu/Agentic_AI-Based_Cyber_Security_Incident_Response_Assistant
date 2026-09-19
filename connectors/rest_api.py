"""Connector for a security web application that exposes a REST API."""
from typing import Any, List, Tuple

import requests

from .base import BaseConnector, ConnectorError
from .models import (
    Host, Incident, Indicators, Network, Technique, TimelineEvent,
    normalize_severity, technique_name,
)

# How the source app labels an indicator -> which list it goes into.
_INDICATOR_LISTS = {
    "ip": "ips", "domain": "domains", "hash": "hashes",
    "email": "emails", "cve": "cves", "url": "urls",
}


def to_common_format(native: dict, source: str) -> Incident:
    """Convert one incident from the source app's own format to the common format."""
    incident_id = str(native.get("id", "")).strip()
    if not incident_id:
        raise ConnectorError("The server returned an incident without an id.")

    asset = native.get("asset") or {}
    net = native.get("network") or {}

    indicators = Indicators()
    for item in native.get("indicators") or []:
        target = _INDICATOR_LISTS.get(str(item.get("type", "")).lower())
        value = str(item.get("value", "")).strip()
        if target and value and value not in getattr(indicators, target):
            getattr(indicators, target).append(value)

    return Incident(
        id=incident_id,
        source=source,
        title=str(native.get("title", "")),
        severity=normalize_severity(native.get("risk")),
        status=str(native.get("state", "")).capitalize(),
        created_at=str(native.get("created", "")),
        summary=str(native.get("description", "")),
        threat_actor=str(native.get("actor") or ""),
        host=Host(
            hostname=str(asset.get("name", "")),
            os=str(asset.get("os", "")),
            department=str(asset.get("dept", "")),
            username=str(asset.get("owner", "")),
            ip=str(asset.get("ip", "")),
        ),
        network=Network(
            source_ip=str(net.get("src", "")),
            destination_ips=[str(ip) for ip in net.get("dst") or []],
            protocol=str(net.get("proto", "")),
            port=str(net.get("dport", "")),
            country=str(net.get("geo", "")),
            bytes_out=str(net.get("bytes_out", "")),
        ),
        indicators=indicators,
        techniques=[
            Technique(id=str(t), name=technique_name(str(t)))
            for t in native.get("attack_techniques") or []
        ],
        timeline=[
            TimelineEvent(time=str(e.get("ts", "")), event=str(e.get("msg", "")))
            for e in native.get("events") or []
        ],
        raw=native,
    )


class RestApiConnector(BaseConnector):
    type_label = "REST API"

    def __init__(self, name: str, base_url: str, api_key: str, timeout: int = 10):
        super().__init__(name)
        base_url = (base_url or "").strip().rstrip("/")
        if base_url and "://" not in base_url:
            base_url = "http://" + base_url
        self.base_url = base_url
        self.api_key = (api_key or "").strip()
        self.timeout = timeout

    # ---- HTTP helper ----
    def _get(self, path: str, authenticated: bool = True) -> Any:
        if not self.base_url:
            raise ConnectorError("Enter the base URL of the web application.")
        headers = {"X-API-Key": self.api_key} if authenticated else {}
        try:
            response = requests.get(self.base_url + path, headers=headers, timeout=self.timeout)
        except requests.exceptions.Timeout:
            raise ConnectorError(f"The server at {self.base_url} did not answer in time.")
        except requests.exceptions.InvalidURL:
            raise ConnectorError(f"'{self.base_url}' is not a valid URL.")
        except requests.exceptions.ConnectionError:
            raise ConnectorError(
                f"Could not reach {self.base_url}. Check the URL and that the server is running."
            )
        except requests.exceptions.RequestException as exc:
            raise ConnectorError(f"Request failed: {exc}")

        if response.status_code in (401, 403):
            raise ConnectorError("The server rejected the API key.")
        if response.status_code == 404:
            raise ConnectorError(f"Nothing found at {path}. Is this the right server?")
        if not response.ok:
            raise ConnectorError(f"The server returned an error (HTTP {response.status_code}).")
        try:
            return response.json()
        except ValueError:
            raise ConnectorError("The server did not return JSON.")

    # ---- connector interface ----
    def test_connection(self) -> Tuple[bool, str]:
        try:
            self._get("/health", authenticated=False)      # reachable?
            data = self._get("/api/v1/incidents")          # key accepted?
        except ConnectorError as exc:
            return False, str(exc)
        count = data.get("count", len(data.get("incidents", []))) if isinstance(data, dict) else 0
        return True, f"Connected. {count} incident(s) available."

    def fetch_incidents(self) -> List[Incident]:
        listing = self._get("/api/v1/incidents")
        summaries = listing.get("incidents", []) if isinstance(listing, dict) else []
        incidents = []
        for summary in summaries:
            incident_id = summary.get("id")
            if not incident_id:
                continue
            detail = self._get(f"/api/v1/incidents/{incident_id}")
            incidents.append(to_common_format(detail, self.name))
        return incidents
