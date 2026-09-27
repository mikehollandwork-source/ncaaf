"""Shared helpers: HTTP, time parsing, number coercion, CSV storage."""
from __future__ import annotations

import csv
import gzip
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger("ncaaf")

USER_AGENT = "ncaaf-line-tracker/1.0 (research; hourly polling)"


def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
        respect_retry_after_header=True,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=16))
    s.headers["User-Agent"] = USER_AGENT
    s.headers["Accept"] = "application/json"
    return s


def get_json(session: requests.Session, url: str, params: dict | None = None,
             timeout: float = 30, min_interval: float = 0.0) -> Any:
    resp = session.get(url, params=params, timeout=timeout)
    if min_interval:
        time.sleep(min_interval)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------- time

def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(v: Any) -> datetime | None:
    """Parse the assorted timestamp formats the sources use into aware UTC."""
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        ts = float(v)
        if ts > 1e12:  # milliseconds
            ts /= 1000
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    s = str(v).strip()
    if re.fullmatch(r"\d{10,13}", s):
        return parse_time(int(s))
    s = s.replace(" ", "T", 1) if re.match(r"\d{4}-\d\d-\d\d \d", s) else s
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    # "+00" -> "+00:00"
    s = re.sub(r"([+-]\d\d)$", r"\1:00", s)
    # ESPN uses "2026-09-26T16:00Z" (no seconds)
    if re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d[+-]\d\d:\d\d", s):
        s = s[:16] + ":00" + s[16:]
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ---------------------------------------------------------------- numbers

def to_float(v: Any) -> float | None:
    if v is None or v == "" or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().upper()
    if s in ("EVEN", "EV"):
        return 100.0
    if s in ("PK", "PICK", "PICKEM"):
        return 0.0
    s = re.sub(r"^[OU]\s*", "", s)  # ESPN totals: "o52.5"
    try:
        return float(s)
    except ValueError:
        return None


def json_list(v: Any) -> list:
    """Polymarket encodes arrays as JSON strings."""
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v.startswith("["):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return []
    return []


# ---------------------------------------------------------------- storage

def write_table(data_dir: Path, table: str, rows: Iterable[dict], ts: datetime) -> Path | None:
    """Append-only storage: one gzip CSV per table per run.

    data/<table>/<YYYY-MM-DD>/<table>_<YYYYmmddTHHMMSSZ>.csv.gz
    New files per run means git stores only new blobs (no rewriting a growing DB).
    """
    rows = list(rows)
    if not rows:
        return None
    fields: list[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    out_dir = data_dir / table / ts.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{table}_{ts.strftime('%Y%m%dT%H%M%SZ')}.csv.gz"
    with gzip.open(path, "wt", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _cell(v) for k, v in r.items()})
    return path


def _cell(v: Any) -> Any:
    if isinstance(v, (dict, list)):
        return json.dumps(v, separators=(",", ":"))
    if isinstance(v, datetime):
        return iso(v)
    return v


def env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default
