
import os
import re
import time
import json
import math
import typing as T
from dataclasses import dataclass
from urllib.parse import urlencode
from datetime import datetime, timedelta, timezone

import requests
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

OPENALEX_BASE = "https://api.openalex.org"
DEFAULT_PER_PAGE = 200

def _normalize_author_id(x: str) -> str:
    """
    Accepts variations like 'A12345', 'https://openalex.org/A12345', 'authors/A12345'.
    Returns the short OpenAlex Author ID like 'A12345' (always uppercase A).
    """
    if not x:
        return ""
    x = str(x).strip()
    # pull last path segment
    m = re.search(r'([aA]\d+)$', x)
    if m:
        return "A" + m.group(1)[1:]
    return x

def _polite_params() -> dict:
    polite = {}
    email = os.environ.get("OPENALEX_EMAIL")
    if email:
        polite["mailto"] = email
    return polite

class OpenAlexError(Exception):
    pass

@retry(wait=wait_exponential(multiplier=1, min=1, max=10),
       stop=stop_after_attempt(5),
       retry=retry_if_exception_type((requests.RequestException, OpenAlexError)))
def _get(url: str, params: dict) -> dict:
    """GET with retries and basic error handling."""
    merged = dict(params or {})
    merged.update(_polite_params())
    resp = requests.get(url, params=merged, timeout=30)
    if resp.status_code == 429:
        # too many requests; backoff based on headers if present
        retry_after = int(resp.headers.get("Retry-After", "1"))
        time.sleep(max(retry_after, 1))
        raise OpenAlexError("Rate limited")
    resp.raise_for_status()
    return resp.json()

def page_works(filter_str: str, select: T.Optional[str] = None) -> T.Iterator[dict]:
    """
    Cursor-paginate /works with given filter string.
    Yields Work objects.
    """
    cursor = "*"
    params = {"filter": filter_str, "per-page": DEFAULT_PER_PAGE, "cursor": cursor}
    if select:
        params["select"] = select
    while True:
        data = _get(f"{OPENALEX_BASE}/works", params)
        results = data.get("results", [])
        if not results:
            break
        for w in results:
            yield w
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break
        params["cursor"] = cursor

def reconstruct_abstract(inv: dict) -> str:
    """
    Reverse OpenAlex abstract_inverted_index into plaintext.
    """
    if not inv:
        return ""
    # inv is dict[word] = [positions]
    pairs = []
    for word, positions in inv.items():
        for pos in positions:
            pairs.append((pos, word))
    if not pairs:
        return ""
    pairs.sort(key=lambda x: x[0])
    # Positions are 0-based; insert spaces
    words = [w for (_, w) in pairs]
    return " ".join(words)

def today_utc_date():
    return datetime.now(timezone.utc).date()

def dates_last_n_days(n: int):
    end = today_utc_date()
    start = end - timedelta(days=n)
    return start.isoformat(), end.isoformat()

def normalized_faculty_from_csv(csv_path: str):
    """
    Returns list of dicts: {'name': '...', 'openalex_id': 'A...'}
    Accepts flexible column headers like Name, Full Name; OpenAlexID, openalex_id, openalex, id
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    # find likely columns
    cols = {c.lower().strip(): c for c in df.columns}
    name_col = None
    for k in ["name", "full name", "full_name"]:
        if k in cols:
            name_col = cols[k]
            break
    if name_col is None:
        # choose first string-like column
        name_col = df.columns[0]
    id_col = None
    for k in ["openalexid", "openalex_id", "openalex", "id", "author_id"]:
        if k in cols:
            id_col = cols[k]
            break
    if id_col is None and len(df.columns) >= 2:
        id_col = df.columns[1]
    faculty = []
    for _, row in df.iterrows():
        name = str(row.get(name_col, "")).strip()
        aid = _normalize_author_id(row.get(id_col, ""))
        if aid:
            faculty.append({"name": name, "openalex_id": aid, "openalex_url": f"https://openalex.org/authors/{aid.lower()}"})
    return faculty

def collect_recent_works_for_authors(author_ids, start_date: str, end_date: str) -> dict:
    """
    Fetch works for list of OpenAlex Author IDs in date window.
    Returns dict keyed by work_id with merged info and a set of matching authors from the cohort.
    """
    work_map = {}
    # Build filter with pipe-separated authors up to 100 (OpenAlex allows many values)
    # If >100 authors, we chunk the requests.
    CHUNK = 80
    for i in range(0, len(author_ids), CHUNK):
        chunk = author_ids[i:i+CHUNK]
        author_filter_val = "|".join(chunk)
        filter_str = f"author.id:{author_filter_val},from_publication_date:{start_date},to_publication_date:{end_date}"
        select = None  # we want full fields
        for w in page_works(filter_str, select=select):
            wid = w.get("id", "")
            if not wid:
                # sometimes 'ids.openalex' is the canonical; but 'id' should be present
                wid = w.get("ids", {}).get("openalex", "")
            if not wid:
                continue
            if wid not in work_map:
                work_map[wid] = w
            else:
                # Merge: no real need since works are identical; keep first
                pass
    return work_map

def extract_journal_name(work: dict) -> str:
    """
    Use primary_location.source.display_name if present; fall back to best_oa_location.source.display_name or empty.
    """
    pl = work.get("primary_location") or {}
    src = pl.get("source") or {}
    name = src.get("display_name") or ""
    if not name:
        boa = work.get("best_oa_location") or {}
        src2 = (boa.get("source") or {})
        name = src2.get("display_name") or ""
    return name

def extract_authors_list(work: dict) -> T.List[dict]:
    authors = []
    for a in (work.get("authorships") or []):
        author = a.get("author") or {}
        if not author:
            continue
        authors.append({
            "id": author.get("id", ""),
            "openalex_id": _normalize_author_id(author.get("id", "")),
            "display_name": author.get("display_name", ""),
            "orcid": author.get("orcid", ""),
            "author_position": a.get("author_position", "")
        })
    return authors

def match_cohort_authors(authors: T.List[dict], cohort_ids: T.Set[str]) -> T.List[dict]:
    matches = []
    for a in authors:
        aid = a.get("openalex_id")
        if aid and aid in cohort_ids:
            matches.append(a)
    return matches
