import os
import re
import time
import json
import typing as T
from urllib.parse import urlencode
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime


import requests
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

OPENALEX_BASE = "https://api.openalex.org"
DEFAULT_PER_PAGE = 200

_NAN_LIKE = {"", "nan", "none", "null"}

def _normalize_author_id(x: T.Any) -> str:
    """
    Accepts variations like 'A12345', 'https://openalex.org/A12345', 'authors/A12345'.
    Returns canonical short OpenAlex Author ID like 'A12345' (uppercase 'A').
    Returns '' for blanks/NaN-ish values.
    """
    if x is None:
        return ""
    sx = str(x).strip()
    if not sx or sx.lower() in _NAN_LIKE:
        return ""
    # allow full URL or trailing segment
    m = re.search(r'(?:openalex\.org/)?([aA]\d+)$', sx)
    if m:
        return m.group(1).upper()
    # If it's exactly 'A' + digits but case odd
    m2 = re.match(r'^[aA]\d+$', sx)
    if m2:
        return sx.upper()
    return ""

def _author_uri(aid: str) -> str:
    """Convert 'A12345' -> 'https://openalex.org/A12345'"""
    return f"https://openalex.org/{aid}"

def _polite_params() -> dict:
    """
    Build OpenAlex 'polite pool' params.
    Prefer API key; fall back to mailto. Supports either OPENALEX_MAILTO or OPENALEX_EMAIL.
    """
    polite = {}

    api_key = os.getenv("OPENALEX_API_KEY")
    if api_key:
        polite["api_key"] = api_key

    # Accept either name; MAILTO takes precedence if both are set
    mailto = os.getenv("OPENALEX_MAILTO") or os.getenv("OPENALEX_EMAIL")
    if mailto:
        polite["mailto"] = mailto

    return polite


class OpenAlexError(Exception):
    pass


def _parse_retry_after(value: str) -> int:
    """
    Parse HTTP Retry-After header; return seconds to wait (>=1).
    Accepts either:
      - delta-seconds: "120"
      - HTTP-date: "Mon, 06 Oct 2025 01:18:51 GMT"
    """
    if not value:
        return 1
    v = value.strip()

    # Case 1: simple integer seconds
    if v.isdigit():
        try:
            return max(1, int(v))
        except Exception:
            return 1

    # Case 2: HTTP-date
    try:
        dt = parsedate_to_datetime(v)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        now = datetime.now(timezone.utc)
        seconds = int((dt - now).total_seconds())
        return max(1, seconds)
    except Exception:
        return 1
@retry(wait=wait_exponential(multiplier=1, min=1, max=10),
       stop=stop_after_attempt(5),
       retry=retry_if_exception_type((requests.RequestException, OpenAlexError)))
def _get(url: str, params: dict) -> dict:
    """GET with retries and basic error handling, including 403 backoffs."""
    merged = dict(params or {})
    merged.update(_polite_params())
    resp = requests.get(url, params=merged, timeout=30)
    # Handle throttling explicitly
    if resp.status_code == 429:
        retry_after = _parse_retry_after(resp.headers.get("Retry-After", "1"))
        retry_after = min(max(retry_after, 1), 300)  # 1..300 s
        import time as _t; _t.sleep(retry_after)
        raise OpenAlexError("Rate limited (429)")
    # Surface 400/403 bodies to logs to aid debugging
    if resp.status_code in (400, 403):
        try:
            body = resp.text[:500]
        except Exception:
            body = "<no-body>"
        raise OpenAlexError(f"{resp.status_code} error from OpenAlex: {body}")
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
    """Reverse OpenAlex abstract_inverted_index into plaintext."""
    if not inv:
        return ""
    pairs = []
    for word, positions in inv.items():
        for pos in positions:
            pairs.append((pos, word))
    if not pairs:
        return ""
    pairs.sort(key=lambda x: x[0])
    return " ".join([w for _, w in pairs])

def today_utc_date():
    return datetime.now(timezone.utc).date()

def dates_last_n_days(n: int):
    end = today_utc_date()
    start = end - timedelta(days=n)
    return start.isoformat(), end.isoformat()

def normalized_faculty_from_csv(csv_path: str):
    """
    Returns list of dicts:
      {'name': '...', 'openalex_id': 'A...', 'email': 'person@uni...', 'openalex_url': 'https://openalex.org/authors/a...'}
    Accepts flexible column headers: Name/Full Name; OpenAlexID/openalex_id/openalex/id; Email/e-mail/mail.
    Skips blank/NaN/non-matching IDs.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}

    name_col = next((cols[k] for k in ["name", "full name", "full_name"] if k in cols), None) or df.columns[0]
    id_col = next((cols[k] for k in ["openalexid", "openalex_id", "openalex", "id", "author_id"] if k in cols), None)
    if id_col is None and len(df.columns) >= 2:
        id_col = df.columns[1]
    email_col = next((cols[k] for k in ["email", "e-mail", "mail"] if k in cols), None)

    faculty = []
    for _, row in df.iterrows():
        name = str(row.get(name_col, "")).strip()
        aid = _normalize_author_id(row.get(id_col, ""))
        email = str(row.get(email_col, "")).strip() if email_col else ""
        if aid:
            faculty.append({
                "name": name,
                "openalex_id": aid,
                "email": email,
                "openalex_url": f"https://openalex.org/authors/{aid.lower()}"
            })
    return faculty

def _clean_ids(author_ids: T.Iterable[str]) -> T.List[str]:
    """Keep only A\d+; drop blanks/invalid."""
    out = []
    for aid in author_ids:
        if not aid:
            continue
        if re.match(r"^A\d+$", aid):
            out.append(aid)
    # de-dup while preserving order
    seen = set()
    uniq = []
    for a in out:
        if a not in seen:
            uniq.append(a)
            seen.add(a)
    return uniq

def collect_recent_works_for_authors(author_ids, start_date: str, end_date: str) -> dict:
    """
    Fetch works for list of OpenAlex Author IDs in date window.
    Returns dict keyed by work_id.
    """
    work_map = {}

    clean_ids = _clean_ids(author_ids)
    if not clean_ids:
        return work_map

    # Use smaller chunks and full URIs in the filter for robustness.
    CHUNK = 25
    for i in range(0, len(clean_ids), CHUNK):
        chunk = clean_ids[i:i+CHUNK]
        author_filter_val = "|".join(_author_uri(a) for a in chunk)
        filter_str = (
            f"author.id:{author_filter_val},"
            f"from_publication_date:{start_date},to_publication_date:{end_date}"
        )
        for w in page_works(filter_str, select=None):
            wid = w.get("id", "") or w.get("ids", {}).get("openalex", "")
            if not wid:
                continue
            if wid not in work_map:
                work_map[wid] = w

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
        openalex_id = _normalize_author_id(author.get("id", ""))
        authors.append({
            "id": author.get("id", ""),
            "openalex_id": openalex_id,
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
