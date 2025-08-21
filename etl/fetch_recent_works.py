# `etl/openalex_utils.py` (modified)

```python
import os
import re
import time
import json
import typing as T
from urllib.parse import urlencode
from datetime import datetime, timedelta, timezone

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
    polite = {}
    email = os.environ.get("OPENALEX_EMAIL")
    if email:
        polite["mailto"] = email  # OpenAlex 'polite pool'
    return polite

class OpenAlexError(Exception):
    pass

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
        retry_after = int(resp.headers.get("Retry-After", "1"))
        time.sleep(max(retry_after, 1))
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
```

---

# `etl/fetch_recent_works.py` (modified)

```python
#!/usr/bin/env python3
"""
Fetch recent works for cohort authors, store JSON in data/.
"""
import os, json, argparse, re
from datetime import datetime, timezone, timedelta
from pathlib import Path

from openalex_utils import (normalized_faculty_from_csv, dates_last_n_days, collect_recent_works_for_authors,
                            extract_journal_name, extract_authors_list, match_cohort_authors, reconstruct_abstract)

def first_n_words(text: str, n: int = 250) -> str:
    if not text:
        return ""
    words = re.findall(r"\S+", text.strip())
    if len(words) <= n:
        return " ".join(words)
    return " ".join(words[:n]) + "…"

def normalize_doi_url(doi: str | None) -> str:
    if not doi:
        return ""
    doi = doi.strip()
    if doi.startswith("http://") or doi.startswith("https://"):
        return doi
    if doi.startswith("10."):
        return "https://doi.org/" + doi
    return doi  # fallback

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/full_time_faculty.csv", help="CSV with Name, OpenAlexID, Email columns")
    ap.add_argument("--days", default=7, type=int, help="Lookback window in days")
    ap.add_argument("--out", default="data/recent_works.json", help="Output JSON")
    args = ap.parse_args()

    start_date, end_date = dates_last_n_days(args.days)
    cohort = normalized_faculty_from_csv(args.input)
    cohort_ids = {c["openalex_id"] for c in cohort}
    id_to_email = {c["openalex_id"]: (c.get("email") or "").strip() for c in cohort}

    works_map = collect_recent_works_for_authors(sorted(list(cohort_ids)), start_date, end_date)

    # Build final list
    results = []
    for wid, w in works_map.items():
        authors = extract_authors_list(w)
        matches = match_cohort_authors(authors, cohort_ids)
        # annotate matches with email if present
        for m in matches:
            m["email"] = id_to_email.get(m.get("openalex_id", ""), "")

        abstract_inv = w.get("abstract_inverted_index")
        abstract_text = reconstruct_abstract(abstract_inv) if abstract_inv else ""
        abstract_snippet_250w = first_n_words(abstract_text, 250)

        results.append({
            "openalex_id": w.get("id", ""),
            "title": w.get("display_name") or w.get("title"),
            "publication_date": w.get("publication_date"),
            "publication_year": w.get("publication_year"),
            "journal": extract_journal_name(w),
            "type": w.get("type"),
            "doi": w.get("doi"),
            "doi_url": normalize_doi_url(w.get("doi")),
            "authorships": authors,
            "cohort_matches": matches,  # subset of authors found in input CSV, with email
            "abstract_inverted_index": abstract_inv,
            "abstract_text": abstract_text,                 # FULL abstract stays here
            "abstract_snippet_250w": abstract_snippet_250w, # GPT-only snippet
            "is_oa": (w.get("open_access") or {}).get("is_oa", None),
        })

    # Sort by date desc, title
    results.sort(key=lambda x: (x.get("publication_date") or "", x.get("title") or ""), reverse=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window": {"start": start_date, "end": end_date},
        "cohort_size": len(cohort_ids),
        "works_count": len(results),
        "works": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Also write a dated snapshot for traceability
    dated = Path("data") / f"recent_works_{datetime.now(timezone.utc).date().isoformat()}.json"
    with open(dated, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Wrote {args.out} with {len(results)} works between {start_date} and {end_date}.")

if __name__ == "__main__":
    main()
```

---

# `etl/generate_summary.py` (modified)

```python
#!/usr/bin/env python3
"""
Generate an HTML summary via OpenAI and write docs/index.html

- Reads data JSON produced by fetch_recent_works.py
- Calls OpenAI (if key present) to produce a concise summary
- Builds a static HTML page with the summary and a table of works
- Adds "Congratulate" mailto buttons per matched cohort author (Email column in CSV)
"""
import os
import re
import json
import argparse
import html
import urllib.parse as _url
from datetime import datetime, timezone
from pathlib import Path

def first_n_words(text: str, n: int = 250) -> str:
    if not text:
        return ""
    words = re.findall(r"\S+", text.strip())
    if len(words) <= n:
        return " ".join(words)
    return " ".join(words[:n]) + "…"

def render_fallback_summary(data: dict) -> str:
    """Produce a simple HTML summary without OpenAI if key is missing or call fails."""
    works = data.get("works", [])
    window = data.get("window", {})
    by_journal = {}
    people = set()
    for w in works:
        j = w.get("journal") or "Unknown venue"
        by_journal[j] = by_journal.get(j, 0) + 1
        for a in (w.get("cohort_matches") or []):
            dn = a.get("display_name")
            if dn:
                people.add(dn)

    journals_html = "".join(
        f"<li>{html.escape(j)} — {n}</li>"
        for j, n in sorted(by_journal.items(), key=lambda kv: kv[1], reverse=True)
    )
    people_html = ", ".join(sorted(people)) if people else "—"

    return f"""
    <section>
      <h2>New works from {html.escape(str(window.get('start')))} to {html.escape(str(window.get('end')))}</h2>
      <p>Total works: {len(works)}</p>
      <h3>Journals (counts)</h3>
      <ul>{journals_html}</ul>
      <h3>People mentioned</h3>
      <p>{html.escape(people_html)}</p>
    </section>
    """

def _get_model() -> str:
    """
    Choose model. Falls back to 'gpt-5' if OPENAI_MODEL is unset/blank.
    """
    m = (os.environ.get("OPENAI_MODEL") or "").strip()
    return m or "gpt-5"

def make_messages(data: dict) -> list:
    """
    Build a messages list for Chat Completions / Responses API.
    Provide compact lines to control token use.
    Format per line: TITLE ¦ JOURNAL ¦ DATE ¦ COHORT_AUTHORS ¦ ABSTRACT_SNIPPET
    """
    window = data.get("window", {})
    works = data.get("works", [])
    lines = []
    for w in works:
        names = "; ".join(
            [a.get("display_name", "") for a in (w.get("cohort_matches") or []) if a.get("display_name")]
        )
        # Use the 250-word snippet if present; compute on the fly otherwise
        abstract = w.get("abstract_snippet_250w") or first_n_words((w.get("abstract_text") or ""), 250)
        # Replace '|' with '¦' to avoid delimiter collisions
        line = f"{w.get('title','')} | {w.get('journal','')} | {w.get('publication_date','')} | {names} | {abstract}"
        line = line.replace("|", "¦")
        lines.append(line)

    joined = "\n".join(lines[:400])  # cap to avoid token bloat

    system = (
        "You are a research analyst summarizing new scholarly works. "
        "Be precise, concise, and explicitly name journals and in-cohort authors."
    )
    user = f"""
Summarize new works published between {window.get('start')} and {window.get('end')}.
Goals:
- Highlight notable findings (group by theme if possible).
- Explicitly mention which cohort authors appear (by full name).
- Name the journals/venues.
- Keep to ~3 short bullet points per individual work plus a 'By the numbers' section (counts by journal, count of works).
- Don't suggest follow up queries.

Data (one per line: TITLE ¦ JOURNAL ¦ DATE ¦ COHORT_AUTHORS ¦ ABSTRACT_SNIPPET):
{joined}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def call_openai_and_summarize(data: dict) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return render_fallback_summary(data)

    # Prefer Chat Completions, fallback to Responses if needed
    model = _get_model()
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        messages = make_messages(data)
        rsp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )
        content = rsp.choices[0].message.content
    except Exception as e_chat:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            messages = make_messages(data)
            rsp = client.responses.create(
                model=model,
                input=json.dumps(messages),
            )
            content = getattr(rsp, "output_text", None) or str(rsp)
        except Exception as e_resp:
            # Return a helpful fallback with error details
            safe_err = html.escape(f"OpenAI summarization failed (model={model}): chat_error={e_chat}; resp_error={e_resp}")
            content = f"<p><em>{safe_err}</em></p>"

    html_block = f"""
    <section>
      <h2>Automated weekly summary</h2>
      <div style="white-space:pre-wrap;line-height:1.4">{content}</div>
    </section>
    """
    return html_block

# --- Mailto helpers ---

def _mailto_link(email: str, subject: str, body: str) -> str:
    """Build a safe mailto URL with subject and body."""
    if not email:
        return ""
    q = {
        "subject": subject,
        "body": body
    }
    return f"mailto:{_url.quote(email)}?{_url.urlencode(q)}"

def build_html(data: dict, summary_html: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    rows = []
    for w in data.get("works", []):
        cohort_matches = (w.get("cohort_matches") or [])
        openalex_link = w.get("openalex_id") or ""
        doi_url = w.get("doi_url") or ""
        link = doi_url or openalex_link or ""
        title = (w.get("title") or "").strip()
        journal = (w.get("journal") or "").strip()
        date = (w.get("publication_date") or "").strip()

        # FULL abstract for HTML (no prompt cap)
        abstract_full = (w.get("abstract_text") or "").strip()
        abstract_html = html.escape(abstract_full or "—")

        # Cohort authors (names)
        cohort_names = ", ".join([a.get("display_name", "") for a in cohort_matches if a.get("display_name")])
        cohort_names = cohort_names or "—"

        # Build congratulate buttons (one per matched author with an email)
        buttons = []
        for a in cohort_matches:
            email = (a.get("email") or "").strip()
            if not email:
                continue
            person = a.get("display_name") or "colleague"
            subj = f"Congrats on “{title}” in {journal}" if title and journal else f"Congratulations on your new paper"
            body_lines = [
                f"Hi {person},",
                "",
                f"Congrats on your new paper{f' "{title}"' if title else ''}{f' in {journal}' if journal else ''}{f' ({date})' if date else ''}.",
                f"Link: {link}" if link else "",
                "",
                "—"
            ]
            body = "\n".join([ln for ln in body_lines if ln is not None])
            mailto = _mailto_link(email, subj, body)
            safe_label = html.escape(f"Email {person.split()[0] if person else 'author'}")
            buttons.append(f'<a class="btn" href="{mailto}">{safe_label}</a>')

        buttons_html = " ".join(buttons) if buttons else "—"

        # Escape text for table cells
        title_html = html.escape(title)
        journal_html = html.escape(journal)
        date_html = html.escape(date)
        cohort_names_html = html.escape(cohort_names)
        link_attr = html.escape(link)
        openalex_attr = html.escape(openalex_link)

        rows.append(f"""
        <tr>
          <td><a href="{openalex_attr}">{title_html}</a>{f' <small>· <a href="{link_attr}">doi/link</a></small>' if link else ''}</td>
          <td>{journal_html}</td>
          <td>{date_html}</td>
          <td>{cohort_names_html}</td>
          <td><details><summary>view</summary><div style="white-space:pre-wrap">{abstract_html}</div></details></td>
          <td>{buttons_html}</td>
        </tr>
        """)

    table_html = "\n".join(rows[:3000])  # safety cap

    template = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>UCVM Research Weekly Summary</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 2rem; }}
    header, footer {{ color: #444; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 1.5rem; }}
    th, td {{ border: 1px solid #ddd; padding: 0.5rem; vertical-align: top; }}
    th {{ background: #f7f7f7; text-align: left; }}
    .meta {{ font-size: 0.9rem; color: #666; }}
    details summary {{ cursor: pointer; }}
    .btn {{
      display: inline-block;
      padding: 0.35rem 0.6rem;
      border-radius: 0.45rem;
      border: 1px solid #ccc;
      text-decoration: none;
      font-size: 0.9rem;
      background: #fafafa;
    }}
    .btn:hover {{ background: #f0f0f0; }}
  </style>
</head>
<body>
  <header>
    <h1>UCVM Research Weekly Summary</h1>
    <p class="meta">Generated: {now} | Window: {html.escape(str(data.get('window',{}).get('start')))} → {html.escape(str(data.get('window',{}).get('end')))} | Works: {data.get('works_count')}</p>
  </header>

  {summary_html}

  <h2>All works in window</h2>
  <table>
    <thead><tr><th>Title</th><th>Journal</th><th>Date</th><th>Cohort Authors</th><th>Abstract</th><th>Congratulate</th></tr></thead>
    <tbody>
      {table_html}
    </tbody>
  </table>

  <footer><p class="meta">Built by GitHub Actions • Data from OpenAlex</p></footer>
</body>
</html>"""
    return template

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default="data/recent_works.json")
    ap.add_argument("--out", dest="outfile", default="docs/index.html")
    args = ap.parse_args()

    with open(args.infile, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary_html = call_openai_and_summarize(data)
    html_page = build_html(data, summary_html)

    Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)
    with open(args.outfile, "w", encoding="utf-8") as f:
        f.write(html_page)

    # Also save the summary page as an attachment for email
    with open("data/summary_latest.html", "w", encoding="utf-8") as f:
        f.write(html_page)

    print(f"Wrote {args.outfile} and data/summary_latest.html")

if __name__ == "__main__":
    main()
```
