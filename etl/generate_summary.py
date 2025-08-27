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
        "You are a scientific communications expert summarizing new scholarly works. "
        "Be precise, concise, compelling and explicitly name journals and in-cohort authors."
    )
    user = f"""
Summarize new works published between {window.get('start')} and {window.get('end')}.
Goals:
- Highlight notable findings.
- Explicitly mention which cohort authors appear (by full name).
- Name the journals/venues.
- Keep to max 3 short points (not bullet points) per individual work plus a 'By the numbers' section (counts by journal, count of works).
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
    """
    Build a mailto: link using RFC 6068 encoding (percent-encoding).
    Spaces -> %20, newlines -> %0A. Avoid application/x-www-form-urlencoded (+).
    """
    if not email:
        return ""
    # Encode subject and body; keep email readable (don't encode '@', '+', '.', '-', '_')
    email_safe = _url.quote(email, safe='@+._-')
    parts = []
    if subject:
        parts.append("subject=" + _url.quote(subject, safe=""))
    if body:
        # Normalize newlines
        body = body.replace("\r\n", "\n").replace("\r", "\n")
        parts.append("body=" + _url.quote(body, safe=""))
    query = ("?" + "&".join(parts)) if parts else ""
    return f"mailto:{email_safe}{query}"

def build_html(data: dict, summary_html: str) -> str:
    window = data.get("window", {}) or {}
    start = (window.get("start") or "").strip()
    end = (window.get("end") or "").strip()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    rows = []
    for w in data.get("works", []):
        cohort_matches = (w.get("cohort_matches") or [])
        openalex_link = (w.get("openalex_id") or "").strip()
        doi_url = (w.get("doi_url") or "").strip()
        link = doi_url or openalex_link or ""

        title = (w.get("title") or "").strip()
        journal = (w.get("journal") or "").strip()
        date = (w.get("publication_date") or "").strip()
        abstract_full = (w.get("abstract_text") or "").strip()

        # Escape text for HTML
        title_html = html.escape(title) or "—"
        journal_html = html.escape(journal) or "—"
        date_html = html.escape(date) or "—"
        abstract_html = html.escape(abstract_full or "—")
        openalex_attr = html.escape(openalex_link)
        link_attr = html.escape(link)

        # Cohort author names (displayed even if no email)
        cohort_names = ", ".join([a.get("display_name", "") for a in cohort_matches if a.get("display_name")]) or "—"
        cohort_names_html = html.escape(cohort_names)

        # Build congratulate buttons (only for authors with email)
        buttons = []
        for a in cohort_matches:
            email = (a.get("email") or "").strip()
            if not email:
                continue
            person = (a.get("display_name") or "colleague").strip()
            first = person.split()[0] if person else "author"

            subj = f"Congrats on your recent paper in {journal}" if journal else "Congratulations on your new paper"
            phrase = "Congrats on your new paper"
            if title:
                phrase += f' "{title}"'
            if journal:
                phrase += f" in {journal}"
            if date:
                phrase += f" ({date})"
            phrase += "."

            body_lines = [
                f"Hi {person},",
                "",
                phrase,
                f"Link: {link}" if link else "",
                "",
                "—",
            ]
            body = "\n".join([ln for ln in body_lines if ln is not None])
            mailto = _mailto_link(email, subj, body)
            buttons.append(f'<a class="btn" href="{mailto}">Email {html.escape(first)}</a>')

        buttons_html = " ".join(buttons) if buttons else "—"

        # Title cell with link to OpenAlex (and DOI/link if available)
        title_cell = f'<a href="{openalex_attr}">{title_html}</a>'
        if link:
            title_cell += f' <small>· <a href="{link_attr}">doi/link</a></small>'

        rows.append(f"""
        <tr>
          <td data-label="Title">{title_cell}</td>
          <td data-label="Journal">{journal_html}</td>
          <td data-label="Date">{date_html}</td>
          <td data-label="Cohort Authors">{cohort_names_html}</td>
          <td data-label="Abstract"><details><summary>view</summary><div style="white-space:pre-wrap">{abstract_html}</div></details></td>
          <td data-label="Congratulate">{buttons_html}</td>
        </tr>
        """)

    table_html = (
        '\n'.join(rows[:3000])
        if rows else '<tr><td colspan="6">No works found in this window.</td></tr>'
    )

    # Build the page with REAL meta, REAL summary_html, and REAL table_html
    n_works = len(rows)
    template = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>UCVM Research Weekly Summary</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <div class="page">
    <header class="header">
      <div class="header__brand">
        <h1>UCVM Research Weekly Summary</h1>
        <p class="meta">Generated: {html.escape(now)} | Window: {html.escape(start)} → {html.escape(end)} | Works: {n_works}</p>
      </div>
      <img class="header__logo" src="logo.png" alt="Logo">
    </header>

    {summary_html}

    <section class="section">
      <h2>All works in window</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Title</th><th>Journal</th><th>Date</th>
              <th>Cohort Authors</th><th>Abstract</th><th>Congratulate</th>
            </tr>
          </thead>
          <tbody>
            {table_html}
          </tbody>
        </table>
      </div>
    </section>

    <footer class="footer">
      Built by GitHub Actions • Data from OpenAlex
    </footer>
  </div>
</body>
</html>
"""
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
