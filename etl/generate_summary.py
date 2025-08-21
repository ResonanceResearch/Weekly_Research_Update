
#!/usr/bin/env python3
"""
Generate an HTML summary via OpenAI and write docs/index.html
"""
import os, json, argparse, sys, textwrap
from datetime import datetime, timezone
from pathlib import Path

def render_fallback_summary(data: dict) -> str:
    """Produce a simple HTML summary without OpenAI if key is missing."""
    works = data.get("works", [])
    window = data.get("window", {})
    by_journal = {}
    people = set()
    for w in works:
        j = w.get("journal") or "Unknown venue"
        by_journal.setdefault(j, 0)
        by_journal[j] += 1
        for a in (w.get("cohort_matches") or []):
            people.add(a.get("display_name"))
    journals_html = "".join(f"<li>{j} — {n}</li>" for j, n in sorted(by_journal.items(), key=lambda kv: kv[1], reverse=True))
    people_html = ", ".join(sorted([p for p in people if p]))
    return f"""
    <section>
      <h2>New works from {window.get('start')} to {window.get('end')}</h2>
      <p>Total works: {len(works)}</p>
      <h3>Journals (counts)</h3>
      <ul>{journals_html}</ul>
      <h3>People mentioned</h3>
      <p>{people_html or '—'}</p>
    </section>
    """

def make_prompt(data: dict) -> list:
    """
    Build a messages list for Chat Completions API.
    We provide a compact CSV-like list of works to control token size.
    """
    window = data.get("window", {})
    works = data.get("works", [])
    # compact lines: TITLE | JOURNAL | DATE | COHORT_AUTHORS | ABSTRACT (first 600 chars)
    lines = []
    for w in works:
        names = "; ".join([a.get("display_name","") for a in (w.get("cohort_matches") or []) if a.get("display_name")])
        abstract = (w.get("abstract_text") or "")
        if len(abstract) > 600:
            abstract = abstract[:600] + "…"
        line = f"{w.get('title','')} | {w.get('journal','')} | {w.get('publication_date','')} | {names} | {abstract}"
        # strip pipes inside text
        line = line.replace("|", "¦")
        lines.append(line)
    joined = "\n".join(lines[:400])  # hard cap to avoid token bloat

    system = "You are a research analyst summarizing new scholarly works. Be precise, concise, and name journals and in-cohort authors explicitly."
    user = f"""
Summarize new works published between {window.get('start')} and {window.get('end')}. 
Goals:
- Highlight notable findings (group by theme if possible).
- Explicitly mention which cohort authors appear (by full name).
- Name the journals/venues.
- Keep to ~6 short bullet points plus a 'By the numbers' section (counts by journal, count of works).
Data (one per line: TITLE ¦ JOURNAL ¦ DATE ¦ COHORT_AUTHORS ¦ ABSTRACT_SNIPPET):
{joined}
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

def call_openai_and_summarize(data: dict) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return render_fallback_summary(data)

    # Import here so requirements can be optional during basic tests
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        messages = make_prompt(data)
        # Prefer a small, cost-effective model; user can change via env
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        rsp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )
        content = rsp.choices[0].message.content
    except Exception as e:
        # Try responses API fallback if chat.completions failed
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            prompt = make_prompt(data)
            rsp = client.responses.create(model=os.environ.get("OPENAI_MODEL","gpt-4o-mini"),
                                          input=json.dumps(prompt))
            content = getattr(rsp, "output_text", None) or str(rsp)
        except Exception as e2:
            content = f"<p><em>OpenAI summarization failed: {e2}</em></p>"

    # Wrap content as HTML
    html = f"""
    <section>
      <h2>Automated weekly summary</h2>
      <div style="white-space:pre-wrap;line-height:1.4">{content}</div>
    </section>
    """
    return html

def build_html(data: dict, summary_html: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    # Build a simple index including a table of works
    rows = []
    for w in data.get("works", []):
        cohort_names = ", ".join([a.get("display_name","") for a in (w.get("cohort_matches") or []) if a.get("display_name")])
        rows.append(f"""
        <tr>
          <td><a href="{w.get('openalex_id','')}">{w.get('title','')}</a></td>
          <td>{w.get('journal','')}</td>
          <td>{w.get('publication_date','')}</td>
          <td>{cohort_names or '—'}</td>
        </tr>
        """)
    table_html = "\n".join(rows[:3000])  # sanity cap

    template = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>OpenAlex Weekly Summary</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 2rem; }}
    header, footer {{ color: #444; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 1.5rem; }}
    th, td {{ border: 1px solid #ddd; padding: 0.5rem; vertical-align: top; }}
    th {{ background: #f7f7f7; text-align: left; }}
    .meta {{ font-size: 0.9rem; color: #666; }}
  </style>
</head>
<body>
  <header>
    <h1>OpenAlex Weekly Summary</h1>
    <p class="meta">Generated: {now} | Window: {data.get('window',{}).get('start')} → {data.get('window',{}).get('end')} | Works: {data.get('works_count')}</p>
  </header>

  {summary_html}

  <h2>All works in window</h2>
  <table>
    <thead><tr><th>Title</th><th>Journal</th><th>Date</th><th>Cohort Authors</th></tr></thead>
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
    html = build_html(data, summary_html)

    Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)
    with open(args.outfile, "w", encoding="utf-8") as f:
        f.write(html)

    # Also save the summary as standalone for email
    with open("data/summary_latest.html", "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Wrote {args.outfile} and data/summary_latest.html")

if __name__ == "__main__":
    main()
