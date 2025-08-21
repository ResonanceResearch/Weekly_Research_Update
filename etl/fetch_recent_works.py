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
return doi # fallback


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
main()
