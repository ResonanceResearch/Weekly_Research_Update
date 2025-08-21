
# OpenAlex Weekly Summary

This repository fetches papers authored by people in `data/full_time_faculty.csv` that were **published in the last 14 days**, extracts authors and journals, reconstructs plaintext abstracts (from `abstract_inverted_index`), and writes a JSON file to `data/`.  
On a **weekly schedule**, it then asks OpenAI to write a short highlight summary. The summary is published to **GitHub Pages** (`docs/index.html`) and **emailed** to the configured address.

## What you get

- `etl/fetch_recent_works.py`: Queries OpenAlex for each **OpenAlex Author ID** in your CSV and collects works in the last 14 days.
- `etl/generate_summary.py`: Summarizes the results via OpenAI and writes `docs/index.html` (plus `data/summary_latest.html`).
- `data/full_time_faculty.csv`: Your cohort roster (copied from the file you provided).
- GitHub Actions workflow: runs every week and can also be triggered manually.

## CSV format

Minimal example (`data/full_time_faculty.csv`):

```csv
Name,OpenAlexID
Jane Doe,A5084554132
```

The parser is flexible—headers like `OpenAlexID`, `openalex_id`, `openalex`, or `id` all work. IDs can be plain (`A5084554132`) or full URLs (`https://openalex.org/authors/a5084554132`).

## One-time setup (GitHub)

1. **Create a new GitHub repository** and push these files.
2. In **Settings → Secrets and variables → Actions → Secrets**, add:
   - `OPENAI_API_KEY` – your OpenAI API key.
   - (optional) `OPENAI_MODEL` – default `gpt-4o-mini`.
   - `OPENALEX_EMAIL` – set to your email (`jdebuck@ucalgary.ca`) to join the OpenAlex “polite pool”.
   - SMTP settings (for email sending—see below):
     - `MAIL_USERNAME` – full mailbox username (e.g. your Gmail address)
     - `MAIL_PASSWORD` – **App password** (Gmail requires an _App Password_ when 2FA is enabled)
3. In **Settings → Pages**:
   - Under **Build and deployment**, set **Source = GitHub Actions** (or **Deploy from a branch** with `/docs` folder).
4. Commit and push. You can run the workflow manually from the **Actions** tab (workflow: _weekly-openalex-summary_).

## Email delivery (SMTP via GitHub Actions)

This repo uses the community action [`dawidd6/action-send-mail`](https://github.com/dawidd6/action-send-mail) to email the summary.  
For Gmail:

- `MAIL_USERNAME` = `yourname@gmail.com`
- `MAIL_PASSWORD` = _App Password_ (from Google Account → Security → App passwords)
- Server: `smtp.gmail.com`, Port: `465`

If you prefer another provider, update the workflow accordingly.

## Local run (optional)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) Fetch data
python etl/fetch_recent_works.py --input data/full_time_faculty.csv --days 14 --out data/recent_works.json

# 2) Make summary + HTML
export OPENAI_API_KEY=sk-...
export OPENALEX_EMAIL=jdebuck@ucalgary.ca
python etl/generate_summary.py --in data/recent_works.json --out docs/index.html
```

## How it works

- Uses the OpenAlex `/works` endpoint with filters:
  - `author.id:<pipe-separated cohort>`, `from_publication_date:<start>`, `to_publication_date:<end>`
- Journal is taken from `primary_location.source.display_name` (or `best_oa_location.source.display_name`).
- Authors are in `authorships`; we match any that appear in your CSV.
- Abstracts are provided by OpenAlex as `abstract_inverted_index` and reconstructed to plaintext for summarization.

## Limitations

- `authorships` is limited to the first **100** authors per work in OpenAlex responses.
- Publication dates come from `publication_date`; see OpenAlex docs about accuracy nuance.
- If there are _many_ results, we cap table rows/summary input for token safety.

## Change the schedule

Edit `.github/workflows/weekly_openalex_summary.yml` – it runs weekly (Monday 14:00 UTC). You can also `workflow_dispatch` it on demand.
