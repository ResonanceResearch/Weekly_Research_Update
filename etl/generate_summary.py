"""
<td><details><summary>view</summary><div style="white-space:pre-wrap">{abstract_html}</div></details></td>
<td>{buttons_html}</td>
</tr>
""")


table_html = "\n".join(rows[:3000]) # safety cap


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
