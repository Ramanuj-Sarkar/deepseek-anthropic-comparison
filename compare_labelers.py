"""
Run both Claude and DeepSeek (deepseek-chat) extraction on the pilot sample
and render a side-by-side HTML report, hardest documents first, so you can
manually judge whether deepseek-chat holds up on the cases most likely to
break it before committing to it for the full labeling run.

Input:  pilot_sample.jsonl (from select_hard_examples.py)
Output: pilot_comparison.html
"""

import json
import html

# Reuse the extraction functions from the two labeling scripts.
from generate_gold_labels_anthropic import extract_one as extract_claude
from generate_gold_labels_deepseek import extract_one as extract_deepseek

ROW_TEMPLATE = """
<tr class="bucket-{bucket}">
  <td colspan="3"><strong>{nct_id}</strong> &mdash; bucket: {bucket}, difficulty: {score}
    <details><summary>source text</summary><pre>{source}</pre></details>
  </td>
</tr>
<tr>
  <td class="label">Claude</td>
  <td><pre>{claude_out}</pre></td>
  <td class="label">DeepSeek</td>
</tr>
<tr>
  <td colspan="3"><pre>{deepseek_out}</pre></td>
</tr>
"""

PAGE_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Labeler comparison</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #f7f7f8; }}
table {{ width: 100%; border-collapse: collapse; margin-bottom: 2rem; background: white; }}
td {{ border: 1px solid #ddd; padding: 0.5rem; vertical-align: top; font-size: 0.85rem; }}
.label {{ font-weight: bold; width: 6rem; background: #eee; }}
.bucket-hard {{ background: #fff0f0; }}
.bucket-easy {{ background: #f0fff4; }}
pre {{ white-space: pre-wrap; word-break: break-word; margin: 0; }}
</style></head><body>
<h1>Claude vs. DeepSeek eligibility extraction — pilot comparison</h1>
<p>Sorted hardest first. Look for: dropped criteria, invented criteria not in the source,
wrong inclusion/exclusion assignment, and mishandled negation/exception language.</p>
<table>{rows}</table>
</body></html>
"""


def main():
    with open("pilot_sample.jsonl") as f:
        pilot = [json.loads(line) for line in f]

    # hardest first -- that's where disagreements are most likely and most informative
    pilot.sort(key=lambda s: s["difficulty"]["difficulty_score"], reverse=True)

    rows = []
    for i, study in enumerate(pilot):
        print(f"[{i+1}/{len(pilot)}] {study['nct_id']} ({study['difficulty_bucket']})")
        text = study["eligibility_criteria"]

        try:
            claude_result = extract_claude(text)
        except Exception as e:
            claude_result = {"error": str(e)}

        deepseek_result = extract_deepseek(text) or {"error": "failed after retry"}

        rows.append(ROW_TEMPLATE.format(
            nct_id=study["nct_id"],
            bucket=study["difficulty_bucket"],
            score=study["difficulty"]["difficulty_score"],
            source=html.escape(text[:1500] + ("..." if len(text) > 1500 else "")),
            claude_out=html.escape(json.dumps(claude_result, indent=2)),
            deepseek_out=html.escape(json.dumps(deepseek_result, indent=2)),
        ))

    with open("pilot_comparison.html", "w") as f:
        f.write(PAGE_TEMPLATE.format(rows="".join(rows)))

    print("\nWrote pilot_comparison.html -- open it and read the 'hard' bucket rows first.")


if __name__ == "__main__":
    main()
