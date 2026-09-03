# Description

This project involves a comparison between
Anthropic Claude Sonnet 4.5 and
Deepseek Chat 3.2.

I am comparing them based on cost and accuracy
to gain a more holistic understanding of the tradeoffs,
if any exist, between the two for this use case
(extracting information from messy files into structured JSON).

# Steps

1. Run `fetch_clinical_trials.py` to get the basic data (`clinical_trials_raw.jsonl`).
2. Run `select_hard_examples.py` to get the hard examples (`pilot_sample.jsonl`).
3. Run `compare_labelers.py` to test the two head-to-head (`pilot_comparison.html`).
4. Make a decision on which one to choose:
   1. If Claude, then choose `generate_gold_labels_claude.py`
      (`clinical_trials_labeled_anthropic.jsonl` and `flagged_for_review_anthropic.jsonl`).
   2. If Deepseek, then choose `generate_gold_labels_deepseek.py`.
      (`clinical_trials_labeled_deepseek.jsonl` and `flagged_for_review_deepseek.jsonl`).

# Results

I got better and cheaper results from Deepseek,
so I ended up labeling these files using Deepseek.
I then used these files in another project.
