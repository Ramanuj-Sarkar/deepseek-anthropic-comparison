"""
Turn clinical_trials_labeled_deepseek.jsonl into a fine-tuning-ready dataset:
canonicalize the inclusion/exclusion criteria (join into single text blocks
so inconsistent list-splitting across documents stops being a comparison
problem), then split into train/eval by NCT ID so no document's text can
leak between the two sets.

Input:  clinical_trials_labeled_deepseek.jsonl
Output: train.jsonl, eval.jsonl
"""

import json
import random

random.seed(0)

EVAL_FRACTION = 0.15


def canonicalize(criteria_list) -> str:
    """Join a list of criterion strings into one normalized text block.
    Keeps 100% of the content; removes granularity as a variable."""
    if not criteria_list:
        return ""
    cleaned = [c.strip().rstrip(".") for c in criteria_list if c.strip()]
    return "\n".join(f"- {c}" for c in cleaned)


def to_training_example(record: dict) -> dict:
    extraction = record["gold_extraction"]
    return {
        "nct_id": record["nct_id"],
        "input": record["eligibility_criteria"],
        "output": {
            "min_age": extraction.get("min_age"),
            "max_age": extraction.get("max_age"),
            "sex": extraction.get("sex"),
            "healthy_volunteers_accepted": extraction.get("healthy_volunteers_accepted"),
            "inclusion_criteria_text": canonicalize(extraction.get("inclusion_criteria", [])),
            "exclusion_criteria_text": canonicalize(extraction.get("exclusion_criteria", [])),
        },
    }


def main():
    with open("clinical_trials_labeled_deepseek.jsonl") as f:
        records = [json.loads(line) for line in f]

    examples = [to_training_example(r) for r in records]

    # Split by NCT ID (each document appears in exactly one split -- no leakage).
    ids = [e["nct_id"] for e in examples]
    random.shuffle(ids)
    n_eval = int(len(ids) * EVAL_FRACTION)
    eval_ids = set(ids[:n_eval])

    train = [e for e in examples if e["nct_id"] not in eval_ids]
    eval_ = [e for e in examples if e["nct_id"] in eval_ids]

    with open("train.jsonl", "w") as f:
        for e in train:
            f.write(json.dumps(e) + "\n")
    with open("eval.jsonl", "w") as f:
        for e in eval_:
            f.write(json.dumps(e) + "\n")

    print(f"{len(examples)} total examples -> {len(train)} train / {len(eval_)} eval")
    print("Criteria are now single canonical text blocks per side, "
          "not lists -- see inclusion_criteria_text / exclusion_criteria_text.")


if __name__ == "__main__":
    main()
