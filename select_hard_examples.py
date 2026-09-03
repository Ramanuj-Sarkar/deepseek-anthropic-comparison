"""
Score eligibility texts for extraction difficulty and pull a stratified pilot
sample -- hardest, easiest, and a random middle band -- rather than a plain
random sample, so a Claude-vs-DeepSeek comparison actually tests where the
two models are likely to diverge.

Input:  clinical_trials_raw.jsonl (from fetch_clinical_trials.py)
Output: pilot_sample.jsonl (50 documents with difficulty scores attached)
"""

import json
import re
import random

random.seed(0)

# Proxies for "hard to extract from," not proxies for "medically complex" --
# these are properties of the TEXT, not the trial itself.
NEGATION_EXCEPTION_TERMS = [
    "unless", "except", "excluding", "other than", "not eligible",
    "not be eligible", "provided that", "with the exception of",
]
CONDITIONAL_TERMS = [
    "if the", "in the event", "at the discretion", "as determined by",
    "may be considered", "on a case-by-case",
]
NUMERIC_LAB_PATTERN = re.compile(
    r"\b\d+(\.\d+)?\s*(mg|mL|mmol|mcg|ng|%|kg|IU|units?|mmHg|x10)\b", re.IGNORECASE
)
BULLET_PATTERN = re.compile(r"(^|\n)\s*[-*•]|\n\s*\d+[.)]")


def score_difficulty(text: str) -> dict:
    length = len(text)
    bullets = len(BULLET_PATTERN.findall(text))
    # low bullet density on a long document = unstructured wall of text = harder to parse
    bullet_density = bullets / max(length / 500, 1)

    negation_count = sum(text.lower().count(term) for term in NEGATION_EXCEPTION_TERMS)
    conditional_count = sum(text.lower().count(term) for term in CONDITIONAL_TERMS)
    numeric_count = len(NUMERIC_LAB_PATTERN.findall(text))

    # weighted score: unstructured + long + full of exceptions/conditionals/numbers = hard
    score = (
        (length / 1000) * 1.0
        + max(0, 2 - bullet_density) * 3.0   # penalize low bullet density most heavily
        + negation_count * 2.0
        + conditional_count * 2.5
        + numeric_count * 1.5
    )

    return {
        "difficulty_score": round(score, 2),
        "length": length,
        "bullet_density": round(bullet_density, 2),
        "negation_count": negation_count,
        "conditional_count": conditional_count,
        "numeric_count": numeric_count,
    }


def main():
    with open("clinical_trials_raw.jsonl") as f:
        studies = [json.loads(line) for line in f]

    for study in studies:
        study["difficulty"] = score_difficulty(study["eligibility_criteria"])

    studies.sort(key=lambda s: s["difficulty"]["difficulty_score"])

    n = len(studies)
    easiest = studies[: n // 10][:15]                 # bottom decile
    hardest = studies[-(n // 10):][-20:]               # top decile, weighted slightly heavier
    middle_pool = studies[n // 10 : -(n // 10)]
    middle = random.sample(middle_pool, min(15, len(middle_pool)))

    pilot = easiest + middle + hardest
    for s in pilot:
        s["difficulty_bucket"] = (
            "easy" if s in easiest else "hard" if s in hardest else "middle"
        )

    with open("pilot_sample.jsonl", "w") as f:
        for s in pilot:
            f.write(json.dumps(s) + "\n")

    scores = [s["difficulty"]["difficulty_score"] for s in studies]
    print(f"Corpus: {n} documents, difficulty score range {min(scores):.1f}-{max(scores):.1f}")
    print(f"Pilot sample: {len(easiest)} easy, {len(middle)} middle, {len(hardest)} hard "
          f"-> pilot_sample.jsonl ({len(pilot)} total)")
    print("\nHardest document preview:")
    print(f"  {hardest[-1]['nct_id']}: score={hardest[-1]['difficulty']['difficulty_score']}, "
          f"{hardest[-1]['difficulty']}")


if __name__ == "__main__":
    main()
