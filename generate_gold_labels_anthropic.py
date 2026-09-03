"""
Generate 'gold' JSON labels for clinical trial eligibility text using Claude,
via forced tool-use so the output is schema-validated at the API level
rather than hoping the model returns clean JSON in prose.

Input:  clinical_trials_raw.jsonl (from fetch_clinical_trials.py)
Output: clinical_trials_labeled.jsonl, plus a flagged.jsonl subset worth
        spot-checking by hand before you trust it as training data.
"""

import json
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

MODEL = "claude-sonnet-4-5"  # use a strong model for labeling; you're fine-tuning Mistral, not this

SYSTEM_PROMPT = """You are a precise clinical trial data extractor. You extract \
structured information ONLY from the exact text provided -- never use outside \
medical knowledge to infer, complete, or correct anything. If a field is not \
explicitly stated in the text, its value must be null (for scalars) or an \
empty list (for arrays). Do not paraphrase language that is genuinely \
ambiguous; when criteria are unclear or contradictory, include them as-is in \
the relevant list rather than guessing intent."""

EXTRACTION_TOOL = {
    "name": "record_eligibility_extraction",
    "description": "Record the structured extraction of clinical trial eligibility criteria.",
    "input_schema": {
        "type": "object",
        "properties": {
            "min_age": {"type": ["string", "null"], "description": "e.g. '18 Years', or null if not stated"},
            "max_age": {"type": ["string", "null"], "description": "e.g. '75 Years', or null if no upper bound stated"},
            "sex": {"type": "string", "enum": ["ALL", "FEMALE", "MALE"]},
            "healthy_volunteers_accepted": {"type": ["boolean", "null"]},
            "inclusion_criteria": {"type": "array", "items": {"type": "string"}},
            "exclusion_criteria": {"type": "array", "items": {"type": "string"}},
            "condition_summary": {"type": "string", "description": "one sentence, grounded only in the text given"},
        },
        "required": ["sex", "inclusion_criteria", "exclusion_criteria", "condition_summary"],
    },
}


def extract_one(eligibility_text: str) -> dict:
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        tools=[EXTRACTION_TOOL],
        tool_choice={"type": "tool", "name": "record_eligibility_extraction"},
        messages=[{
            "role": "user",
            "content": (
                "Extract eligibility fields from this clinical trial text. "
                "Split criteria by which side of the inclusion/exclusion divide "
                "they fall on in the source, and preserve specific clinical "
                "detail (lab values, dosages, time windows) rather than "
                f"summarizing it away.\n\nEligibility text:\n\"\"\"\n{eligibility_text}\n\"\"\""
            ),
        }],
    )

    for block in response.content:
        if block.type == "tool_use":
            return block.input
    raise ValueError("No tool_use block in response -- inspect manually")


def looks_suspicious(extraction: dict, source_text: str) -> bool:
    """Cheap heuristics to flag outputs worth a human look, not a full check."""
    if not extraction.get("inclusion_criteria") and not extraction.get("exclusion_criteria"):
        return True  # extracted nothing from non-trivial text -- likely a miss
    total_criteria_chars = sum(
        len(c) for c in extraction.get("inclusion_criteria", []) + extraction.get("exclusion_criteria", [])
    )
    if total_criteria_chars > len(source_text) * 1.5:
        return True  # extraction is suspiciously longer than the source -- possible invention
    return False


def main():
    labeled, flagged = [], []

    with open("clinical_trials_raw.jsonl") as f:
        studies = [json.loads(line) for line in f]

    for i, study in enumerate(studies):
        print(f"[{i+1}/{len(studies)}] {study['nct_id']}")
        try:
            extraction = extract_one(study["eligibility_criteria"])
        except Exception as e:
            print(f"  skipped: {e}")
            continue

        record = {**study, "gold_extraction": extraction}
        labeled.append(record)
        if looks_suspicious(extraction, study["eligibility_criteria"]):
            flagged.append(record)

    with open("clinical_trials_labeled_anthropic.jsonl", "w") as f:
        for r in labeled:
            f.write(json.dumps(r) + "\n")

    with open("flagged_for_review_anthropic.jsonl", "w") as f:
        for r in flagged:
            f.write(json.dumps(r) + "\n")

    print(f"\nLabeled {len(labeled)} studies. {len(flagged)} flagged for manual review "
          f"({len(flagged) / max(len(labeled), 1):.1%}).")


if __name__ == "__main__":
    main()
