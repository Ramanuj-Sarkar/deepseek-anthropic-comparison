"""
Generate 'gold' JSON labels for clinical trial eligibility text using DeepSeek
instead of Claude, via the OpenAI-compatible client and function calling.

DeepSeek doesn't document a hard tool_choice guarantee the way Anthropic does,
so this version validates the returned JSON against the schema and retries
once before falling back to a flagged skip -- treat this as a real difference
from the Claude version, not a cosmetic one, and mention it in your README.

Input:  clinical_trials_raw.jsonl (from fetch_clinical_trials.py)
Output: clinical_trials_labeled_deepseek.jsonl, flagged_for_review_deepseek.jsonl
"""

import json, os
from openai import OpenAI
from jsonschema import validate, ValidationError
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],  # or read from env: os.environ["DEEPSEEK_API_KEY"]
    base_url="https://api.deepseek.com",
)

MODEL = "deepseek-chat"  # NOT deepseek-v4-pro/-flash: V4's default "thinking" mode rejects a
                         # forced tool_choice targeting a specific function ("Thinking mode
                         # does not support this tool_choice"), and there's no documented way
                         # to disable thinking mode on V4 yet. deepseek-chat (V3.2) supports
                         # tool_choice targeting a specific function normally.

SYSTEM_PROMPT = """You are a precise clinical trial data extractor. You extract \
structured information ONLY from the exact text provided -- never use outside \
medical knowledge to infer, complete, or correct anything. If a field is not \
explicitly stated in the text, its value must be null (for scalars) or an \
empty list (for arrays). Do not paraphrase language that is genuinely \
ambiguous; when criteria are unclear or contradictory, include them as-is in \
the relevant list rather than guessing intent. Always respond by calling the \
record_eligibility_extraction function -- never respond in plain text."""

SCHEMA = {
    "type": "object",
    "properties": {
        "min_age": {"type": ["string", "null"]},
        "max_age": {"type": ["string", "null"]},
        "sex": {"type": "string", "enum": ["ALL", "FEMALE", "MALE"]},
        "healthy_volunteers_accepted": {"type": ["boolean", "null"]},
        "inclusion_criteria": {"type": "array", "items": {"type": "string"}},
        "exclusion_criteria": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["sex", "inclusion_criteria", "exclusion_criteria"],
}

TOOLS = [{
    "type": "function",
    "function": {
        "name": "record_eligibility_extraction",
        "description": "Record the structured extraction of clinical trial eligibility criteria.",
        "parameters": SCHEMA,
    },
}]


def extract_one(eligibility_text: str, attempt: int = 1) -> dict | None:
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                "Extract eligibility fields from this clinical trial text. "
                "Split criteria by which side of the inclusion/exclusion divide "
                "they fall on in the source, and preserve specific clinical "
                "detail (lab values, dosages, time windows) rather than "
                f"summarizing it away.\n\nEligibility text:\n\"\"\"\n{eligibility_text}\n\"\"\""
            )},
        ],
        tools=TOOLS,
        tool_choice={"type": "function", "function": {"name": "record_eligibility_extraction"}},
    )

    message = response.choices[0].message
    if not message.tool_calls:
        if attempt < 2:
            return extract_one(eligibility_text, attempt=attempt + 1)  # retry once
        return None

    try:
        extraction = json.loads(message.tool_calls[0].function.arguments)
        validate(instance=extraction, schema=SCHEMA)
        return extraction
    except (json.JSONDecodeError, ValidationError):
        if attempt < 2:
            return extract_one(eligibility_text, attempt=attempt + 1)
        return None


def looks_suspicious(extraction: dict, source_text: str) -> bool:
    if not extraction.get("inclusion_criteria") and not extraction.get("exclusion_criteria"):
        return True
    total_chars = sum(
        len(c) for c in extraction.get("inclusion_criteria", []) + extraction.get("exclusion_criteria", [])
    )
    return total_chars > len(source_text) * 1.5


def main():
    labeled, flagged, failed = [], [], 0

    with open("clinical_trials_raw.jsonl") as f:
        studies = [json.loads(line) for line in f]

    for i, study in enumerate(studies):
        print(f"[{i+1}/{len(studies)}] {study['nct_id']}")
        extraction = extract_one(study["eligibility_criteria"])

        if extraction is None:
            failed += 1
            print("  failed after retry -- skipped")
            continue

        record = {**study, "gold_extraction": extraction}
        labeled.append(record)
        if looks_suspicious(extraction, study["eligibility_criteria"]):
            flagged.append(record)

    with open("clinical_trials_labeled_deepseek.jsonl", "w") as f:
        for r in labeled:
            f.write(json.dumps(r) + "\n")

    with open("flagged_for_review_deepseek.jsonl", "w") as f:
        for r in flagged:
            f.write(json.dumps(r) + "\n")

    print(f"\nLabeled {len(labeled)} studies. {failed} failed after retry. "
          f"{len(flagged)} flagged for manual review "
          f"({len(flagged) / max(len(labeled), 1):.1%}).")


if __name__ == "__main__":
    main()