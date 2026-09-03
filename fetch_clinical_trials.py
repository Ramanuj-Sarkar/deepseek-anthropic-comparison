"""
Pull a diverse, messy sample of clinical trial free text from ClinicalTrials.gov
for the llm-finetune-eval-harness extraction project.

Docs: https://clinicaltrials.gov/data-api/about-api
Base: https://clinicaltrials.gov/api/v2/studies
"""

import json
import time
import requests

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

# Fields worth pulling: the free-text ones are what you'll extract FROM,
# the structured ones (conditions, phase, age, sex) are your GROUND TRUTH
# to check your extracted JSON against, since ClinicalTrials.gov already
# parsed some of this itself.
FIELDS = [
    "NCTId",                 # protocolSection.identificationModule.nctId
    "BriefTitle",            # protocolSection.identificationModule.briefTitle
    "BriefSummary",          # protocolSection.descriptionModule.briefSummary
    "DetailedDescription",   # protocolSection.descriptionModule.detailedDescription
    "EligibilityCriteria",   # protocolSection.eligibilityModule.eligibilityCriteria (the messy block)
    "MinimumAge",            # protocolSection.eligibilityModule.minimumAge
    "MaximumAge",            # protocolSection.eligibilityModule.maximumAge
    "Sex",                   # protocolSection.eligibilityModule.sex
    "HealthyVolunteers",     # protocolSection.eligibilityModule.healthyVolunteers
    "Condition",             # protocolSection.conditionsModule.conditions (array)
    "Phase",                 # protocolSection.designModule.phases (array)
    "StudyType",             # protocolSection.designModule.studyType
    "OverallStatus",         # protocolSection.statusModule.overallStatus
]

# Pull from several condition areas so your model sees varied vocabulary,
# criteria styles, and document lengths rather than one narrow specialty.
CONDITIONS = [
    "Diabetes",
    "Breast Cancer",
    "Alzheimer Disease",
    "Rheumatoid Arthritis",
    "Major Depressive Disorder",
    "Hypertension",
    "Asthma",
    "Parkinson Disease",
]

PAGE_SIZE = 100          # max allowed is 1000, but 100 keeps pages manageable
TARGET_PER_CONDITION = 300  # ~300 studies x 8 conditions ≈ 2,400 documents


def fetch_condition(condition: str, target: int):
    """Page through results for one condition until target or data runs out."""
    studies = []
    page_token = None

    while len(studies) < target:
        params = {
            "query.cond": condition,
            "filter.overallStatus": "COMPLETED",  # completed trials have final, stable text
            "fields": ",".join(FIELDS),
            "pageSize": PAGE_SIZE,
            "format": "json",
        }
        if page_token:
            params["pageToken"] = page_token

        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        for study in data.get("studies", []):
            protocol = study.get("protocolSection", {})
            eligibility = protocol.get("eligibilityModule", {})
            description = protocol.get("descriptionModule", {})

            criteria_text = eligibility.get("eligibilityCriteria", "")
            # skip studies with little or no free text -- nothing to extract from
            if len(criteria_text) < 200:
                continue

            studies.append({
                "nct_id": protocol.get("identificationModule", {}).get("nctId"),
                "title": protocol.get("identificationModule", {}).get("briefTitle"),
                "brief_summary": description.get("briefSummary", ""),
                "detailed_description": description.get("detailedDescription", ""),
                "eligibility_criteria": criteria_text,
                "min_age": eligibility.get("minimumAge"),
                "max_age": eligibility.get("maximumAge"),
                "sex": eligibility.get("sex"),
                "healthy_volunteers": eligibility.get("healthyVolunteers"),
                "conditions": protocol.get("conditionsModule", {}).get("conditions", []),
                "phases": protocol.get("designModule", {}).get("phases", []),
                "study_type": protocol.get("designModule", {}).get("studyType"),
            })

        page_token = data.get("nextPageToken")
        if not page_token:
            break  # ran out of studies for this condition before hitting target

        time.sleep(0.2)  # be polite to the API

    return studies[:target]


def main():
    all_studies = []
    for condition in CONDITIONS:
        print(f"Fetching '{condition}'...")
        studies = fetch_condition(condition, TARGET_PER_CONDITION)
        print(f"  -> got {len(studies)} studies with usable eligibility text")
        all_studies.extend(studies)

    with open("clinical_trials_raw.jsonl", "w") as f:
        for study in all_studies:
            f.write(json.dumps(study) + "\n")

    print(f"\nSaved {len(all_studies)} studies to clinical_trials_raw.jsonl")


if __name__ == "__main__":
    main()
