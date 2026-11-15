
import os
import json
import re
import hashlib
import time
from pathlib import Path

import requests
from dotenv import load_dotenv


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "query_cache"

CACHE_DIR.mkdir(exist_ok=True)

load_dotenv(BASE_DIR / ".env")

BRINE_URL = os.environ["BRINE_URL"].rstrip("/")
BRINE_API_KEY = os.environ["BRINE_API_KEY"]
BRINE_MODEL = os.environ["BRINE_MODEL"]

PLAN_VERSION = 2


# --------------------------------------------------
# 2. VALIDATE SEARCH PLANS
# --------------------------------------------------

def validate_plan(plan):
    """
    Ensure Qwen returned the expected structure.
    """

    if not isinstance(plan, dict):
        raise ValueError("Search plan must be a dictionary.")

    for key in ("interventions", "outcomes"):

        values = plan.get(key)

        if not isinstance(values, list):
            raise ValueError(f"{key} must be a list.")

        if not all(
            isinstance(value, str) and value.strip()
            for value in values
        ):
            raise ValueError(
                f"{key} must contain nonempty strings."
            )

    synonyms = plan.get("synonyms")

    if not isinstance(synonyms, dict):
        raise ValueError("synonyms must be a dictionary.")

    concepts = set(
        plan["interventions"] + plan["outcomes"]
    )

    for concept, terms in synonyms.items():

        if concept not in concepts:
            raise ValueError(
                f"Unknown synonym concept: {concept}"
            )

        if not isinstance(terms, list):
            raise ValueError(
                f"Synonyms for {concept} must be a list."
            )

        if not all(
            isinstance(term, str) and term.strip()
            for term in terms
        ):
            raise ValueError(
                f"Invalid synonym for {concept}"
            )

    return {
        "interventions": plan["interventions"],
        "outcomes": plan["outcomes"],
        "synonyms": synonyms
    }


# --------------------------------------------------
# 3. MERGE SYNONYMS INTO SEARCH LISTS
# --------------------------------------------------

def merge_search_terms(plan):
    """
    Merge synonyms into interventions and outcomes.
    Remove duplicates while preserving order.
    """

    synonyms = plan["synonyms"]

    def expand_terms(concepts):

        expanded = []
        seen = set()

        for concept in concepts:

            terms = [
                concept
            ] + synonyms.get(concept, [])

            for term in terms:

                term = term.strip()
                normalized = term.casefold()

                if term and normalized not in seen:

                    expanded.append(term)
                    seen.add(normalized)

        return expanded

    return {
        "interventions": expand_terms(
            plan["interventions"]
        ),

        "outcomes": expand_terms(
            plan["outcomes"]
        )
    }


# --------------------------------------------------
# 4. PARSE MODEL OUTPUT
# --------------------------------------------------

def parse_json_response(content):
    """
    Extract JSON from Qwen's response.
    """

    content = content.strip()

    content = re.sub(
        r"^```(?:json)?\s*",
        "",
        content
    )

    content = re.sub(
        r"\s*```$",
        "",
        content
    )

    start = content.find("{")
    end = content.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "No complete JSON object found."
        )

    return json.loads(content[start:end + 1])


# --------------------------------------------------
# 5. CACHE HELPERS
# --------------------------------------------------

def get_cache_path(question):
    """
    Cache by question text rather than question ID.
    Supports unseen ecological questions.
    """

    question_hash = hashlib.sha256(
        question.encode("utf-8")
    ).hexdigest()[:16]

    return CACHE_DIR / (
        f"plan_v{PLAN_VERSION}_{question_hash}.json"
    )


# --------------------------------------------------
# 6. QUESTION PLANNER
# --------------------------------------------------

def plan_question(question, question_id=None):
    """
    Extract interventions, outcomes, and synonyms
    from any ecological research question.

    Return two expanded lists:
    interventions and outcomes.
    """

    if not isinstance(question, str):
        raise TypeError("Question must be a string.")

    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    cache_path = get_cache_path(question)

    # ----------------------------------------------
    # LOAD CACHE
    # ----------------------------------------------

    if cache_path.exists():

        try:

            with open(
                cache_path,
                "r",
                encoding="utf-8"
            ) as file:

                cached = json.load(file)

            return merge_search_terms(
                validate_plan(cached)
            )

        except (ValueError, KeyError, TypeError):

            print(
                "Invalid cached plan; regenerating...",
                flush=True
            )

    # ----------------------------------------------
    # QWEN PROMPT
    # ----------------------------------------------

    prompt = f"""
You are an ecological research question interpreter.

Your ONLY task is to extract searchable concepts
from a research question for a Python graph
retrieval algorithm.

Do not answer the question.
Do not analyze scientific evidence.
Do not invent scientific findings.

DEFINITIONS:

INTERVENTIONS:
Treatments, management actions, disturbances,
environmental drivers, or explanatory variables
whose effects are being investigated.

OUTCOMES:
Ecological responses, effects, or dependent
variables being measured or investigated.

SYNONYMS:
Alternative scientific names, abbreviations,
or equivalent terms useful for graph retrieval.

RULES:

1. Identify the main interventions and outcomes.
2. Include all distinct interventions explicitly
   requested in the question.
3. Include all distinct outcomes explicitly
   requested in the question.
4. Do not include locations, ecosystems, or
   background context as interventions unless
   they are explanatory variables.
5. Avoid broad umbrella terms when more specific
   interventions are already listed.
6. Do not invent additional outcomes.
7. Provide up to 3 synonyms per concept.
8. Synonym keys must exactly match an intervention
   or outcome in the output lists.
9. Synonyms must preserve the scientific meaning.
   Do not treat opposite effects as synonyms.
10. If no interventions or outcomes are identifiable,
    use an empty list for that category.
11. Do not generate subquestions.
12. Return valid JSON only, without Markdown.

OUTPUT FORMAT:

{{
    "interventions": [
        "specific intervention or driver"
    ],
    "outcomes": [
        "specific ecological outcome"
    ],
    "synonyms": {{
        "specific intervention or outcome": [
            "alternative term",
            "abbreviation"
        ]
    }}
}}

Keep the JSON compact.
Use at most 3 synonyms per concept.
Do not include explanations.

QUESTION:
{question}
"""

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    # ----------------------------------------------
    # QWEN API REQUEST WITH AUTOMATIC RETRIES
    # ----------------------------------------------

    data = None

    for attempt in range(2):

        try:

            response = requests.post(

                f"{BRINE_URL}/chat/completions",

                headers={
                    "Authorization": f"Bearer {BRINE_API_KEY}",
                    "Content-Type": "application/json"
                },

                json={
                    "model": BRINE_MODEL,
                    "messages": messages,
                    "temperature": 0.2,
                    "max_tokens": 2500
                },

                timeout=120
            )

            response.raise_for_status()

            data = response.json()

            break

        except requests.exceptions.Timeout as error:

            if attempt == 1:
                raise RuntimeError(
                    "Qwen timed out after 2 attempts. "
                    "Check Nibi availability."
                ) from error

            time.sleep(2)

        except requests.exceptions.ConnectionError as error:

            if attempt == 1:
                raise RuntimeError(
                    "Could not connect to Qwen after "
                    "2 attempts. Check the Nibi endpoint."
                ) from error

            time.sleep(2)

    # ----------------------------------------------
    # CHECK RESPONSE
    # ----------------------------------------------

    if not data or not data.get("choices"):
        raise ValueError(
            "Qwen returned an empty response."
        )

    choice = data["choices"][0]

    content = choice["message"].get("content", "")

    finish_reason = choice.get("finish_reason")

    if not content:
        raise ValueError(
            "Qwen returned no content. "
            f"Finish reason: {finish_reason}"
        )

    # ----------------------------------------------
    # PARSE AND VALIDATE
    # ----------------------------------------------

    try:

        if finish_reason == "length":
            raise ValueError(
                "Qwen response was truncated."
            )

        plan = parse_json_response(content)

        plan = validate_plan(plan)

    except (json.JSONDecodeError, ValueError):

        debug_path = CACHE_DIR / "debug_response.txt"

        with open(
            debug_path,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(content)

        raise

    # ----------------------------------------------
    # CACHE SUCCESSFUL PLAN
    # ----------------------------------------------

    with open(
        cache_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            plan,
            file,
            indent=2,
            ensure_ascii=False
        )

    # ----------------------------------------------
    # RETURN EXPANDED SEARCH TERMS
    # ----------------------------------------------

    return merge_search_terms(plan)


# --------------------------------------------------
# 7. TEST
# --------------------------------------------------

if __name__ == "__main__":

    question = (
        "Which seed-mix species establish, persist "
        "and self-seed in each Ontario ELC ecosite "
        "type (e.g. dry/wet deciduous forest, "
        "meadow, riparian)?"
    )

    plan = plan_question(question)


