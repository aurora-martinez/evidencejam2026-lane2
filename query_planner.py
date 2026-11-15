
import os
import json
import re
import hashlib
import time
from pathlib import Path

from openai import OpenAI, APITimeoutError, APIConnectionError
from dotenv import load_dotenv


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "query_cache"
CACHE_DIR.mkdir(exist_ok=True)

load_dotenv(BASE_DIR / ".env")

client = OpenAI(
    base_url=os.environ["BRINE_URL"],
    api_key=os.environ["BRINE_API_KEY"],
    timeout=45.0,
    max_retries=0
)

PLAN_VERSION = 2


# --------------------------------------------------
# VALIDATE SEARCH PLANS
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
# MERGE SYNONYMS INTO SEARCH LISTS
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
            terms = [concept] + synonyms.get(concept, [])

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
# PARSE MODEL OUTPUT
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
# CACHE HELPERS
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
# QUESTION PLANNER
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

   
    # ----------------------------------------------
    # QWEN API REQUEST WITH AUTOMATIC RETRIES
    # ----------------------------------------------

    response = None

    for attempt in range(2):

        try:

            response = client.chat.completions.create(
                model=os.environ["BRINE_MODEL"],
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You extract interventions, "
                            "outcomes and scientific synonyms "
                            "from ecological questions. "
                            "Return valid JSON only. "
                            "Do not reason aloud."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt + "\n/no_think"
                    }
                ],
                max_tokens=3000,
                temperature=0
            )

            

            break

        except (APITimeoutError, APIConnectionError) as error:


            if attempt == 1:
                raise RuntimeError(
                    "Qwen is unavailable after "
                    "2 attempts. Check BRINE server "
                    "availability and try again."
                ) from error


            time.sleep(2)

    # ----------------------------------------------
    # CHECK RESPONSE
    # ----------------------------------------------

    choice = response.choices[0]
    content = choice.message.content

    if not content:
        raise ValueError(
            "Qwen returned no content. "
            f"Finish reason: {choice.finish_reason}"
        )

    # ----------------------------------------------
    # DEBUG OUTPUT
    # ----------------------------------------------

   

    # ----------------------------------------------
    # PARSE AND VALIDATE
    # ----------------------------------------------

    try:

        if choice.finish_reason == "length":
            raise ValueError(
                "Qwen response was truncated."
            )

        plan = parse_json_response(content)
        plan = validate_plan(plan)

    except (json.JSONDecodeError, ValueError) as error:


        debug_path = (
            CACHE_DIR / "debug_response.txt"
        )

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


    # Merge synonyms into the two final lists
    return merge_search_terms(plan)


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    question = (
        "Which seed-mix species establish, persist "
        "and self-seed in each Ontario ELC ecosite "
        "type (e.g. dry/wet deciduous forest, "
        "meadow, riparian)?"
    )

    plan = plan_question(question)

   
