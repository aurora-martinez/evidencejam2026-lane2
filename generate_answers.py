
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import yaml
import time
from search import search_question
from retrieval import nodes, edges as graph_edges, graph
source_documents = {}

for doc in graph.get("source_documents", []):
    for key in (doc.get("document_id"), doc.get("doi")):
        if key:
            source_documents[key] = doc
from answer_generator import generate_answer, prepare_evidence

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

TEAM_NAME = "YC Startup"
TOP_K = 20
QUESTION_RETRIES = 2
RETRY_DELAY = 3
time.sleep(RETRY_DELAY)
VALID_DIRECTIONS = {
    "increase", "decrease", "no_effect", "mixed", "unclear"
}

DIRECTION_PATTERNS = {
    "no_effect": [
        r"\bno significant (?:effect|change|difference)\b",
        r"\bno detectable (?:effect|change|difference)\b",
        r"\bno effect\b",
        r"\bnot significantly (?:affect|change|alter)\b",
    ],
    "increase": [
        r"\bincreas(?:e|es|ed|ing)\b",
        r"\benhanc(?:e|es|ed|ing)\b",
        r"\b(?:higher|greater|more)\b",
        r"\bpromot(?:e|es|ed|ing)\b",
        r"\bimprov(?:e|es|ed|ing)\b",
    ],
    "decrease": [
        r"\bdecreas(?:e|es|ed|ing)\b",
        r"\breduc(?:e|es|ed|ing)\b",
        r"\b(?:lower|less|fewer)\b",
        r"\bsuppress(?:es|ed|ing)?\b",
        r"\bdeclin(?:e|es|ed|ing)\b",
    ],
}

CERTAINTY_MAP = {
    "high": "high",
    "strong": "high",
    "moderate": "moderate",
    "medium": "moderate",
    "low": "low",
    "weak": "low",
    "very_low": "very_low",
    "very low": "very_low",
}

CERTAINTY_VALUES = {
    "high": 1.0,
    "moderate": 0.75,
    "low": 0.45,
    "very_low": 0.20,
}

# Only allow edge IDs that actually exist in the graph.
VALID_EDGE_IDS = {
    str(edge["id"])
    for edge in graph_edges
}


# --------------------------------------------------
# LOAD QUESTIONS
# --------------------------------------------------

def load_questions(path):

    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if isinstance(data, dict):
        data = data.get("questions", [])

    if not isinstance(data, list):
        raise ValueError(
            "Expected a list of questions in YAML."
        )

    for item in data:

        if not isinstance(item, dict):
            raise ValueError("Invalid question entry.")

        if "id" not in item or "question" not in item:
            raise ValueError(
                "Each question must have an id and question."
            )

    return data


# --------------------------------------------------
# EXTRACT EVIDENCE
# --------------------------------------------------

def get_unique_edges(search_result):
    """
    Extract unique, valid graph edges from retrieval.
    """

    unique = {}

    for item in search_result.get("results", []):

        edge = item.get("edge", {})
        edge_id = str(edge.get("id", ""))

        if edge_id in VALID_EDGE_IDS:
            unique[edge_id] = edge

    return list(unique.values())


# --------------------------------------------------
# DETERMINE EDGE DIRECTION
# --------------------------------------------------

def find_directions(text):
    """
    Identify directional language in text.
    """

    text = str(text or "").lower()

    matches = set()

    for direction, patterns in DIRECTION_PATTERNS.items():

        for pattern in patterns:

            if re.search(pattern, text):
                matches.add(direction)
                break

    return matches


def infer_edge_direction(edge):
    """
    Infer the direction of an individual edge.

    Prefer explicit outcome qualifiers.
    Fall back to causal language and evidence text.

    Ambiguous or conflicting evidence is unclear.
    """

    if edge.get("negated") is True:
        return "unclear"

    outcome_node = nodes.get(
        edge.get("object"), {}
    )

    qualifier = str(
        outcome_node.get("state_or_change_qualifier")
        or ""
    )

    qualifier_matches = find_directions(qualifier)

    if len(qualifier_matches) == 1:
        return next(iter(qualifier_matches))

    if len(qualifier_matches) > 1:
        return "unclear"

    predicate = str(
        edge.get("predicate") or ""
    ).lower()

    # Association alone does not establish a
    # directional causal effect.
    if predicate in {
        "associated_with",
        "correlated_with"
    }:
        return "unclear"

    causal_language = str(
        edge.get("causal_language") or ""
    )

    language_matches = find_directions(
        causal_language
    )

    if len(language_matches) == 1:
        return next(iter(language_matches))

    if len(language_matches) > 1:
        return "unclear"

    # The original sentence may contain a
    # direction, but is harder to interpret.
    sentence = str(
        edge.get("original_sentence") or ""
    )

    sentence_matches = find_directions(sentence)

    if len(sentence_matches) == 1:
        return next(iter(sentence_matches))

    return "unclear"


def aggregate_direction(edges):
    """
    Combine directions across retrieved edges.
    """

    if not edges:
        return "unclear"

    directions = [
        infer_edge_direction(edge)
        for edge in edges
    ]

    known = [
        direction
        for direction in directions
        if direction != "unclear"
    ]

    if not known:
        return "unclear"

    if len(set(known)) > 1:
        return "mixed"

    # Avoid claiming a direction when most
    # retrieved evidence is ambiguous.
    if len(known) < len(edges) / 2:
        return "unclear"

    return known[0]


# --------------------------------------------------
# CERTAINTY
# --------------------------------------------------

def normalize_certainty(raw_grade):

    raw = str(
        raw_grade or ""
    ).lower().strip().replace("-", "_")

    return CERTAINTY_MAP.get(
        raw,
        "very_low"
    )


def infer_certainty(edges):
    """
    Determine the most common evidence grade.
    """

    if not edges:
        return "very_low"

    grades = [
        normalize_certainty(
            edge.get("certainty_grade")
        )
        for edge in edges
    ]

    return Counter(grades).most_common(1)[0][0]


# --------------------------------------------------
# EVIDENCE-BASED CONFIDENCE SCORING
# --------------------------------------------------

def estimate_confidence(edges, direction, search_result):
    """
    Calculate a confidence score between 0 and 1.

    Components:

    40% Evidence certainty
    30% Direction agreement
    20% Retrieval relevance
    10% Evidence quantity

    This is a heuristic confidence estimate,
    not a statistically calibrated probability.
    """

    if not edges:
        return 0.0

    # ----------------------------------------------
    # 1. EVIDENCE CERTAINTY (40%)
    # ----------------------------------------------

    certainty_scores = []

    for edge in edges:

        certainty = normalize_certainty(
            edge.get("certainty_grade")
        )

        certainty_scores.append(
            CERTAINTY_VALUES[certainty]
        )

    certainty_score = (
        sum(certainty_scores)
        / len(certainty_scores)
    )

    # ----------------------------------------------
    # 2. DIRECTION AGREEMENT (30%)
    # ----------------------------------------------

    edge_directions = [
        infer_edge_direction(edge)
        for edge in edges
    ]

    known_directions = [
        d for d in edge_directions
        if d != "unclear"
    ]

    if not known_directions:
        agreement_score = 0.0

    elif direction == "mixed":
        # Conflicting evidence reduces confidence
        # in a single directional conclusion.
        agreement_score = 0.35

    elif direction == "unclear":
        agreement_score = 0.0

    else:
        matching = sum(
            1 for d in edge_directions
            if d == direction
        )

        agreement_score = (
            matching / len(edges)
        )

    # ----------------------------------------------
    # 3. RETRIEVAL RELEVANCE (20%)
    # ----------------------------------------------

    valid_ids = {
        str(edge["id"])
        for edge in edges
    }

    retrieval_scores = [
        item["score"]
        for item in search_result.get("results", [])
        if (
            str(item.get("edge", {}).get("id"))
            in valid_ids
            and isinstance(
                item.get("score"),
                (int, float)
            )
            and math.isfinite(item["score"])
        )
    ]

    if retrieval_scores:

        average_score = (
            sum(retrieval_scores)
            / len(retrieval_scores)
        )

        # A score of 8 corresponds roughly to
        # strong matches in two high-weight fields.
        # This is a heuristic normalization.
        relevance_score = min(
            max(average_score / 8.0, 0.0),
            1.0
        )

    else:
        relevance_score = 0.0

    # ----------------------------------------------
    # 4. EVIDENCE QUANTITY (10%)
    # ----------------------------------------------

    unique_edge_ids = {
        str(edge["id"])
        for edge in edges
    }

    quantity_score = min(
        len(unique_edge_ids) / 5.0,
        1.0
    )

    # ----------------------------------------------
    # FINAL WEIGHTED SCORE
    # ----------------------------------------------

    confidence = (
        0.40 * certainty_score
        + 0.30 * agreement_score
        + 0.20 * relevance_score
        + 0.10 * quantity_score
    )

    return round(
        max(0.0, min(confidence, 1.0)),
        3
    )


# --------------------------------------------------
# GENERATE ANSWER TEXT
# --------------------------------------------------


def build_answer(question, edges, direction):
    """
    Generate a publication-level evidence report without Qwen.
    """

    if not edges:
        return (
            "Insufficient evidence: no direct graph "
            "relationships were retrieved for this question."
        )

    # Group edges by unique publication
    publications = {}

    for edge in edges:

        source_id = edge.get("source_document")

        if not source_id:
            source_id = edge.get("id", "Unknown source")

        if source_id not in publications:

            doc = source_documents.get(source_id, {})

            publications[source_id] = {
                "title": doc.get("title", "Title unavailable"),
                "authors": doc.get("authors", []),
                "year": doc.get("year"),
                "journal": doc.get("journal"),
                "doi": doc.get("doi", source_id),
                "location": doc.get("location"),
                "findings": [],
                "seen": set()
            }

        subject = nodes.get(edge.get("subject"), {})
        outcome = nodes.get(edge.get("object"), {})

        finding = edge.get("original_sentence", "")

        finding_key = (
            subject.get("name"),
            outcome.get("name"),
            finding
        )

        if finding and finding_key not in publications[source_id]["seen"]:

            publications[source_id]["seen"].add(finding_key)

            publications[source_id]["findings"].append({
                "subject": subject.get("name", "Unknown intervention"),
                "outcome": outcome.get("name", "Unknown outcome"),
                "direction": outcome.get("state_or_change_qualifier"),
                "negated": edge.get("negated"),
                "sentence": finding
            })

    # Build report
    report = [
        f"Research question: {question}",
        "",
        f"Retrieved evidence from {len(publications)} unique publications "
        f"({len(edges)} graph edges).",
        "",
        "PUBLICATION FINDINGS",
        ""
    ]

    for index, (source_id, publication) in enumerate(
        publications.items(), 1
    ):

        authors = publication["authors"]

        if isinstance(authors, list):
            author_text = ", ".join(authors[:3])
            if len(authors) > 3:
                author_text += " et al."
        else:
            author_text = str(authors) if authors else "Authors unavailable"

        metadata = [
            author_text,
            str(publication["year"]) if publication["year"] else None,
            publication["journal"],
            publication["doi"]
        ]

        metadata_text = " | ".join(
            str(value) for value in metadata if value
        )

        report.append(
            f"{index}. {publication['title']}"
        )
        report.append(metadata_text)

        if publication["location"]:
            report.append(
                f"Location: {publication['location']}"
            )

        findings = publication["findings"]

        if not findings:
            report.append("No usable finding sentence was retrieved.")
        else:
            report.append("Retrieved findings:")

            # Show up to two distinct evidence statements.
            for finding in findings[:2]:

                sentence = finding["sentence"]

                # Keep displayed source text concise.
                words = sentence.split()
                if len(words) > 25:
                    sentence = " ".join(words[:25]) + "..."

                report.append(
                    f"- {finding['subject']} -> "
                    f"{finding['outcome']}: {sentence}"
                )

        report.append("")

    # Overall interpretation
    report.append("OVERALL INTERPRETATION")

    report.append(
        f"The aggregate direction calculated from the retrieved "
        f"graph edges is: {direction or 'undetermined'}."
    )

    report.append(
        "These results represent retrieved graph evidence, "
        "not a comprehensive review of all scientific literature. "
        "Different outcomes, species, sites and treatments should "
        "not be treated as directly comparable."
    )

    return "\n".join(report)

# --------------------------------------------------
# QWEN ANSWER GENERATION
# --------------------------------------------------


def format_answer(question_item, search_result):

    question_id = str(question_item["id"])
    question = question_item["question"]

    # Original graph edges, used for calculations
    unique_edges = get_unique_edges(search_result)


    # ----------------------------------------------
    # NO EVIDENCE
    # ----------------------------------------------

    if not unique_edges:

        searched_plan = search_result.get(
            "searched_for", {}
        )

        if isinstance(searched_plan, dict):

            interventions = searched_plan.get(
                "interventions", []
            )

            outcomes = searched_plan.get(
                "outcomes", []
            )

            search_description = (
                "No direct edge found between "
                f"intervention terms {interventions} "
                f"and outcome terms {outcomes}. "
                "Indirect paths were not checked."
            )

        else:

            search_description = (
                "No matching direct graph edges "
                f"were retrieved for: {question}. "
                "Indirect paths were not checked."
            )

        diagnostics = search_result.get(
            "diagnostics", {}
        )

        intervention_matches = diagnostics.get(
            "matching_intervention_nodes", 0
        )

        outcome_matches = diagnostics.get(
            "matching_outcome_nodes", 0
        )

        if intervention_matches and outcome_matches:
            no_evidence_confidence = 0.35
        elif intervention_matches or outcome_matches:
            no_evidence_confidence = 0.45
        else:
            no_evidence_confidence = 0.50

        return {
            "id": question_id,
            "status": "no_evidence",
            "answer": (
                "No direct graph evidence was retrieved "
                "to answer this question."
            ),
            "direction": None,
            "confidence": no_evidence_confidence,
            "certainty": None,
            "edge_ids": [],
            "searched_for": search_description,
            "extras": []
        }

    # ----------------------------------------------
    # EVIDENCE FOUND
    # ----------------------------------------------

    direction = aggregate_direction(unique_edges)

    certainty = infer_certainty(unique_edges)

    confidence = estimate_confidence(
        unique_edges,
        direction,
        search_result
    )

    # Deterministic fallback uses original edge list
    fallback_answer = build_answer(
        question,
        unique_edges,
        direction
    )

    # Qwen receives publication-level evidence
    answer_text = generate_answers(
        question,
        search_result,
        unique_edges,
        direction
    )

    return {
        "id": question_id,
        "status": "answered",
        "answer": answer_text,
        "direction": direction,
        "confidence": confidence,
        "certainty": certainty,
        "edge_ids": [
            str(edge["id"])
            for edge in unique_edges
        ],
        "searched_for": None,
        "extras": []
    }


# --------------------------------------------------
# FALLBACK WHEN QWEN OR RETRIEVAL FAILS
# --------------------------------------------------

def processing_failure_answer(question_item, error):
    """
    Produce a schema-valid entry if processing fails.

    IMPORTANT:
    This does not mean the graph lacks evidence.
    Confidence is therefore zero.
    """

    return {
        "id": str(question_item["id"]),
        "status": "processing_error",
        "answer": (
            "Unable to determine an answer because "
            "question processing did not complete."
        ),
        "direction": None,
        "confidence": 0.0,
        "certainty": None,
        "edge_ids": [],
        "searched_for": (
            "The graph search was not completed "
            f"due to {type(error).__name__}. "
            "No conclusion about evidence "
            "availability can be made."
        ),
        "extras": []
    }


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

def save_answers(output_path, answers, team_name):
    """
    Save answers after each processed question.

    Write to a temporary file first to reduce
    the risk of corrupting answers.json.
    """

    output = {
        "team": team_name,
        "answers": answers
    }

    output_file = Path(output_path)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temp_file = output_file.with_name(
        output_file.name + ".tmp"
    )

    with open(
        temp_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )

    temp_file.replace(output_file)

    return output


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def generate_answers(
    questions_path,
    output_path,
    team_name=TEAM_NAME
):

    questions = load_questions(
        questions_path
    )

    answers = []

    for index, item in enumerate(
        questions,
        start=1
    ):

        question_id = item["id"]
        question = item["question"]

        print(
            f"\n[{index}/{len(questions)}] "
            f"Processing {question_id}...",
            flush=True
        )

        answer = None

        # Retry the entire question if the
        # Qwen planner or retrieval fails.
        for attempt in range(QUESTION_RETRIES):

            try:

                print(
                    f"Question attempt "
                    f"{attempt + 1}/{QUESTION_RETRIES}",
                    flush=True
                )

                search_result = search_question(
                    question,
                    top_k=TOP_K
                )

                answer = format_answer(
                    item,
                    search_result
                )

                break

            except Exception as error:
                import traceback
                traceback.print_exc()

                print(
                    f"ERROR processing {question_id}: "
                    f"{type(error).__name__}: {error}",
                    flush=True
                )

                if attempt == QUESTION_RETRIES - 1:

                    answer = processing_failure_answer(
                        item,
                        error
                    )

                else:

                    print(
                        f"Waiting {RETRY_DELAY} seconds "
                        "before retrying...",
                        flush=True
                    )

        answers.append(answer)

        # Save progress after every question.
        save_answers(
            output_path,
            answers,
            team_name
        )

        print(
            f"Status: {answer['status']}",
            flush=True
        )

        print(
            f"Confidence: {answer['confidence']}",
            flush=True
        )

    output = save_answers(
        output_path,
        answers,
        team_name
    )

    print(
        f"\nSaved {len(answers)} answers "
        f"to {output_path}",
        flush=True
    )

    return output


# --------------------------------------------------
# COMMAND LINE
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Generate answers.json from ecological "
            "questions using Qwen and graph retrieval."
        )
    )

    parser.add_argument(
        "questions",
        help="Input questions YAML file"
    )

    parser.add_argument(
        "output",
        help="Output answers JSON file"
    )

    parser.add_argument(
        "--team",
        default=TEAM_NAME,
        help="Team name"
    )

    args = parser.parse_args()

    generate_answers(
        args.questions,
        args.output,
        team_name=args.team
    )
