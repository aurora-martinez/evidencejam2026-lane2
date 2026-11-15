
import re
import json
from pathlib import Path

from retrieval import edges, nodes


# --------------------------------------------------
# 1. SEARCH CONFIGURATION
# --------------------------------------------------

FIELD_WEIGHTS = {
    "name": 4.0,
    "entity_term": 3.0,
    "measured_attribute": 3.0,
    "embedding_text": 0.5,
}

INTERVENTION_TYPES = {
    "management_intervention",
    "environmental_process",
}

# Set to False if the entity_type filter removes too many
# potentially relevant intervention nodes.
FILTER_INTERVENTION_TYPES = True

MIN_SCORE = 2.0


# --------------------------------------------------
# 2. SYNONYMS
# --------------------------------------------------

SYNONYMS = {
    "invasive": [
        "non-native",
        "alien",
        "exotic",
    ],
    "wildfire": [
        "fire",
        "wildland fire",
    ],
    "prescribed burn": [
        "burning",
        "controlled burn",
    ],
    "mowing": [
        "cutting",
        "vegetation cutting",
    ],
    "regeneration": [
        "regrowth",
        "recruitment",
    ],
    "survival": [
        "persistence",
        "establishment",
    ],
    "growth": [
        "development",
        "biomass",
    ],
    "abundance": [
        "density",
        "population size",
    ],
    "richness": [
        "species richness",
        "taxonomic richness",
    ],
}


def expand_synonyms(term):
    """Return a term and its synonyms, including reverse matches."""

    term = term.lower().strip()
    expanded = {term}

    for main_term, alternatives in SYNONYMS.items():
        all_terms = [main_term] + alternatives

        if term in all_terms:
            expanded.update(all_terms)

    return expanded


# --------------------------------------------------
# 3. FIELD-AWARE MATCHING
# --------------------------------------------------

def get_node_fields(node):
    """Return searchable node fields separately."""

    return {
        field: str(node.get(field) or "").lower()
        for field in FIELD_WEIGHTS
    }


def contains_term(text, term):
    """Match a whole word or phrase, including simple plurals."""

    term = term.lower().strip()

    pattern = (
        r"\b"
        + re.escape(term)
        + r"(?:s|es)?\b"
    )

    return re.search(pattern, text, flags=re.IGNORECASE) is not None
def match_term(node, term):
    """
    Find the best field in which a term appears.

    Returns the field weight, or zero if there is no match.
    """

    fields = get_node_fields(node)
    best_score = 0.0

    for field, text in fields.items():
        if contains_term(text, term):
            best_score = max(best_score, FIELD_WEIGHTS[field])

    return best_score


def match_group(node, group):
    """
    Match one group of terms.

    A list means AND: every term must match.
    A string means one term.
    A dictionary supports field-specific matching.

    Examples:

        "wildfire"

        ["wildfire", "boreal"]

        {
            "entity_term": ["invasive"],
            "measured_attribute": ["abundance"]
        }
    """

    fields = get_node_fields(node)

    # Field-specific matching
    if isinstance(group, dict):

        total_score = 0.0

        for field, terms in group.items():

            if field not in FIELD_WEIGHTS:
                raise ValueError(f"Unknown node field: {field}")

            if isinstance(terms, str):
                terms = [terms]

            for term in terms:

                alternatives = expand_synonyms(term)

                best_score = 0.0

                for alternative in alternatives:
                    if contains_term(fields[field], alternative):
                        best_score = FIELD_WEIGHTS[field]
                        break

                # AND: every required field term must match.
                if best_score == 0:
                    return 0.0

                total_score += best_score

        return total_score

    # Convert a single string into a one-term group.
    if isinstance(group, str):
        terms = [group]

    else:
        terms = group

    total_score = 0.0

    # AND: every term must match somewhere in the node.
    for term in terms:

        alternatives = expand_synonyms(term)

        best_score = 0.0

        for alternative in alternatives:
            score = match_term(node, alternative)
            best_score = max(best_score, score)

        if best_score == 0:
            return 0.0

        total_score += best_score

    return total_score


def match_plan_groups(node, groups):
    """
    Match a node against a collection of groups.

    OR between groups.
    AND within each group.

    Use the best matching group rather than adding scores
    from alternative groups. This avoids rewarding a node
    just because it matches several alternative queries.
    """

    if not groups:
        return 0.0

    scores = [
        match_group(node, group)
        for group in groups
    ]

    return max(scores)


# --------------------------------------------------
# 4. INTERVENTION NODE FILTER
# --------------------------------------------------

def is_valid_intervention(node):
    """Check whether a node is an appropriate intervention."""

    if not FILTER_INTERVENTION_TYPES:
        return True

    entity_type = node.get("entity_type")

    # Keep nodes without an entity_type rather than silently
    # discarding potentially useful evidence.
    if not entity_type:
        return True

    return entity_type in INTERVENTION_TYPES


# --------------------------------------------------
# 5. SEARCH DIRECT EDGES
# --------------------------------------------------

def search_plan(plan, top_k=15):
    """
    Search for direct edges using an intervention/outcome plan.

    Example plan:

    {
        "interventions": [
            ["wildfire", "boreal"],
            "prescribed burn"
        ],
        "outcomes": [
            ["conifer", "regeneration"],
            "survival"
        ]
    }
    """

    intervention_groups = plan.get("interventions", [])
    outcome_groups = plan.get("outcomes", [])

    intervention_scores = {}
    outcome_scores = {}

    # Find matching intervention nodes.
    for node_id, node in nodes.items():

        if not is_valid_intervention(node):
            continue

        score = match_plan_groups(node, intervention_groups)

        if score > 0:
            intervention_scores[node_id] = score

    # Find matching outcome nodes.
    for node_id, node in nodes.items():

        score = match_plan_groups(node, outcome_groups)

        if score > 0:
            outcome_scores[node_id] = score

    # Find edges connecting matching nodes.
    results = []

    for edge in edges:

        subject_id = edge["subject"]
        object_id = edge["object"]

        if (
            subject_id in intervention_scores
            and object_id in outcome_scores
        ):

            score = (
                intervention_scores[subject_id]
                + outcome_scores[object_id]
            )

            results.append({
                "edge": edge,
                "score": score,
                "intervention_score": intervention_scores[subject_id],
                "outcome_score": outcome_scores[object_id],
            })

    results.sort(
        key=lambda result: result["score"],
        reverse=True,
    )

    # Diagnostics help distinguish different kinds of failure.
    diagnostics = {
        "matching_intervention_nodes": len(intervention_scores),
        "matching_outcome_nodes": len(outcome_scores),
        "matching_direct_edges": len(results),
    }

    if results:
        status = "evidence_found"

    elif intervention_scores or outcome_scores:
        status = "no_connecting_edge"

    else:
        status = "no_evidence"

    return {
        "status": status,
        "results": results[:top_k],
        "diagnostics": diagnostics,
        "searched_for": plan,
    }


# --------------------------------------------------
# 6. BASIC QUESTION PARSER
# --------------------------------------------------

STOP_WORDS = {
    "the", "and", "for", "with", "from", "that",
    "this", "what", "which", "where", "when",
    "does", "do", "how", "are", "into", "they",
    "their", "have", "has", "such", "also",
    "after", "each", "versus", "about", "can",
    "effect", "effects", "impact", "impacts",
    "influence", "influences",
}


QUESTION_PATTERNS = [
    r"what effect does (.+?) have on (.+)",
    r"what is the effect of (.+?) on (.+)",
    r"what impact does (.+?) have on (.+)",
    r"what is the impact of (.+?) on (.+)",
    r"how does (.+?) affect (.+)",
    r"how do (.+?) affect (.+)",
    r"what influence does (.+?) have on (.+)",
]


def clean_phrase(phrase):
    """Clean punctuation and whitespace from an extracted phrase."""

    phrase = phrase.lower().strip()
    phrase = re.sub(r"[?.!,;:]+$", "", phrase)
    return phrase.strip()


def extract_keywords(phrase):
    """Remove common words from a phrase."""

    words = re.findall(r"[a-z]+", phrase.lower())

    return [
        word for word in words
        if word not in STOP_WORDS and len(word) > 2
    ]


def parse_question(question):
    """
    Parse simple intervention-outcome questions.

    Returns None if the question does not match a supported format.
    """

    question = question.lower().strip()

    for pattern in QUESTION_PATTERNS:

        match = re.search(pattern, question)

        if match:

            intervention = clean_phrase(match.group(1))
            outcome = clean_phrase(match.group(2))

            return {
                "interventions": [intervention],
                "outcomes": [outcome],
            }

    return None


# --------------------------------------------------
# 7. SEARCH A NATURAL-LANGUAGE QUESTION
# --------------------------------------------------

def search_question(question, top_k=15):
    """Parse a question and search the graph."""

    plan = parse_question(question)

    if plan is None:
        return {
            "status": "unsupported_question_format",
            "results": [],
            "searched_for": question,
        }

    result = search_plan(plan, top_k=top_k)

    # Separate weak matches from stronger matches.
    if result["results"]:

        if result["results"][0]["score"] < MIN_SCORE:
            result["status"] = "weak_matches"

    return result


# --------------------------------------------------
# 8. DISPLAY RESULTS
# --------------------------------------------------

def display_results(result):

    print("\nSTATUS:", result["status"])

    if "diagnostics" in result:
        print("\nSEARCH DIAGNOSTICS:")
        for key, value in result["diagnostics"].items():
            print(f"{key}: {value}")

    if not result["results"]:
        print("\nNo direct evidence found.")
        print("Searched for:", result["searched_for"])
        return

    print(f"\nFound {len(result['results'])} matching edges")

    for item in result["results"]:

        edge = item["edge"]

        subject = nodes[edge["subject"]]
        outcome = nodes[edge["object"]]

        print("\n" + "=" * 60)

        print("EDGE:", edge["id"])
        print("SCORE:", item["score"])

        print("SUBJECT:", subject.get("name"))
        print("OUTCOME:", outcome.get("name"))

        print("PREDICATE:", edge.get("predicate"))

        print(
            "DIRECTION QUALIFIER:",
            outcome.get("state_or_change_qualifier")
        )

        print("CERTAINTY:", edge.get("certainty_grade"))
        print("EVIDENCE:", edge.get("original_sentence"))


# --------------------------------------------------
# 9. GRAPH EXPLORATION
# --------------------------------------------------

def explore_graph():
    """Print basic statistics about the graph."""

    entity_types = {}
    entity_terms = {}
    outgoing = {}

    for node_id, node in nodes.items():

        entity_type = node.get("entity_type", "unknown")
        entity_types[entity_type] = (
            entity_types.get(entity_type, 0) + 1
        )

        entity_term = node.get("entity_term")

        if entity_term:
            entity_terms[entity_term] = (
                entity_terms.get(entity_term, 0) + 1
            )

        outgoing[node_id] = []

    for edge in edges:
        outgoing.setdefault(edge["subject"], []).append(edge["object"])

    print("\nNODE COUNT:", len(nodes))
    print("EDGE COUNT:", len(edges))

    print("\nENTITY TYPES:")
    for entity_type, count in sorted(
        entity_types.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(entity_type, count)

    print("\nMOST COMMON ENTITY TERMS:")
    for term, count in sorted(
        entity_terms.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:20]:
        print(term, count)

    # Count possible two-edge paths.
    path_count = 0

    for middle_node in outgoing:
        path_count += len(outgoing[middle_node])

        # Count actual directed two-edge paths:
    # A -> B -> C

    two_edge_paths = 0

    for middle_node in outgoing:
        two_edge_paths += len(outgoing[middle_node]) * sum(
            1 for edge in edges
            if edge["object"] == middle_node
        )

    print("\nTWO-EDGE PATHS:", two_edge_paths)


# --------------------------------------------------
# 10. EVALUATION
# --------------------------------------------------

def evaluate_retrieval(gold_file, top_k=15):
    """
    Evaluate retrieval against manually labelled relevant edge IDs.

    Expected JSON format:

    [
        {
            "question_id": "c4-modified",
            "question": "How do planting-stock traits affect survival?",
            "plan": {
                "interventions": ["planting-stock traits"],
                "outcomes": ["survival"]
            },
            "relevant_edge_ids": [12, 18, 24]
        }
    ]
    """

    with open(gold_file, encoding="utf-8") as file:
        gold_questions = json.load(file)

    precision_scores = []
    recall_scores = []

    for item in gold_questions:

        result = search_plan(item["plan"], top_k=top_k)

        retrieved_ids = {
            result_item["edge"]["id"]
            for result_item in result["results"]
        }

        relevant_ids = set(item["relevant_edge_ids"])

        true_positives = len(retrieved_ids & relevant_ids)

        precision = (
            true_positives / len(retrieved_ids)
            if retrieved_ids else 0.0
        )

        recall = (
            true_positives / len(relevant_ids)
            if relevant_ids else 0.0
        )

        precision_scores.append(precision)
        recall_scores.append(recall)

        print("\nQUESTION:", item["question_id"])
        print("PRECISION:", round(precision, 3))
        print("RECALL:", round(recall, 3))

    if not gold_questions:
        print("No gold questions to evaluate.")
        return

    print("\nAVERAGE PRECISION:", round(
        sum(precision_scores) / len(precision_scores), 3
    ))

    print("AVERAGE RECALL:", round(
        sum(recall_scores) / len(recall_scores), 3
    ))


# --------------------------------------------------
# 11. RUN
# --------------------------------------------------

if __name__ == "__main__":
    title = input().strip()

    if title == "SEARCH PLAN":
        plan = {
            "interventions": [
                ["wildfire", "boreal"],
                ["fire", "boreal"]
            ],
            "outcomes": [
                ["conifer", "regeneration"],
                ["conifer", "regrowth"]
            ],
        }

        display_results(search_plan(plan))