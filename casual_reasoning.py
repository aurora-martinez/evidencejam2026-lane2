
# causal_reasoning.py

# --------------------------------------------------
# CAUSAL RELATIONSHIP CLASSIFICATION
# --------------------------------------------------

CAUSAL_PREDICATES = {
    "causes",
    "contributes_to",
    "enables",
    "prevents",
    "disrupts"
}

ASSOCIATION_PREDICATES = {
    "associated_with",
    "correlated_with"
}

COMPLEX_PREDICATES = {
    "moderates",
    "mediates",
    "regulates"
}


def classify_relationship(edge):
    """
    Classify a causal graph edge.

    This function describes the relationship type.
    It does not infer the direction of an ecological
    outcome from the predicate alone.
    """

    predicate = edge.get("predicate", "")
    negated = edge.get("negated", False)

    if negated:
        return "explicit_no_relationship"

    if predicate in CAUSAL_PREDICATES:
        return "causal"

    if predicate in ASSOCIATION_PREDICATES:
        return "association"

    if predicate in COMPLEX_PREDICATES:
        return "complex_relationship"

    return "unknown"


# --------------------------------------------------
# EVIDENCE CERTAINTY
# --------------------------------------------------

CERTAINTY_SCORES = {
    "high": 4,
    "moderate": 3,
    "low": 2,
    "very_low": 1,
    "not_assessed": 0
}


def certainty_score(edge):
    """
    Convert an edge's certainty grade into
    a numeric ranking score.
    """

    grade = edge.get(
        "certainty_grade",
        "not_assessed"
    )

    return CERTAINTY_SCORES.get(grade, 0)


# --------------------------------------------------
# ANALYZE RETRIEVED EVIDENCE
# --------------------------------------------------

def analyze_evidence(edges):
    """
    Analyze a list of original graph edge dictionaries.

    Returns classified evidence with valid edge IDs.
    """

    results = []

    for edge in edges:

        relationship = classify_relationship(edge)

        results.append({
            "edge_id": edge.get("id"),
            "subject": edge.get("subject"),
            "object": edge.get("object"),
            "predicate": edge.get("predicate"),
            "relationship_type": relationship,
            "certainty": edge.get(
                "certainty_grade",
                "not_assessed"
            ),
            "certainty_score": certainty_score(edge),
            "original_sentence": edge.get(
                "original_sentence",
                ""
            )
        })

    # Sort strongest assessed certainty first
    results.sort(
        key=lambda item: item["certainty_score"],
        reverse=True
    )

    return results


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    test_edges = [
        {
            "id": "test-edge-1",
            "subject": "peatland_rewetting",
            "object": "carbon_sequestration",
            "predicate": "causes",
            "certainty_grade": "high",
            "negated": False
        },
        {
            "id": "test-edge-2",
            "subject": "water_table",
            "object": "methane_flux",
            "predicate": "associated_with",
            "certainty_grade": "moderate",
            "negated": False
        },
        {
            "id": "test-edge-3",
            "subject": "restoration",
            "object": "biodiversity",
            "predicate": "causes",
            "certainty_grade": "low",
            "negated": True
        }
    ]

    results = analyze_evidence(test_edges)

    for result in results:
        print(
            result["edge_id"],
            "|",
            result["relationship_type"],
            "| certainty:",
            result["certainty"]
        )
