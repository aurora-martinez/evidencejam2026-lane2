
from retrieval import search_edges, nodes

# Sign values from the CAMO schema
QUALIFIER_SIGNS = {
    "increased": 1,
    "decreased": -1,
    "present": 1,
    "absent": -1,
    "introduced": 1,
    "removed": -1,
    "initiated": 1,
    "terminated": -1,
    "interrupted": -1,
    "aborted": -1,
    "occurred": 1,
    "ongoing": 1
}

PREDICATE_SIGNS = {
    "causes": 1,
    "contributes_to": 1,
    "enables": 1,
    "prevents": -1,
    "disrupts": -1
}


def interpret_edge(edge):
    subject = nodes[edge["subject"]]
    outcome = nodes[edge["object"]]

    # Explicit null result
    if (edge.get("negated") is True
            and edge.get("causal_language") == "no_relationship"):
        return "no_effect"

    subject_qualifier = subject.get(
        "state_or_change_qualifier"
    )
    outcome_qualifier = outcome.get(
        "state_or_change_qualifier"
    )

    # Unchanged states have no variable-level sign
    if (subject_qualifier == "unchanged"
            or outcome_qualifier == "unchanged"):
        return "unclear"

    subject_sign = QUALIFIER_SIGNS.get(
        subject_qualifier, 1
    )
    outcome_sign = QUALIFIER_SIGNS.get(
        outcome_qualifier, 1
    )
    predicate_sign = PREDICATE_SIGNS.get(
        edge.get("predicate")
    )

    # Cannot infer direction from unsigned predicates
    if predicate_sign is None:
        return "unclear"

    final_sign = (
        subject_sign * predicate_sign * outcome_sign
    )

    return "increase" if final_sign > 0 else "decrease"


if __name__ == "__main__":
    results = search_edges(
        "water table position",
        "graminoid cover"
    )

    for edge in results:
        print("Edge:", edge["id"])
        print("Direction:", interpret_edge(edge))
        print("Evidence:", edge.get("original_sentence"))
        print()
