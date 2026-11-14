
from collections import Counter
from retrieval import search_edges, nodes


def analyze_evidence(intervention, outcome):
    relevant_edges = search_edges(intervention, outcome)

    if not relevant_edges:
        return {
            "status": "no_evidence",
            "direction": None,
            "edge_ids": [],
            "searched_for": (
                f"Direct edges from {intervention} "
                f"to {outcome}"
            )
        }

    directions = []

    for edge in relevant_edges:
        outcome_node = nodes[edge["object"]]
        qualifier = outcome_node.get(
            "state_or_change_qualifier"
        )

        if qualifier == "increased":
            directions.append("increase")
        elif qualifier == "decreased":
            directions.append("decrease")
        else:
            directions.append("unclear")

    counts = Counter(directions)

    if counts["increase"] and counts["decrease"]:
        direction = "mixed"
    else:
        direction = counts.most_common(1)[0][0]

    return {
        "status": "answered",
        "direction": direction,
        "edge_ids": [
            edge["id"] for edge in relevant_edges
        ],
        "searched_for": None
    }


if __name__ == "__main__":
    result = analyze_evidence(
        "water table position",
        "graminoid cover"
    )

    print(result)
