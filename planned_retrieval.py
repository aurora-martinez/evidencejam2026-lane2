
from retrieval import edges, nodes

# Example search plan for question b2.
# Later, Qwen will generate this automatically.
search_plan = {
    "interventions": [
        "peatland",
        "rewetting",
        "moss transfer",
        "moss layer transfer",
        "salt marsh",
        "water table"
    ],
    "outcomes": [
        "carbon",
        "methane",
        "CH4",
        "CO2",
        "greenhouse gas",
        "ecosystem exchange",
        "carbon sink"
    ]
}


def node_text(node):
    return " ".join([
        str(node.get("name") or ""),
        str(node.get("entity_term") or ""),
        str(node.get("embedding_text") or "")
    ]).lower()


def matches_any(text, terms):
    return sum(
        term.lower() in text
        for term in terms
    )


def planned_search(plan, top_k=20):
    results = []

    for edge in edges:
        subject = nodes[edge["subject"]]
        outcome = nodes[edge["object"]]

        subject_text = node_text(subject)
        outcome_text = node_text(outcome)

        intervention_score = matches_any(
            subject_text,
            plan["interventions"]
        )

        outcome_score = matches_any(
            outcome_text,
            plan["outcomes"]
        )

        # Prefer edges connecting an intervention to an outcome
        if intervention_score > 0 and outcome_score > 0:
            score = 10 + intervention_score + outcome_score
            results.append((score, edge))

    results.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return [edge for score, edge in results[:top_k]]


if __name__ == "__main__":

    results = planned_search(search_plan)

    print(f"Found {len(results)} relevant edges.\n")

    for edge in results:
        subject = nodes[edge["subject"]]
        outcome = nodes[edge["object"]]

        print("=" * 60)
        print("EDGE:", edge["id"])
        print("SUBJECT:", subject.get("name"))
        print("PREDICATE:", edge.get("predicate"))
        print("OUTCOME:", outcome.get("name"))
        print("CERTAINTY:", edge.get("certainty_grade"))
        print()
