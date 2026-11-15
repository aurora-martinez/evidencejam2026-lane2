
import json
from pathlib import Path

# Load graph
BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "causal_graph.json", encoding="utf-8") as file:
    graph = json.load(file)

nodes = {node["id"]: node for node in graph["nodes"]}
edges = graph["edges"]


def search_nodes(keyword):
    """Search nodes by their names and descriptions."""

    matches = []

    for node in nodes.values():

        text = " ".join([
            str(node.get("name") or ""),
            str(node.get("entity_term") or ""),
            str(node.get("measured_attribute") or ""),
            str(node.get("embedding_text") or "")
        ]).lower()

        if keyword.lower() in text:
            matches.append(node)

    return matches


def search_edges(intervention, outcome):
    """Find direct edges connecting matching concepts."""

    intervention_nodes = search_nodes(intervention)
    outcome_nodes = search_nodes(outcome)

    intervention_ids = {
        node["id"] for node in intervention_nodes
    }

    outcome_ids = {
        node["id"] for node in outcome_nodes
    }

    matching_edges = []

    for edge in edges:

        if (edge["subject"] in intervention_ids
                and edge["object"] in outcome_ids):

            matching_edges.append(edge)

    return matching_edges


def display_results(results):

    print(f"\nFound {len(results)} matching edges")

    for edge in results[:10]:

        subject = nodes[edge["subject"]]
        outcome = nodes[edge["object"]]

        print("\n" + "=" * 60)

        print("EDGE:", edge["id"])

        print("SUBJECT:", subject.get("name"))
        print("OUTCOME:", outcome.get("name"))

        print("PREDICATE:", edge.get("predicate"))

        print(
            "DIRECTION QUALIFIER:",
            outcome.get("state_or_change_qualifier")
        )

        print("CERTAINTY:", edge.get("certainty_grade"))

        print("EVIDENCE:", edge.get("original_sentence"))


if __name__ == "__main__":

    intervention = input("Enter intervention: ").strip()
    outcome = input("Enter outcome: ").strip()

    results = search_edges(intervention, outcome)

    display_results(results)

