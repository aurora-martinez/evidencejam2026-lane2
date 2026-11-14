
import json
from pathlib import Path

graph_path = Path(__file__).resolve().parent / "causal_graph.json"

with open(graph_path, encoding="utf-8") as file:
    graph = json.load(file)

nodes = {node["id"]: node for node in graph["nodes"]}
edges = graph["edges"]


def inspect_edge(edge):
    subject = nodes[edge["subject"]]
    outcome = nodes[edge["object"]]

    print("\n" + "=" * 60)

    print("EDGE ID:", edge["id"])

    print("\nINTERVENTION / SUBJECT:")
    print("Name:", subject.get("name"))
    print("Type:", subject.get("entity_type"))
    print("Term:", subject.get("entity_term"))

    print("\nOUTCOME / OBJECT:")
    print("Name:", outcome.get("name"))
    print("Type:", outcome.get("entity_type"))
    print("Attribute:", outcome.get("measured_attribute"))
    print("Qualifier:", outcome.get("state_or_change_qualifier"))

    print("\nRELATIONSHIP:")
    print("Predicate:", edge.get("predicate"))
    print("Causal language:", edge.get("causal_language"))
    print("Negated:", edge.get("negated"))

    print("\nEVIDENCE:")
    print("Sentence:", edge.get("original_sentence"))
    print("Certainty:", edge.get("certainty_grade"))
    print("Source:", edge.get("source_document"))

    print("=" * 60)


for edge in edges[:5]:
    inspect_edge(edge)
