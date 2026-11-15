
import json
from collections import defaultdict

with open("answers.json", encoding="utf-8") as file:
    data = json.load(file)

with open("causal_graph.json", encoding="utf-8") as file:
    graph = json.load(file)

edge_lookup = {
    edge["id"]: edge
    for edge in graph["edges"]
}

for answer in data["answers"]:
    print("\n" + "=" * 65)
    print("QUESTION:", answer["id"])

    sources = defaultdict(list)

    for edge_id in answer.get("edge_ids", []):
        edge = edge_lookup.get(edge_id)

        if edge is None:
            continue

        source_id = edge.get("source_document", "Unknown source")
        sources[source_id].append(edge)

    print("TOTAL EDGES:", sum(len(v) for v in sources.values()))
    print("UNIQUE SOURCES:", len(sources))

    for source_id, source_edges in sources.items():
        print("\nSOURCE:", source_id)
        print("NUMBER OF EDGES:", len(source_edges))

        for edge in source_edges:
            print("INTERVENTION:", edge.get("subject"))
            print("OUTCOME:", edge.get("object"))
            print("PREDICATE:", edge.get("predicate"))
            print("NEGATED:", edge.get("negated"))
            print("EVIDENCE:", edge.get("original_sentence"))
