
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


import re

# Common words that don't help identify ecological evidence
STOP_WORDS = {
    "the", "and", "for", "with", "from", "that",
    "this", "what", "which", "where", "when",
    "does", "do", "how", "are", "into", "they",
    "their", "have", "has", "such", "also",
    "after", "each", "versus", "about", "can"
}


def extract_keywords(question):
    # Convert the question into lowercase words
    words = re.findall(r"[a-zA-Z]+", question.lower())

    # Remove common words and very short words
    keywords = [
        word for word in words
        if word not in STOP_WORDS and len(word) > 3
    ]

    return list(dict.fromkeys(keywords))


def search_question(question, top_k=15):
    keywords = extract_keywords(question)

    results = []

    for edge in edges:
        subject = nodes[edge["subject"]]
        outcome = nodes[edge["object"]]

        # Search the node descriptions
        subject_text = " ".join([
            str(subject.get("name") or ""),
            str(subject.get("entity_term") or ""),
            str(subject.get("embedding_text") or "")
        ]).lower()

        outcome_text = " ".join([
            str(outcome.get("name") or ""),
            str(outcome.get("entity_term") or ""),
            str(outcome.get("embedding_text") or "")
        ]).lower()

        # Count matching keywords
        subject_score = sum(
            word in subject_text for word in keywords
        )

        outcome_score = sum(
            word in outcome_text for word in keywords
        )

        # Reward edges that match both sides
        score = (
            subject_score
            + outcome_score
            + 2 * min(subject_score, outcome_score)
        )

        if score > 0:
            results.append((score, edge))

    # Sort from most relevant to least relevant
    results.sort(key=lambda item: item[0], reverse=True)

    return [edge for score, edge in results[:top_k]]

