
import re
from retrieval import edges, nodes


def node_text(node):
    return " ".join([
        str(node.get("name") or ""),
        str(node.get("entity_term") or ""),
        str(node.get("measured_attribute") or ""),
        str(node.get("embedding_text") or ""),
    ]).lower()


# Built once, not once per edge
NODE_TEXT = {nid: node_text(n) for nid, n in nodes.items()}


def compile_terms(terms):
    compiled = []

    for group in terms:

        # Allow either a single string or a list of strings
        if isinstance(group, str):
            group = [group]

        compiled.append([
            (t, re.compile(
                r"\b" + re.escape(t.lower()) + r"(?:s|es)?\b"
            ))
            for t in group
        ])

    return compiled


def term_hits(compiled):
    """Find keyword groups where ALL terms appear in a node."""
    out = {}

    for nid, text in NODE_TEXT.items():
        found = []

        for group in compiled:
            if all(pat.search(text) for t, pat in group):
                found.append(" AND ".join(t for t, pat in group))

        if found:
            out[nid] = found

    return out


def planned_search(plan):
    iv_hits = term_hits(compile_terms(plan["interventions"]))
    oc_hits = term_hits(compile_terms(plan["outcomes"]))

    results = []

    for edge in edges:
        a = iv_hits.get(edge["subject"])
        b = oc_hits.get(edge["object"])

        if a and b:
            results.append({
                "edge": edge,
                "score": len(a) + len(b),
                "intervention_terms": a,
                "outcome_terms": b,
            })

    results.sort(key=lambda r: r["score"], reverse=True)

    return results


if __name__ == "__main__":

    plan = {
        "interventions": [
            "planting-stock traits",
            ["seedling", "size"],
            "nursery fertilization",
            ["mycorrhizal", "inoculation"]
        ],

        "outcomes": [
            "survival",
            "growth"
        ],
    }

    results = planned_search(plan)

    print(f"Found {len(results)} edges\n")

    for r in results[:20]:
        e = r["edge"]

        print("=" * 60)
        print("EDGE:", e["id"])
        print("CAUSE:", nodes[e["subject"]].get("name"),
              r["intervention_terms"])
        print("EFFECT:", nodes[e["object"]].get("name"),
              r["outcome_terms"])
        print("predicate:", e.get("predicate"),
              "| negated:", e.get("negated"))

    if not results:
        print("no_evidence. searched for:", plan)