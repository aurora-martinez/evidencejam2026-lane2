
from retrieval import search_question, nodes

question = (
    "Which restoration actions, such as peatland rewetting, "
    "moss layer transfer or salt marsh realignment, "
    "turn degraded sites back into carbon sinks, "
    "and what happens to methane emissions?"
)

results = search_question(question, top_k=10)

print("QUESTION:", question)
print("\nFOUND:", len(results), "edges")

for i, edge in enumerate(results, start=1):
    subject = nodes[edge["subject"]]
    outcome = nodes[edge["object"]]

    print("\n" + "=" * 60)
    print("RESULT", i)
    print("EDGE ID:", edge["id"])
    print("SUBJECT:", subject.get("name"))
    print("PREDICATE:", edge.get("predicate"))
    print("OUTCOME:", outcome.get("name"))
    print("CERTAINTY:", edge.get("certainty_grade"))
    print("EVIDENCE:", edge.get("original_sentence"))
