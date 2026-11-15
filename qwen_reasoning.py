
import os
import json
import time
import requests
from retrieval import search_question, nodes
import os

required_variables = [
    "BRINE_URL",
    "BRINE_MODEL",
    "BRINE_API_KEY"
]

missing = [
    variable for variable in required_variables
    if not os.environ.get(variable)
]

if missing:
    raise EnvironmentError(
        f"Missing environment variables: {', '.join(missing)}"
    )

def reason_about_question(question):

    # Step 1: Retrieve candidate evidence
    print("Searching the causal graph...", flush=True)

    edges = search_question(question, top_k=5)

    print(f"Found {len(edges)} candidate edges.", flush=True)

    print("\nRETRIEVED EDGE IDS:")

    for edge in edges:
        print(edge["id"])

    evidence = []

    for edge in edges:
        subject = nodes[edge["subject"]]
        outcome = nodes[edge["object"]]

        evidence.append({
            "edge_id": edge["id"],
            "subject": subject.get("name"),
            "subject_qualifier": subject.get(
                "state_or_change_qualifier"
            ),
            "predicate": edge.get("predicate"),
            "outcome": outcome.get("name"),
            "outcome_qualifier": outcome.get(
                "state_or_change_qualifier"
            ),
            "causal_language": edge.get("causal_language"),
            "negated": edge.get("negated"),
            "certainty": edge.get("certainty_grade"),
            "source": edge.get("original_sentence")
        })

    # Step 2: Prepare evidence for Qwen
    evidence_text = json.dumps(evidence, indent=2)

    prompt = f"""
You are a scientific causal reasoning assistant.

QUESTION:
{question}

RETRIEVED EVIDENCE:
{evidence_text}

TASK:
1. Identify which edges directly help answer the question.
2. Explain what each relevant edge establishes.
3. Distinguish causation from association.
4. Identify conflicting or uncertain evidence.
5. Do not invent evidence or edge IDs.
6. If the evidence is insufficient, explicitly say so.

Return a concise answer with supporting edge IDs.
"""

    # Step 3: Send request directly to Qwen
    print("Sending evidence to Qwen...", flush=True)

    url = os.environ["BRINE_URL"].rstrip("/") + "/chat/completions"

    headers = {
        "Authorization": f"Bearer {os.environ['BRINE_API_KEY']}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": os.environ["BRINE_MODEL"],
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an evidence-grounded ecological "
                    "reasoning assistant."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 2500
    }

    start_time = time.perf_counter()

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    elapsed_time = time.perf_counter() - start_time

    print("Qwen response received!", flush=True)
    print(f"Response time: {elapsed_time:.2f} seconds")

    # Step 4: Process Qwen response
    result = response.json()

    choice = result["choices"][0]
    message = choice["message"]

    print("\n--- QWEN DEBUG INFORMATION ---")
    print("Finish reason:", choice.get("finish_reason"))
    print("Content:", repr(message.get("content")))

    print(
        "Reasoning content:",
        repr(message.get("reasoning_content"))[:1000]
    )

    print("Token usage:", result.get("usage"))

    print("------------------------------\n")

    # Step 5: Return final answer
    return message.get("content")


if __name__ == "__main__":

    question = (
        "Which restoration actions, such as peatland rewetting, "
        "moss layer transfer or salt marsh realignment, "
        "turn degraded sites back into carbon sinks, "
        "and what happens to methane emissions?"
    )

    answer = reason_about_question(question)

    print("\nQWEN ANSWER:\n")

    if answer:
        print(answer)
    else:
        print("No final text answer was returned.")
