
import json
from query_planner import plan_question


def interpret_question(question):
    """
    Interpret any ecological question.

    Returns structured search terms for
    Aurora's graph retrieval algorithm.
    """

    if not isinstance(question, str):
        raise TypeError("Question must be a string.")

    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    # No question ID: works with unseen questions
    plan = plan_question(question)

    return {
    "interventions": plan["interventions"],
    "outcomes": plan["outcomes"],
}


if __name__ == "__main__":

    question = input(
        "Enter an ecological question: "
    )

    result = interpret_question(question)

    print("\nSEARCH PLAN:\n")
    print(json.dumps(result, indent=2))
