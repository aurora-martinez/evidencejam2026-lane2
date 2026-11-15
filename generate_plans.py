
import json
import time
import yaml

from pathlib import Path
from query_planner import plan_question

BASE_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = BASE_DIR / "questions.yaml"
OUTPUT_FILE = BASE_DIR / "all_search_plans.json"


def generate_all_plans():

    # Read the competition questions
    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        data = yaml.safe_load(file)

    questions = data["questions"]

    print(f"Loaded {len(questions)} questions.")

    all_plans = {}

    for index, item in enumerate(questions, start=1):

        question_id = item["id"]
        question_text = item["question"]
        question_kind = item.get("kind", "unknown")

        print(
            f"\n[{index}/{len(questions)}] "
            f"Processing {question_id} ({question_kind})",
            flush=True
        )

        try:
            plan = plan_question(
                question_text,
                question_id=question_id
            )

            all_plans[question_id] = {
                "kind": question_kind,
                "question": question_text,
                "plan": plan
            }

            # Save after every successful question
            with open(
                OUTPUT_FILE,
                "w",
                encoding="utf-8"
            ) as file:
                json.dump(
                    all_plans,
                    file,
                    indent=2,
                    ensure_ascii=False
                )

            print(
                f"Saved plan for {question_id}",
                flush=True
            )

        except Exception as error:
            print(
                f"Failed on {question_id}: {error}",
                flush=True
            )

        # Small pause between API requests
        time.sleep(1)

    print(
        f"\nCompleted {len(all_plans)} "
        f"of {len(questions)} questions."
    )

    print(f"Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_all_plans()
