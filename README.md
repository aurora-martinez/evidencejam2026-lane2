# YC Startup — Evidence Jam 2026

**Causapalooza | Lane 2: Causal Reasoning**

An AI-powered ecological evidence retrieval system that uses Qwen and causal graph analysis to answer scientific questions with traceable supporting evidence.

## Quick Start

**Run the entire pipeline with one command:**

```bash
python generate_answers.py questions.yaml answers.json
```

## Setup

**1. Install dependencies**

```bash
pip install python-dotenv pyyaml
```

**2. Configure environment variables**

Create a `.env` file using the credentials provided by Evidence Jam:

```env
BRINE_URL=your_endpoint
BRINE_API_KEY=your_api_key
BRINE_MODEL=Qwen3.8-27B
```

**3. Run the program**

```bash
python generate_answers.py questions.yaml answers.json
```

The program processes the questions in `questions.yaml` and generates `answers.json` in the required submission format.

## Project Files

| File | Description |
|---|---|
| `generate_answers.py` | Main execution script |
| `query_planner.py` | Qwen question interpretation |
| `search.py` | Causal graph evidence retrieval |
| `retrieval.py` | Graph loading and access |
| `answer_generator.py` | Scientific answer generation |
| `causal_graph.json` | Ecological evidence graph |
| `questions.yaml` | Input questions |
| `answers.json` | Generated answers |

## Output

Each answer includes:

- An evidence-based scientific response
- Direction and evidence certainty
- A heuristic confidence score between 0 and 1
- Supporting causal graph edge IDs
- Search details when no direct evidence is retrieved

## Notes

- Uses the Evidence Jam BRINE endpoint for Qwen inference.
- Supports compatible question YAML files, including unseen evaluation questions.
- Uses cached query interpretations to reduce repeated model calls.
- Confidence scores are heuristic, not calibrated probabilities.
- Retrieval primarily searches direct graph relationships.

**Team:** YC Startup  
**Event:** Evidence Jam 2026 — Causapalooza
