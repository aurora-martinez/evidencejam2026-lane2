# YC Startup — Evidence Jam 2026

**Causapalooza | Lane 2: Causal Reasoning**

An AI-powered ecological evidence retrieval system that uses Qwen and causal graph analysis to answer scientific research questions with traceable supporting evidence.

## Quick Start

Install dependencies:

```bash
pip install requests python-dotenv pyyaml
```

Create a `.env` file in the project root:

```env
BRINE_URL=your_endpoint
BRINE_API_KEY=your_api_key
BRINE_MODEL=your_model_name
```

Run the complete pipeline:

```bash
python3 generate_answers.py questions.yaml answers.json
```

The program reads research questions from `questions.yaml` and saves generated results to `answers.json`.

## How It Works

The system processes each research question through the following stages:

1. **Question Planning:** Qwen interprets the research question and identifies relevant interventions and outcomes.
2. **Search Planning:** Creates a structured search plan based on the identified concepts.
3. **Graph Retrieval:** Searches the ecological causal graph for relevant relationships and supporting evidence.
4. **Evidence Organization:** Groups retrieved graph relationships by source publication and collects available publication metadata.
5. **Answer Generation:** Qwen synthesizes the retrieved findings into a scientific response.
6. **Fallback Generation:** If Qwen fails or returns an unusable response, a deterministic report is generated from the retrieved evidence.
7. **Output:** Saves answers, confidence scores, evidence direction, and supporting graph edge IDs in JSON format.

## Project Files

| File | Description |
|---|---|
| `generate_answers.py` | Main execution script; coordinates the full pipeline and generates output |
| `query_planner.py` | Interprets research questions and creates structured search plans using Qwen |
| `search.py` | Coordinates graph searches using the generated search plans |
| `retrieval.py` | Loads and accesses the ecological causal graph |
| `answer_generator.py` | Prepares evidence and generates scientific answers using Qwen |
| `causal_graph.json` | Structured ecological evidence graph containing nodes, relationships, and source publications |
| `questions.yaml` | Input research questions |
| `answers.json` | Generated answers and supporting evidence references |

**Team:** YC Startup  
**Event:** Evidence Jam 2026 — Causapalooza
