# Labeling Portal Mock Service

This repository contains a FastAPI-based mock service that demonstrates how an
active-learning aware video labeling portal could orchestrate task creation,
assignment, consensus building, annotator reliability tracking, and model
retraining triggers.

The service is intentionally lightweight and uses in-memory stores so that the
complete annotation life cycle can be explored without additional
infrastructure. It reflects the product concepts outlined in the enhancement
brief: model-centric prioritization, annotator oriented flows, active learning
loops, and ongoing quality management.

## Features

- **Prediction ingestion & task generation** – Capture model predictions,
  persist uncertainty scores, and spin up labeling tasks with dynamic
  prioritization.
- **Reliability-aware auto-assignment** – Route work to annotators based on
  their reliability scores and task difficulty.
- **Annotation capture & voting** – Collect multiple captions per task, record
  review votes, and promote high-agreement samples for consensus.
- **Consensus synthesis** – Produce a single caption that represents semantic
  agreement, along with a mock LLM confidence score.
- **Annotator dashboard metrics** – Expose throughput, reliability, and
  disagreement rates to power the management portal.
- **Retraining hooks & evaluator** – Emit webhook-style payloads for mini-batch
  retraining and simulate comparisons between the original and retrained model.

## Getting Started

Install dependencies and start the development server:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Visit `http://localhost:8000/docs` to interact with the automatically generated
OpenAPI documentation.

## Example Flow

1. **Ingest a model prediction** – `POST /predictions`
2. **Create a labeling task** – `POST /tasks`
3. **Auto-assign to an annotator** – `POST /tasks/assign`
4. **Submit annotations** – `POST /tasks/{task_id}/annotations`
5. **Record quality votes** – `POST /tasks/{task_id}/votes`
6. **Finalize consensus** – `POST /tasks/{task_id}/consensus`
7. **Inspect annotator metrics** – `GET /annotators/{annotator_id}/metrics`
8. **Trigger retraining** – `POST /retraining/trigger`
9. **Review evaluator comparison** – `GET /tasks/{task_id}/evaluation`

Because the service keeps all state in memory, restart the process whenever you
want a clean slate.
