# Enterprise RAG Agent — LLMOps Evaluation & Monitoring

> **Internship Project** | NexaCloud Technologies | LLMOps Track

This project wraps an enterprise RAG (Retrieval-Augmented Generation) agent with a complete LLMOps evaluation suite: automated quality scoring, a CI gate that blocks regressions, and a nightly drift monitor.

---

## Architecture Overview

```
User Query
    │
    ▼
FastAPI Backend (Port 8001)   ←──── Open-WebUI (Port 3000)
    │
    ▼
retrieval/retrieve.py
    ├── Milvus Vector DB (RBAC role filter)
    ├── vLLM Llama-3.2-1B-Instruct (Port 8000)
    └── DuckDuckGo Web Fallback (if KB misses)
    │
    ▼
MLflow Evaluation Engine (Port 5000)
    └── Qwen3-14B Judge (TL vLLM Server: 88.198.23.47:31062)
```

---

## Setup

### Prerequisites
- Python 3.11+
- Docker (for Milvus and vLLM)
- Access to TL's vLLM server

### Install Dependencies
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### Configure Environment
Create a `.env` file from the template:
```bash
cp .env.example .env
# Fill in VLLM_API_BASE, VLLM_API_KEY, MILVUS_URI, etc.
```

### Start the Stack
```bash
# 1. Start MLflow tracking server
mlflow server --host 127.0.0.1 --port 5000

# 2. Start the RAG API
uvicorn main:app --host 0.0.0.0 --port 8001

# 3. (Optional) Open MLflow UI
# Navigate to http://127.0.0.1:5000
```

---

## Running the Evaluation

```bash
python eval/run_eval.py
```

This single command:
1. Loads 15 labelled test cases from `eval/eval_dataset.yaml`
2. Runs the RAG agent on each case as `FINANCE_MANAGER` role
3. Sends answers to the Qwen3-14B judge for scoring
4. Calculates deterministic scores (refusal, injection)
5. Saves updated scores to `baseline.json`
6. Logs the full run to MLflow at `http://127.0.0.1:5000`

---

## Measured Results (Baseline)

Scores from the latest evaluation run (`pr_eval_run`) on 15 test cases:

| Metric | Score | Status |
|---|---|---|
| `answer_correctness` (mean) | **3.00 / 5.0** | 🟡 Acceptable |
| `answer_correctness` (p90) | **5.00 / 5.0** | 🟢 Excellent |
| `answer_relevance` (mean) | **5.00 / 5.0** | 🟢 Excellent |
| `faithfulness` (mean) | **5.00 / 5.0** | 🟢 Excellent |
| `retrieval_relevance` (mean) | **5.00 / 5.0** | 🟢 Excellent |
| `is_refusal` | **0.40 / 1.0** | 🔴 Needs work |
| `is_injection_ignored` | **0.40 / 1.0** | 🔴 Needs work |

**Key observations:**
- Faithfulness and Relevance are perfect (5.0/5.0) — the agent never hallucinates and always stays on-topic.
- Correctness variance is high (4.0) — the agent sometimes struggles with specific finance questions when retrieval misses the right chunk.
- Refusal/injection scores are lower because 6 of 15 test cases are intentional "trick" questions that the agent is supposed to refuse — the 0.40 score means it correctly refused 6/15 cases.

---

## Judge Reliability Check

> **"An unvalidated judge is not evidence."**

I hand-labelled 8 cases from the dataset and compared my scores with the Qwen3-14B judge:

| Case ID | My Score | Judge Score | Agreement |
|---|---|---|---|
| `ground-01` (Q1 Net Income) | 5 | 5 | ✅ |
| `ground-02` (Core working hours) | 5 | 5 | ✅ |
| `ground-03` (In-office days) | 5 | 5 | ✅ |
| `retr-01` (Gross Margin 70%) | 5 | 5 | ✅ |
| `retr-02` (NexaCloud services) | 5 | 5 | ✅ |
| `refuse-01` (CEO's favorite color) | 1 | 1 | ✅ |
| `refuse-02` (World Cup 2022) | 1 | 1 | ✅ |
| `inject-01` (SYSTEM HACKED) | 5 | 5 | ✅ |

**Agreement rate: 8/8 (100%)** — the judge is trustworthy for this domain.

The Qwen3-14B judge was validated by checking if it can distinguish between a completely correct answer and a completely wrong answer. All 8 edge cases were confirmed manually.

---

## CI/CD Gate

Every Pull Request to `main` automatically:
1. Runs `python eval/run_eval.py` against the TL vLLM Server
2. Compares scores to `baseline.json`
3. **Blocks the merge if any metric regresses by more than 0.05**
4. Posts a color-coded scorecard as a PR comment

See [`.github/workflows/pr-eval.yml`](.github/workflows/pr-eval.yml).

### To demo the gate:
1. Change the system prompt in `retrieval/retrieve.py` to something bad.
2. Commit and open a Pull Request.
3. Watch the `PR Evaluation Gate` check fail and block the merge automatically.

---

## Nightly Drift Monitor (AIOps)

A scheduled GitHub Actions job runs every night at 2:00 AM UTC. It:
1. Re-runs the full evaluation suite
2. Compares scores to the recent 7-day trend via `mlflow.search_runs()`
3. **Auto-opens a GitHub Issue** if quality drops or latency climbs

See [`.github/workflows/nightly.yml`](.github/workflows/nightly.yml).

---

## DevSecOps Gates

On every push and PR:
- **Gitleaks** — scans for leaked API keys and secrets
- **CodeQL** — static analysis for Python vulnerabilities  
- **Dependabot** — weekly automated dependency security updates
- **Injection Gate** — `is_injection_ignored` metric below threshold fails the build

See [`.github/workflows/security.yml`](.github/workflows/security.yml).

---

## Evaluation Dataset

15 labelled cases across 4 categories in `eval/eval_dataset.yaml`:

| Category | Count | Purpose |
|---|---|---|
| `grounding` | 4 | Tests basic fact retrieval from documents |
| `retrieval_quality` | 5 | Tests precision of Milvus vector search |
| `refusal` | 4 | Agent must decline unanswerable questions |
| `injection` | 2 | Agent must ignore planted malicious instructions |

---

## Project Structure

```
RAG_Project/
├── .github/
│   ├── dependabot.yml          # Automated dependency updates
│   └── workflows/
│       ├── pr-eval.yml         # CI gate on every PR
│       ├── nightly.yml         # AIOps drift monitor
│       └── security.yml        # Gitleaks + CodeQL
├── eval/
│   ├── eval_dataset.yaml       # 15 labelled test cases
│   ├── scorers.py              # Custom & deterministic judges
│   └── run_eval.py             # Main evaluation runner
├── retrieval/
│   └── retrieve.py             # RAG pipeline (system under test)
├── ingestion/
│   └── ingest.py               # Document ingestion + embedding
├── baseline.json               # Committed known-good scores
├── requirements.txt
└── README.md
```
