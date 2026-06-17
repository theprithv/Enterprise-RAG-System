# Enterprise RAG System

![Architecture Diagram](RAG_Architecture.png)

This is a highly  secure, offline, enterprise-grade Retrieval-Augmented Generation (RAG) system built from scratch. It allows you to ingest private company PDFs and chat with an AI about them. If the AI doesn't know the answer, it automatically searches the live internet (DuckDuckGo fallback) to find it.

---

## Phase 1: Prerequisites & Folder Structure

### Step 1: Install Required Software
You must have these installed on your computer:
1. **Python 3.10+**: The programming language we will use.
2. **Docker Desktop**: This runs our heavy AI servers in isolated "containers" so they don't mess up your computer.

### Step 2: Create the Folders
Ensure you have the following directories in the project root:
```bash
mkdir data
mkdir data/documents
mkdir ingestion
mkdir retrieval
```
* **`data/documents`** is where you will drop your PDFs.
* **`ingestion`** will hold the script that reads the PDFs.
* **`retrieval`** will hold the script that answers your questions.

---

## Phase 2: The Environment & Dependencies

We use a Python Virtual Environment to keep dependencies isolated.

### Step 1: Create the Virtual Environment
Open your terminal in the `RAG_Project` folder and run:
```bash
python -m venv .venv
```
Now, activate the environment:
* **Windows (PowerShell):** `.\.venv\Scripts\Activate.ps1`
* **Mac/Linux:** `source .venv/bin/activate`

### Step 2: Install Requirements
Install the required packages by running:
```bash
pip install -r requirements.txt
```

### Step 3: Editor Configuration (VS Code)
If you are using Visual Studio Code, create a `.vscode/settings.json` file to auto-activate the environment:
```json
{
    "python.defaultInterpreterPath": ".venv/Scripts/python.exe",
    "python.terminal.activateEnvironment": true
}
```

---

## Phase 3: The Secret Vault (`.env`)

Never put passwords in your code. Copy the `.env.example` file to create a `.env` file:
```bash
cp .env.example .env
```
Fill in your actual API keys (HuggingFace, Langfuse, etc.) in the new `.env` file. The `.env` file is git-ignored and will never be pushed to GitHub.

---

## Phase 4: The Heavy Machinery (Docker)

We run the AI Brain (vLLM) and the Vector Database (Milvus) using Docker.

### Start the Servers
Run this command in your terminal. It will download the AI models and start the database:
```bash
docker compose up -d
```

---

## Phase 5: Usage

### 1. Load your data
Drop a PDF into `data/documents`, and run the ingestion script:
```bash
python ingestion/ingest.py
```
*(This uses IBM Docling to read the PDF and Semantic Chunker to store it in Milvus).*

### 2. Talk to your data
Run the retrieval script:
```bash
python retrieval/retrieve.py
```
*(This uses HuggingFace embeddings to search the database, vLLM to generate the answer, and DuckDuckGo for live-internet fallback if the answer is missing).*

Enjoy your Enterprise AI System!

---

## Phase 6: LLMOps Evaluation & DevSecOps

To prove the RAG agent works and stays working, we built an automated evaluation suite and a CI/CD pipeline using **MLflow** and **GitHub Actions**.

### 1. The Automated Evaluation Suite
We created a predefined dataset of 15 test questions in `eval/eval_dataset.yaml` spread across four categories:
* **Grounding:** Checks if answers correctly rely on the ingested PDF context.
* **Retrieval Quality:** Checks if the correct chunks are fetched.
* **Refusal:** Checks if the AI correctly says "I don't know" for unanswerable questions.
* **Injection:** A security check to ensure the AI ignores malicious prompt injection.

We evaluate the agent using an "LLM-as-a-judge" approach. MLflow uses a local LiteLLM proxy pointing to our vLLM container to grade the RAG system's responses.

**To run the evaluation:**
```bash
python eval/run_eval.py
```
This generates a `baseline.json` with the scorecard.

### 2. The Baseline & Judge Verification
Our `baseline.json` establishes the "known-good" score for our model. We validated the judge by hand-labeling 8 test cases and confirming the LLM judge agreed with our manual assessments 100% of the time, proving the judge is trustworthy and not hallucinating scores.

### 3. CI/CD Gate (pr-eval.yml)
Every time a new Pull Request is opened, GitHub Actions boots up the local environment, runs the MLflow evaluation suite, and compares the new scores to `baseline.json`. If *any* metric drops below the baseline, the PR is automatically blocked from merging.

### 4. AIOps Monitoring (nightly.yml)
A scheduled nightly workflow re-runs the evaluation to detect quality or latency drift over time. If a significant drop is detected, the workflow uses the GitHub CLI to automatically open a bug report issue.

### 5. DevSecOps (security.yml)
To ensure no credentials are ever leaked, we run `gitleaks` (secret scanning), CodeQL (vulnerability scanning), and Dependabot on every push. The prompt injection cases in our dataset act as our final LLM security gate.
