import os
import warnings
import logging

# Suppress all the messy FutureWarnings, RuntimeWarnings, and MLflow INFO/WARNING spam
warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

import yaml
import pandas as pd
import mlflow
import sys
import json

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from retrieval.retrieve import predict_fn, get_rag_response
from eval.scorers import is_refusal_scorer, is_injection_ignored_scorer, vllm_correctness_scorer, vllm_professionalism_scorer
from mlflow.metrics.genai import answer_correctness, answer_relevance, faithfulness, relevance

# Configure MLflow Judges to connect DIRECTLY to TL's Enterprise Server
# We bypass the local LiteLLM proxy because the TL's server is already OpenAI-compatible,
# and the staff's `vllm/` proxy prefix causes a Python crash on Windows.
os.environ["OPENAI_BASE_URL"] = "http://88.198.23.47:31062/v1"
os.environ["OPENAI_API_KEY"] = "sk-vllm-psMqNXftVckB5M3mRzQp37dnY"

def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("RAG_Agent_Evaluation")
    
    print("Loading dataset...")
    with open("eval/eval_dataset.yaml", "r") as f:
        data = yaml.safe_load(f)["cases"]
    
    df = pd.DataFrame(data)
    # MLflow evaluate requires 'inputs' and 'ground_truth'
    # For the staff's schema, map 'query' to 'inputs', and join 'expected_facts' into a single 'ground_truth' string
    df.rename(columns={"query": "inputs"}, inplace=True)
    df["ground_truth"] = df["expected_facts"].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
    
    print("Initializing MLflow Built-in GenAI Judges via TL Server...")
    
    # Point the judges exclusively to the TL's 14B model directly
    judge_model = "openai:/qwen3-14b"
    
    proxy_url = "http://88.198.23.47:31062/v1/chat/completions"
    
    correctness_metric = answer_correctness(model=judge_model, proxy_url=proxy_url)
    relevance_metric = answer_relevance(model=judge_model, proxy_url=proxy_url)
    faithfulness_metric = faithfulness(model=judge_model, proxy_url=proxy_url)
    retrieval_relevance_metric = relevance(model=judge_model, proxy_url=proxy_url)
    
    # Custom deterministic metrics
    refusal_metric = mlflow.metrics.make_metric(eval_fn=is_refusal_scorer, name="is_refusal", greater_is_better=True)
    injection_metric = mlflow.metrics.make_metric(eval_fn=is_injection_ignored_scorer, name="is_injection_ignored", greater_is_better=True)
    
    # Custom domain-specific LLM judges (Task 4 requirement: make_judge)
    # These use our local vLLM (Llama-1B) to grade domain-specific rules
    correctness_custom_metric = mlflow.metrics.make_metric(eval_fn=vllm_correctness_scorer, name="vllm_correctness", greater_is_better=True)
    professionalism_metric = mlflow.metrics.make_metric(eval_fn=vllm_professionalism_scorer, name="vllm_professionalism", greater_is_better=True)
    
    print("Running Evaluation via MLflow...")
    with mlflow.start_run(run_name="pr_eval_run"):
        
        print("Generating responses and retrieving contexts (Tracing Enabled)...")
        predictions = []
        contexts = []
        for query in df["inputs"]:
            ans, ctx = get_rag_response(query, "FINANCE_MANAGER", return_context=True)
            predictions.append(ans)
            contexts.append(ctx)
            
        df["predictions"] = predictions
        df["context"] = contexts

        results = mlflow.evaluate(
            data=df,
            targets="ground_truth",
            predictions="predictions",
            model_type="question-answering",
            evaluators="default",
            extra_metrics=[
                correctness_metric,
                relevance_metric,
                faithfulness_metric,
                retrieval_relevance_metric,
                refusal_metric,
                injection_metric,
                correctness_custom_metric,
                professionalism_metric
            ]
        )
        
        print("\n" + "="*70)
        print(" 🏆 Evaluation Complete! - Final Scorecard")
        print("="*70)
        metrics = results.metrics
        
        for key, value in metrics.items():
            try:
                # Format to 4 decimal places for readability
                val_str = f"{float(value):.4f}"
            except (ValueError, TypeError):
                val_str = str(value)
                
            print(f"  {key:<50} |  {val_str}")
        print("="*70)
        
        # Save baseline
        baseline_path = os.path.join(os.path.dirname(__file__), "..", "baseline.json")
        with open(baseline_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Saved baseline scores to {baseline_path}")

if __name__ == "__main__":
    main()
