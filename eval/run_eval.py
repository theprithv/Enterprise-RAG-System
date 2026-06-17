import os
import yaml
import pandas as pd
import mlflow
import sys
import json

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from retrieval.retrieve import predict_fn
from eval.scorers import is_refusal_scorer, is_injection_ignored_scorer, vllm_correctness_scorer, vllm_professionalism_scorer

# Configure LiteLLM proxy (Not strictly needed anymore since we use vLLM directly for grading, but safe to keep)
os.environ["OPENAI_API_BASE"] = "http://127.0.0.1:4000"
os.environ["OPENAI_API_KEY"] = "sk-dummy"

def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("RAG_Agent_Evaluation")
    
    print("Loading dataset...")
    with open("eval/eval_dataset.yaml", "r") as f:
        data = yaml.safe_load(f)["cases"]
    
    df = pd.DataFrame(data)
    # MLflow evaluate requires 'inputs' and 'ground_truth'
    df.rename(columns={"question": "inputs", "expected_answer": "ground_truth"}, inplace=True)
    
    print("Initializing Robust vLLM Judges...")
    
    # Our new robust custom judges
    correctness_metric = mlflow.metrics.make_metric(eval_fn=vllm_correctness_scorer, name="vllm_correctness", greater_is_better=True)
    professionalism_metric = mlflow.metrics.make_metric(eval_fn=vllm_professionalism_scorer, name="vllm_professionalism", greater_is_better=True)
    
    # Custom deterministic metrics
    refusal_metric = mlflow.metrics.make_metric(eval_fn=is_refusal_scorer, name="is_refusal", greater_is_better=True)
    injection_metric = mlflow.metrics.make_metric(eval_fn=is_injection_ignored_scorer, name="is_injection_ignored", greater_is_better=True)
    
    print("Running Evaluation via MLflow...")
    with mlflow.start_run(run_name="pr_eval_run"):
        results = mlflow.evaluate(
            model=predict_fn,
            data=df,
            targets="ground_truth",
            model_type="question-answering",
            evaluators="default",
            extra_metrics=[
                correctness_metric,
                professionalism_metric,
                refusal_metric,
                injection_metric
            ]
        )
        
        print("\nEvaluation Complete!")
        metrics = results.metrics
        print(metrics)
        
        # Save baseline
        baseline_path = os.path.join(os.path.dirname(__file__), "..", "baseline.json")
        with open(baseline_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Saved baseline scores to {baseline_path}")

if __name__ == "__main__":
    main()
