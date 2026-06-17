# ============================================================
# RETRIEVAL PIPELINE
# Runs every time a user asks a question.
# Loads the saved vector store, searches, and answers.
# All settings are read from the .env file — no hard-coded values.
# ============================================================

import sys
import os
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()  # Reads all values from .env file automatically

# pyrefly: ignore [missing-import]
import mlflow
mlflow.autolog()  # Automatically trace LangChain, OpenAI, and Python logic for LLMOps project

# We now use the Langfuse wrapper to automatically track telemetry!
# pyrefly: ignore [missing-import]
from langfuse.openai import OpenAI
# pyrefly: ignore [missing-import]
from langchain_huggingface import HuggingFaceEmbeddings
# pyrefly: ignore [missing-import]
from pymilvus import MilvusClient
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document

# ── Load config from .env ──────────────────────────────────
MILVUS_URI       = os.getenv("MILVUS_URI",        "http://localhost:19530")
MILVUS_FALLBACK  = os.getenv("MILVUS_FALLBACK_DB", "./milvus_rag.db")
COLLECTION_NAME  = os.getenv("MILVUS_COLLECTION",  "rag_knowledge")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL",    "all-MiniLM-L6-v2")
VLLM_MODEL       = os.getenv("VLLM_MODEL",         "meta-llama/Llama-3.2-1B-Instruct")
VLLM_API_BASE    = os.getenv("VLLM_API_BASE",      "http://localhost:8000/v1")
VLLM_API_KEY     = os.getenv("VLLM_API_KEY",       "sk-dummy")
BRAVE_API_KEY    = os.getenv("BRAVE_API_KEY",      "")

# ──────────────────────────────────────────────────────────


def get_milvus_client():
    uri = MILVUS_URI
    try:
        if uri.startswith("http://") or uri.startswith("https://") or ":" in uri:
            print(f"  Attempting connection to Milvus Standalone at: {uri}")
            client = MilvusClient(uri=uri)
            # Dry-run command to test connection
            client.list_collections()
            print("  Successfully connected to Milvus Standalone!")
            return client, uri
        else:
            print(f"  Connecting to local Milvus Lite file: {uri}")
            return MilvusClient(uri), uri
    except Exception as e:
        print(f"  [Warning] Failed to connect to Milvus at {uri}: {e}")
        print(f"  [Fallback] Reverting to local Milvus Lite ({MILVUS_FALLBACK})")
        return MilvusClient(MILVUS_FALLBACK), MILVUS_FALLBACK


# ============================================================
# STEP 1: Load the saved vector store from disk
# ============================================================

def load_vector_store():

    print("Loading embedding model...")

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL   # reads from .env
    )

    print("Loading vector store...")

    client, actual_uri = get_milvus_client()

    if not client.has_collection(COLLECTION_NAME):
        print(f"\n[Warning] Collection '{COLLECTION_NAME}' does not exist in Milvus at {actual_uri}!")
        print("Please run the ingestion pipeline first: python ingestion/ingest.py\n")
    else:
        client.load_collection(COLLECTION_NAME)

    print("Vector store ready.\n")
    return {"client": client, "embedder": embedding_model, "uri": actual_uri}


# ============================================================
# STEP 2: Search the vector store for relevant chunks
# ============================================================

def retrieve_context(vector_db, question, user_role, k=5):
    """
    k=5 means: retrieve the 5 most relevant chunks.
    With 13 chunks total, k=5 covers 38% of the knowledge base per query.
    k=3 was too few (23%) — right chunks existed but weren't fetched.
    LangChain handles the embedding of the question internally.
    """
    client = vector_db["client"]
    embedder = vector_db["embedder"]

    vec = embedder.embed_query(question)
    
    # Secure RBAC Logic
    # We look for the user's role in the allowed_roles string, e.g. "[EMPLOYEE]"
    filter_expr = f'allowed_roles LIKE "%[{user_role}]%"'

    # Ensure collection is loaded in memory before searching
    client.load_collection(COLLECTION_NAME)

    results = client.search(
        collection_name=COLLECTION_NAME,   # reads from .env
        data=[vec],
        limit=k,
        filter=filter_expr,
        output_fields=["text", "source", "parent_document_name", "document_title", "file_path", "extraction_date", "allowed_roles"]
    )

    docs = []
    if results and len(results) > 0:
        for hit in results[0]:
            entity = hit["entity"]
            docs.append(Document(
                page_content=entity["text"],
                metadata={
                    "source":               entity.get("source", "Unknown"),
                    "parent_document_name": entity.get("parent_document_name", ""),
                    "document_title":       entity.get("document_title", ""),
                    "file_path":            entity.get("file_path", ""),
                    "extraction_date":      entity.get("extraction_date", ""),
                    "allowed_roles":        entity.get("allowed_roles", "")
                }
            ))

    return docs


# ============================================================
# STEP 3: Build the augmented prompt
# ============================================================

def build_prompt(question, retrieved_chunks):

    context = "\n\n---\n\n".join([
        doc.page_content for doc in retrieved_chunks
    ])

    system_prompt = "You are an AI assistant. Answer ONLY using the information provided. If the answer is not in the information, say 'I don't have that information.' DO NOT guess."
    
    user_prompt = f"Information:\n{context}\n\nQuestion:\n{question}"

    return system_prompt, user_prompt


# ============================================================
# STEP 4: Send prompt to LLM and get answer
# ============================================================

def generate_answer(system_prompt, user_prompt):

    client = OpenAI(
        base_url=VLLM_API_BASE,
        api_key=VLLM_API_KEY
    )

    response = client.chat.completions.create(
        model=VLLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content


# ============================================================
# STEP 4.5: Web Fallback (Agentic Search)
# ============================================================

def perform_web_search(query):
    import warnings
    # pyrefly: ignore [missing-import]
    from duckduckgo_search import DDGS
        
    try:
        # Suppress the duckduckgo package rename warning during execution
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=10)  # Increased from 3 to 10 for better accuracy
            
            if not results:
                return "No web results found."
                
            snippets = []
            for item in results:
                snippets.append(f"Title: {item.get('title')}\nURL: {item.get('href')}\nSnippet: {item.get('body')}")
            
            return "\n\n".join(snippets)
            
    except Exception as e:
        return f"[Error] Web search failed: {e}"


# ============================================================
# STEP 5: API Entry Point
# ============================================================

# Global variable to hold the loaded vector store in memory
_vector_db_cache = None

def get_rag_response(question: str, user_role: str):
    global _vector_db_cache
    if _vector_db_cache is None:
        _vector_db_cache = load_vector_store()

    # Step A: Find relevant chunks from vector store using RBAC
    print(f"\nSearching knowledge base for role [{user_role}]...")
    retrieved_chunks = retrieve_context(_vector_db_cache, question, user_role, k=5)

    if not retrieved_chunks:
        return "Access denied or no relevant documents found. You don't have permission to access this information."

    print(f"Found {len(retrieved_chunks)} relevant chunk(s):")
    sources = set()
    for i, chunk in enumerate(retrieved_chunks):
        source_file = os.path.basename(chunk.metadata.get("source", "Unknown"))
        sources.add(source_file)
        preview = chunk.page_content[:80].replace('\n', ' ').strip()
        print(f"  [{i+1}] (From: {source_file}) {preview}...")

    # Step B: Build prompt with context + question
    sys_prompt, usr_prompt = build_prompt(question, retrieved_chunks)

    # Step C: Get answer from LLM
    print("\nGenerating answer...")
    answer = generate_answer(sys_prompt, usr_prompt)

    # Step D: Check if we need to fallback to the Web
    fallback_phrases = ["don't have", "do not have", "not in the information", "not provided", "cannot answer"]
    needs_fallback = any(phrase in answer.lower() for phrase in fallback_phrases)
    
    if needs_fallback:
        print("\n  [!] Local knowledge base lacked information.")
        print("  [!] Triggering Agentic Web Search (DuckDuckGo)...")
        
        web_context = perform_web_search(question)
        
        if "[Error]" in web_context:
            print(f"  {web_context}")
        else:
            print("  [✓] Web search successful. Synthesizing final answer from live data...")
            web_sys_prompt = "You are an AI assistant with live internet access. Answer ONLY using the web search results. If the results do not explicitly contain the answer, say 'I cannot find a reliable answer on the web.' DO NOT guess."
            web_usr_prompt = f"Web Search Results:\n{web_context}\n\nQuestion:\n{question}"
            answer = generate_answer(web_sys_prompt, web_usr_prompt)
            sources.add("Web Search (DuckDuckGo)")

    source_str = "\nSource:\n" + "\n".join(sources)
    return answer + "\n" + source_str


# ============================================================
# STEP 6: MLflow Evaluation Wrapper
# ============================================================
@mlflow.trace(name="rag_agent")
def predict_fn(inputs):
    """
    Wrapper function required by mlflow.genai.evaluate()
    """
    import pandas as pd
    if isinstance(inputs, pd.DataFrame):
        queries = inputs.get("inputs", inputs.get("questions", inputs.iloc[:, 0]))
        return [get_rag_response(str(q), "FINANCE_MANAGER") for q in queries]
    elif isinstance(inputs, pd.Series) or isinstance(inputs, list):
        return [get_rag_response(str(q), "FINANCE_MANAGER") for q in inputs]
    elif isinstance(inputs, dict):
        query = inputs.get("query", inputs.get("questions", ""))
        user_role = inputs.get("role", "FINANCE_MANAGER")
        return get_rag_response(query, user_role)
    else:
        # Fallback if passed as string
        query = str(inputs)
        user_role = "FINANCE_MANAGER"
        return get_rag_response(query, user_role)



if __name__ == "__main__":
    pass
