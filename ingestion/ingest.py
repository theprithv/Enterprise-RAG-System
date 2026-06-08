# ============================================================
# INGESTION PIPELINE
# Runs ONCE to load, chunk, embed, and store your documents.
# All settings are read from the .env file — no hard-coded values.
#
# Supported document types (via Unstructured.io):
#   PDF, DOCX, PPTX, XLSX, HTML, TXT, and more
# Just drop any file into the DOCS_DIR folder and run this script.
# ============================================================

import sys
import os
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()  # Reads all values from .env file automatically

from pathlib import Path
from typing import TypedDict, List

# pyrefly: ignore [missing-import]
from langchain_huggingface import HuggingFaceEmbeddings
# pyrefly: ignore [missing-import]
from langchain_experimental.text_splitter import SemanticChunker
# pyrefly: ignore [missing-import]
from pymilvus import MilvusClient
# pyrefly: ignore [missing-import]
from langchain_core.documents import Document
# pyrefly: ignore [missing-import]
from langgraph.graph import StateGraph, START, END
from pymilvus import CollectionSchema, FieldSchema, DataType
from datetime import datetime

# ── Load config from .env ──────────────────────────────────
MILVUS_URI       = os.getenv("MILVUS_URI",        "http://localhost:19530")
MILVUS_FALLBACK  = os.getenv("MILVUS_FALLBACK_DB", "./milvus_rag.db")
COLLECTION_NAME  = os.getenv("MILVUS_COLLECTION",  "rag_knowledge")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL",    "all-MiniLM-L6-v2")
VECTOR_DIM       = int(os.getenv("VECTOR_DIM",     "384"))
DOCS_DIR         = os.getenv("DOCS_DIR",           "data/documents")
# ──────────────────────────────────────────────────────────

# Supported extensions by Unstructured (auto-detected at runtime)
SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".pptx", ".ppt",
    ".xlsx", ".xls", ".html", ".htm", ".txt", ".md",
    ".eml", ".msg", ".rst", ".rtf", ".csv"
}


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
# STEP 1: Define the pipeline state
# LangGraph passes this dictionary between every node.
# Each node reads from it and writes back to it.
# ============================================================

class IngestionState(TypedDict):
    docs_dir:        str           # folder containing documents (from .env)
    documents:       List[Document]
    chunks:          List[Document]
    embedding_model: object
    vector_db:       object


# ============================================================
# STEP 2: Define each node (each node = one step)
# ============================================================

def node_load_document(state: IngestionState) -> dict:
    """
    NEW: Docling (Document Parsing)
    Perfectly parses PDFs with 2-column layouts, tables, and charts.
    """
    from docling.document_converter import DocumentConverter

    print("\n[Node 1] Loading documents with IBM Docling...")

    docs_path = Path(state["docs_dir"])

    # Validate the documents folder exists
    if not docs_path.exists():
        raise FileNotFoundError(
            f"  [Error] DOCS_DIR not found: '{docs_path}'. "
            "Please create the folder and add your documents."
        )

    # Collect all supported files in the folder
    all_files = [
        f for f in sorted(docs_path.iterdir())
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not all_files:
        raise ValueError(
            f"  [Error] No supported documents found in '{docs_path}'. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    print(f"  Found {len(all_files)} document(s) in '{docs_path}':")
    for f in all_files:
        print(f"    • {f.name}  ({f.suffix.upper().lstrip('.')})")

    documents = []
    converter = DocumentConverter()

    for file_path in all_files:
        try:
            extension = file_path.suffix.lower()

            print(f"  [Docling] Parsing: {file_path.name}...")
            
            result = converter.convert(str(file_path))
            text = result.document.export_to_markdown()

            if not text.strip():
                print(f"  [Warning] No text extracted from '{file_path.name}' — skipping.")
                continue

            # Determine role based on filename
            assigned_role = "admin" if "secret" in file_path.name.lower() else "user"

            # Wrap in a LangChain Document (same structure as before)
            doc = Document(
                page_content=text,
                metadata={
                    "source":               str(file_path),
                    "filename":             file_path.name,
                    "extension":            extension,
                    "file_path":            str(file_path.absolute()),
                    "parent_document_name": file_path.stem,
                    "document_title":       file_path.stem,
                    "extraction_date":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "required_role":        assigned_role
                }
            )
            documents.append(doc)

            print(f"  Loaded  : {file_path.name}")
            print(f"  Preview : {text[:80]}...")

        except Exception as e:
            # Log and skip bad files — don't crash the pipeline
            print(f"  [Warning] Failed to parse '{file_path.name}': {e} — skipping.")

    if not documents:
        raise RuntimeError(
            "  [Error] All documents failed to parse. "
            "Check file formats and Unstructured installation."
        )

    print(f"\n  Total documents loaded: {len(documents)}")
    return {"documents": documents}


def node_load_embedding_model(state: IngestionState) -> dict:
    """
    OLD: SentenceTransformer("all-MiniLM-L6-v2")  <- direct, not LangChain compatible
    NEW: HuggingFaceEmbeddings(model_name=...)     <- LangChain compatible wrapper
    Same model, same vectors, works with all LangChain tools.
    """
    print("\n[Node 2] Loading embedding model...")

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL   # reads from .env
    )

    print("  Embedding model ready.")

    return {"embedding_model": embedding_model}


def node_chunk_documents(state: IngestionState) -> dict:
    """
    OLD: document.split("\\n\\n")  <- blind split, breaks meaning
    NEW: SemanticChunker           <- splits based on meaning change
    Keeps related sentences together in the same chunk.
    """
    print("\n[Node 3] Semantic chunking...")

    chunker = SemanticChunker(
        embeddings=state["embedding_model"],
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=60
        # 60 = splits more often → smaller, focused chunks
        # default is 95 = splits rarely → 3 huge chunks (too big)
        # lower number = more chunks = more precise retrieval
    )

    chunks = chunker.split_documents(state["documents"])

    print(f"  Created {len(chunks)} chunks.")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {chunk.page_content[:60]}...")

    return {"chunks": chunks}


def node_store_vectors(state: IngestionState) -> dict:
    """
    Using pure pymilvus.MilvusClient to bypass the langchain-milvus ORM bug.
    """
    print("\n[Node 4] Embedding chunks and saving to Milvus...")

    client, actual_uri = get_milvus_client()
    if client.has_collection(COLLECTION_NAME):
        try:
            client.drop_collection(COLLECTION_NAME)
        except Exception as e:
            print(f"  [Warning] Ignoring drop_collection error: {e}")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="parent_document_name", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="document_title", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="extraction_date", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="required_role", dtype=DataType.VARCHAR, max_length=128),
    ]
    schema = CollectionSchema(fields=fields, description="RAG Knowledge Base with strict Metadata")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema
    )

    data = []
    for i, chunk in enumerate(state["chunks"]):
        vec = state["embedding_model"].embed_query(chunk.page_content)
        data.append({
            "id":                   i,
            "vector":               vec,
            "text":                 chunk.page_content,
            "source":               chunk.metadata.get("source", ""),
            "file_path":            chunk.metadata.get("file_path", ""),
            "parent_document_name": chunk.metadata.get("parent_document_name", ""),
            "document_title":       chunk.metadata.get("document_title", ""),
            "extraction_date":      chunk.metadata.get("extraction_date", ""),
            "required_role":        chunk.metadata.get("required_role", "user")
        })

    client.insert(COLLECTION_NAME, data=data)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="COSINE"
    )
    client.create_index(
        collection_name=COLLECTION_NAME,
        index_params=index_params
    )

    print(f"  Stored  : {len(state['chunks'])} chunks")
    print(f"  Saved to: {actual_uri}")

    return {"vector_db": {"client": client, "uri": actual_uri}}


# ============================================================
# STEP 3: Build the LangGraph pipeline
# Connect the nodes in order. LangGraph controls the flow.
# ============================================================

def build_pipeline():

    graph = StateGraph(IngestionState)

    # Register nodes
    graph.add_node("load_document",   node_load_document)
    graph.add_node("load_embedding",  node_load_embedding_model)
    graph.add_node("chunk_documents", node_chunk_documents)
    graph.add_node("store_vectors",   node_store_vectors)

    # Connect nodes (define execution order)
    graph.add_edge(START,             "load_document")
    graph.add_edge("load_document",   "load_embedding")
    graph.add_edge("load_embedding",  "chunk_documents")
    graph.add_edge("chunk_documents", "store_vectors")
    graph.add_edge("store_vectors",   END)

    return graph.compile()


# ============================================================
# STEP 4: Run the pipeline
# ============================================================

if __name__ == "__main__":

    print("=" * 55)
    print("  INGESTION PIPELINE STARTING")
    print(f"  Docs Dir: {DOCS_DIR}")
    print(f"  Milvus  : {MILVUS_URI}")
    print(f"  Model   : {EMBEDDING_MODEL}")
    print("=" * 55)

    pipeline = build_pipeline()

    result = pipeline.invoke({
        "docs_dir": DOCS_DIR    # reads from .env
    })

    print("\n" + "=" * 55)
    print("  INGESTION COMPLETE")
    print(f"  Total chunks stored : {len(result['chunks'])}")
    print(f"  Vector store saved  : {result['vector_db']['uri']}")
    print("=" * 55)
    print("\nYou can now run: python retrieval/retrieve.py")
