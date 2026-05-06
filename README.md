# Legal Document RAG Pipeline

A Retrieval-Augmented Generation (RAG) system for semantic search over legal documents using LangChain, HuggingFace embeddings, and FAISS vector store.

## Overview

This pipeline ingests legal documents, splits them into semantically meaningful chunks, embeds them using a sentence transformer model, and indexes them in a FAISS vector store for fast similarity search.

## Architecture

```
Raw Documents (DataFrame)
        ↓
RecursiveCharacterTextSplitter  (chunk_size=450, overlap=80)
        ↓
LangChain Document Objects      (with metadata: doc_id, chunk index)
        ↓
HuggingFace Embeddings          (all-MiniLM-L6-v2, GPU-accelerated)
        ↓
FAISS Vector Store              (L2-normalized, saved locally)
```

## Requirements

```bash
pip install langchain langchain-community langchain-text-splitters
pip install sentence-transformers faiss-gpu
```

> Use `faiss-cpu` if no GPU is available.

## Input Format

The pipeline expects a pandas DataFrame `df` with the following columns:

| Column   | Type   | Description                        |
|----------|--------|------------------------------------|
| `doc_id` | string | Unique identifier for the document |
| `text`   | string | Full text content of the document  |

## Configuration

| Parameter             | Value                                    | Description                          |
|-----------------------|------------------------------------------|--------------------------------------|
| `chunk_size`          | 450                                      | Max characters per chunk             |
| `chunk_overlap`       | 80                                       | Overlap between consecutive chunks  |
| `model_name`          | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model                      |
| `device`              | `cuda`                                   | Inference device (GPU)               |
| `normalize_embeddings`| `True`                                   | L2-normalize vectors before indexing |
| `save_path`           | `/content/legal_faiss`                   | FAISS index output path              |

## Usage

```python
# 1. Prepare your DataFrame
# df must have columns: "doc_id", "text"

# 2. Run the pipeline (the script handles chunking → embedding → indexing)
# Output: FAISS index saved to /content/legal_faiss

# 3. Load and query the index
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = FAISS.load_local("/content/legal_faiss", embeddings)

# Semantic search
results = vectorstore.similarity_search("breach of contract clause", k=5)
for doc in results:
    print(doc.metadata, doc.page_content[:200])
```

## Output

- **FAISS index** saved to `/content/legal_faiss/`
  - `index.faiss` — binary vector index
  - `index.pkl` — docstore and metadata

## Metadata Schema

Each indexed chunk carries the following metadata:

```json
{
  "source": "<doc_id>",
  "chunk": <chunk_index>
}
```

## Notes

- Switch `device` to `"cpu"` if running without a GPU.
- Chunk size of 450 characters is tuned for legal clause granularity — adjust based on your document structure.
- `all-MiniLM-L6-v2` produces 384-dimensional embeddings and is a good balance of speed and quality for English legal text.
- For production use, consider replacing FAISS with a persistent vector DB (e.g., Pinecone, Weaviate, Chroma).

## License

MIT