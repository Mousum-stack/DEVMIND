"""
DevMind - Phase 1: RAG Core
File: rag/ingest.py

What this does:
  1. Clones a GitHub repo (or uses a local folder)
  2. Reads all .py and .md files
  3. Splits them into smart chunks
  4. Embeds them with OpenAI
  5. Stores in ChromaDB (local vector store)

Run this ONCE to build your knowledge base.
Then use query.py to ask questions.
"""

import os
import glob
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
REPO_PATH   = "./data/fastapi"          # local path to the cloned repo
CHROMA_PATH = "./data/chroma_db"        # where ChromaDB stores its files
EXTENSIONS  = [".py", ".md"]            # file types to index
CHUNK_SIZE  = 1000                      # characters per chunk
CHUNK_OVERLAP = 150                     # overlap between chunks (keeps context)
EMBED_MODEL = "text-embedding-3-small"  # cheap + good enough
# ─────────────────────────────────────────────────────────────────────────────


def load_documents(repo_path: str) -> list:
    """Walk the repo and load all matching files as LangChain Documents."""
    docs = []
    repo = Path(repo_path)

    if not repo.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Repo not found at: {repo_path}"
            f"\nRun this first:\n  git clone https://github.com/tiangolo/fastapi ./data/fastapi"
        )

    for ext in EXTENSIONS:
        files = list(repo.rglob(f"*{ext}"))
        print(f"  Found {len(files)} {ext} files")

        for filepath in files:
            try:
                loader = TextLoader(str(filepath), encoding="utf-8")
                file_docs = loader.load()

                # Add useful metadata to every chunk
                for doc in file_docs:
                    doc.metadata["source_file"] = str(filepath.relative_to(repo))
                    doc.metadata["extension"]   = ext
                    doc.metadata["repo"]        = repo.name

                docs.extend(file_docs)
            except Exception as e:
                print(f"  [SKIP] {filepath.name}: {e}")

    print(f"\n  Total documents loaded: {len(docs)}")
    return docs


def split_documents(docs: list) -> list:
    """Split documents into chunks. Code-aware splitting."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # These separators are tried in order — prefers splitting at
        # class/function boundaries before arbitrary character positions
        separators=["\nclass ", "\ndef ", "\n\n", "\n", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    print(f"  Split into {len(chunks)} chunks")
    return chunks


def build_vectorstore(chunks: list) -> Chroma:
    """Embed chunks and store in ChromaDB."""
    print(f"\n  Embedding {len(chunks)} chunks with {EMBED_MODEL}...")
    print("  (This costs ~$0.002 for a medium repo — runs once)")

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

    # Chroma saves to disk automatically at CHROMA_PATH
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name="devmind_code",
    )

    print(f"  Saved to {CHROMA_PATH}")
    return vectorstore


def main():
    print("=" * 55)
    print("  DevMind — Phase 1: Ingestion Pipeline")
    print("=" * 55)

    print("\n[1/3] Loading files from repo...")
    docs = load_documents(REPO_PATH)

    print("\n[2/3] Splitting into chunks...")
    chunks = split_documents(docs)

    print("\n[3/3] Building vector store...")
    vectorstore = build_vectorstore(chunks)

    print("\n  Done! Your knowledge base is ready.")
    print("  Next step: run `python rag/query.py` to ask questions.\n")


if __name__ == "__main__":
    main()
