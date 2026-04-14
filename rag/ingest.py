import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

REPO_PATH   = "./data/fastapi"
CHROMA_PATH = "./data/chroma_db"
EXTENSIONS  = [".py", ".md"]
CHUNK_SIZE  = 1000
CHUNK_OVERLAP = 150

def load_documents(repo_path):
    docs = []
    repo = Path(repo_path)
    for ext in EXTENSIONS:
        files = list(repo.rglob(f"*{ext}"))
        print(f"  Found {len(files)} {ext} files")
        for filepath in files:
            try:
                loader = TextLoader(str(filepath), encoding="utf-8")
                file_docs = loader.load()
                for doc in file_docs:
                    doc.metadata["source_file"] = str(filepath.relative_to(repo))
                    doc.metadata["extension"]   = ext
                    doc.metadata["repo"]        = repo.name
                docs.extend(file_docs)
            except Exception as e:
                print(f"  [SKIP] {filepath.name}: {e}")
    print(f"\n  Total documents loaded: {len(docs)}")
    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\nclass ", "\ndef ", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"  Split into {len(chunks)} chunks")
    return chunks

def build_vectorstore(chunks):
    print(f"\n  Embedding {len(chunks)} chunks locally...")
    print("  (First run downloads ~90MB model - then it is offline forever)")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
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
    print("  DevMind - Phase 1: Ingestion Pipeline")
    print("=" * 55)
    print("\n[1/3] Loading files from repo...")
    docs = load_documents(REPO_PATH)
    print("\n[2/3] Splitting into chunks...")
    chunks = split_documents(docs)
    print("\n[3/3] Building vector store...")
    build_vectorstore(chunks)
    print("\n  Done! Run python rag/query.py to ask questions.\n")

if __name__ == "__main__":
    main()
