import os

# Silence TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_transformers import EmbeddingsRedundantFilter

# ------------- CONFIG -------------
CHROMA_PATH = "data/chroma_store"
EMBED_MODEL = "all-MiniLM-L6-v2"


# ------------- LOAD DOCUMENTS -------------
def load_documents(file_paths):
    """
    Load PDF and text files using LangChain loaders.
    """
    docs = []
    for f in file_paths:
        if f.endswith(".pdf"):
            loader = PyPDFLoader(f)
        elif f.endswith(".txt"):
            loader = TextLoader(f)
        else:
            print(f"⚠️ Unsupported file type: {f}")
            continue
        docs.extend(loader.load())
    return docs


# ------------- BUILD VECTOR DB -------------
def build_vector_db(file_paths):
    """
    Uses LangChain's text splitter, embedding model,
    redundancy filter, and Chroma vector store.
    """
    docs = load_documents(file_paths)
    if not docs:
        print("❌ No valid documents found.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    deduper = EmbeddingsRedundantFilter(embeddings=embedder)
    refined = deduper.transform_documents(chunks)

    db = Chroma.from_documents(refined, embedder, persist_directory=CHROMA_PATH)
    print(f"✅ Stored {len(refined)} refined chunks in {CHROMA_PATH}")
    return db


# ------------- QUERY VECTOR DB -------------
def query_vector_db(query, n_results=3):
    """
    Retrieve semantically relevant chunks from the vector DB.
    Works with both older and newer LangChain retriever APIs.
    """
    if not os.path.exists(CHROMA_PATH):
        print("❌ Vector DB not found. Run build_vector_db() first.")
        return []

    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedder)
    retriever = db.as_retriever(search_kwargs={"k": n_results})

    results = None
    if hasattr(retriever, "invoke"):
        results = retriever.invoke(query)
    elif hasattr(retriever, "get_relevant_documents"):
        results = retriever.get_relevant_documents(query)
    elif hasattr(retriever, "retrieve"):
        results = retriever.retrieve(query)
    elif callable(retriever):
        results = retriever(query)
    else:
        print("❌ Unsupported retriever method. Check LangChain version.")
        return []

    if not results:
        print("⚠️ No results found.")
        return []

    return [getattr(r, "page_content", str(r)) for r in results]


# ------------- TEST (Run Standalone) -------------
if __name__ == "__main__":
    # Add documents once
    files = ["data/papers/Agile_Frameworks.pdf"]
    build_vector_db(files)

    # Query
    question = "Explain agile frameworks."
    results = query_vector_db(question, n_results=3)

    print(f"\n🔎 Retrieved {len(results)} chunks:\n")
    for i, txt in enumerate(results, 1):
        print(f"Result {i}\n{'-'*60}\n{txt[:400]}...\n")
