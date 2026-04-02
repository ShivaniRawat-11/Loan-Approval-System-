import os
import tempfile
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Shared embedding model (loaded once)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_vector_store_from_files(uploaded_files):
    """
    Takes a list of Streamlit UploadedFile objects,
    loads them as documents, splits into chunks,
    and returns an in-memory FAISS vector store.
    
    Supports: .pdf, .txt
    """
    all_documents = []
    file_names = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            file_ext = os.path.splitext(file_name)[1].lower()
            tmp_path = os.path.join(tmp_dir, file_name)

            # Save to temp file
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Load based on file type
            try:
                if file_ext == ".pdf":
                    loader = PyPDFLoader(tmp_path)
                elif file_ext == ".txt":
                    loader = TextLoader(tmp_path, encoding="utf-8")
                else:
                    continue  # Skip unsupported files

                docs = loader.load()
                # Tag each doc with its source filename
                for doc in docs:
                    doc.metadata["source_file"] = file_name
                all_documents.extend(docs)
                file_names.append(file_name)
            except Exception as e:
                print(f"Warning: Could not load {file_name}: {e}")
                continue

    if not all_documents:
        return None, []

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    splits = text_splitter.split_documents(all_documents)

    # Build FAISS index in memory
    vector_store = FAISS.from_documents(splits, embeddings)
    return vector_store, file_names


def get_retriever(vector_store, k=4):
    """Returns a retriever from the given vector store."""
    return vector_store.as_retriever(search_kwargs={"k": k})


def search_documents(query: str, retriever) -> str:
    """
    Retrieves relevant document chunks for a given query.
    Returns formatted string with source attribution.
    """
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found in the uploaded documents."
    
    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "Unknown")
        results.append(f"[Source: {source}]\n{doc.page_content}")
    
    return "RETRIEVED DOCUMENT CONTENT:\n\n" + "\n\n---\n\n".join(results)
