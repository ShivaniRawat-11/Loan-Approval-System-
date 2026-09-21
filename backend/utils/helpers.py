import os
import io
import logging
from typing import List, Tuple, Dict
import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Logging Setup ────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that writes to both console and a log file."""
    logger = logging.getLogger(name)
    if logger.handlers:          # avoid adding duplicate handlers on re-import
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_file = os.path.join(LOG_DIR, "app.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

logger = get_logger("utils")

# ── Document Ingestion and Splitting ─────────────────────────────────────────
def load_and_split_files(files_data: List[Dict]) -> Tuple[List[Document], List[str], Dict[str, str]]:
    """
    Takes a list of dicts with 'filename' and 'content' (bytes),
    loads PDF/TXT contents, splits them into recursive character chunks,
    and returns (splits, file_names, raw_texts).
    
    This version accepts raw bytes instead of Streamlit UploadedFile objects.
    """
    all_documents = []
    file_names = []
    raw_texts = {}

    for file_info in files_data:
        file_name = file_info["filename"]
        file_bytes = file_info["content"]
        file_ext = os.path.splitext(file_name)[1].lower()

        try:
            docs = []
            if file_ext == ".pdf":
                pdf_stream = io.BytesIO(file_bytes)
                reader = pypdf.PdfReader(pdf_stream)
                for page_idx, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    docs.append(Document(
                        page_content=page_text,
                        metadata={"source_file": file_name, "page": page_idx + 1}
                    ))
            elif file_ext == ".txt":
                text_content = file_bytes.decode("utf-8", errors="ignore")
                docs.append(Document(
                    page_content=text_content,
                    metadata={"source_file": file_name}
                ))
            else:
                logger.warning(f"Skipping unsupported file type: {file_name}")
                continue

            if not docs or all(len(d.page_content.strip()) == 0 for d in docs):
                logger.warning(f"No content extracted from {file_name}")
                continue

            # Accumulate full raw text for structured analytics
            full_text = ""
            for doc in docs:
                doc.metadata["source_file"] = file_name
                full_text += doc.page_content + "\n\n"
                
            all_documents.extend(docs)
            file_names.append(file_name)
            raw_texts[file_name] = full_text.strip()
            logger.info(f"Loaded {len(docs)} page(s) from {file_name}")
        except Exception as e:
            logger.error(f"Could not load {file_name}: {e}", exc_info=True)
            continue

    if not all_documents:
        logger.warning("No documents were successfully loaded.")
        return [], [], {}

    # Split into chunks (Larger chunks for better context preservation)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    splits = text_splitter.split_documents(all_documents)
    logger.info(f"Split {len(all_documents)} document(s) into {len(splits)} chunks.")
    return splits, file_names, raw_texts

# ── Generic Input Sanitization ───────────────────────────────────────────────
def sanitize_input(user_input: str, max_length: int = 1500) -> str:
    """Sanitize and validate user input length."""
    if not user_input or not user_input.strip():
        return ""
    cleaned = user_input.strip()
    if len(cleaned) > max_length:
        logger.warning(f"User input truncated from {len(cleaned)} to {max_length} chars")
        cleaned = cleaned[:max_length]
    return cleaned
