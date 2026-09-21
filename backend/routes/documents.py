from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List
from services.agent_service import process_documents
from routes.chat import set_user_session
from utils.helpers import get_logger
from utils.security import get_current_user

router = APIRouter()
logger = get_logger("documents")


@router.post("/process")
async def process_uploaded_documents(
    files: List[UploadFile] = File(...),
    email: str = Depends(get_current_user)
):
    """
    Upload and process loan-related documents.
    Builds FAISS index, extracts structured analysis, and stores doc agent in session.
    """
    
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    # Read file contents
    files_data = []
    for f in files:
        content = await f.read()
        if not content:
            logger.warning(f"Empty file skipped: {f.filename}")
            continue
        files_data.append({
            "filename": f.filename,
            "content": content,
        })

    if not files_data:
        raise HTTPException(status_code=400, detail="No valid files uploaded.")

    # Process documents (FAISS + analysis)
    try:
        result = process_documents(files_data)
    except Exception as e:
        logger.error(f"Document processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

    if result is None:
        raise HTTPException(
            status_code=400,
            detail="No text content could be extracted from the uploaded files. Ensure the files are not empty or corrupted."
        )

    # Store doc agent in user session
    email_clean = email.strip().lower()
    set_user_session(email_clean, {
        "doc_agent": result["doc_agent"],
        "file_names": result["file_names"],
        "document_analyses": result["document_analyses"],
    })

    return {
        "success": True,
        "file_names": result["file_names"],
        "document_analyses": result["document_analyses"],
        "message": f"{len(result['file_names'])} document(s) processed & analyzed!"
    }
