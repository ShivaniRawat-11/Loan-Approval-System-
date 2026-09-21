from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from database.auth_db import get_chat_history, save_chat_history, clear_chat_history
from utils.security import get_current_user
from services.agent_service import (
    create_general_agent,
    get_response,
    get_response_stream,
    MODE_GENERAL,
    MODE_DOCUMENT,
)

router = APIRouter()

# ── In-memory user session store for document agents ─────────────────────────
# Keyed by user email → stores doc_agent objects after document processing
_user_sessions = {}


class ChatMessage(BaseModel):
    role: str
    content: str
    credit_score_html: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    mode: str = "general"  # "general" or "document"
    chat_history: List[ChatMessage] = []


class ChatResponse(BaseModel):
    response: str
    credit_score_html: Optional[str] = None


def _get_agent_for_mode(email: str, mode: str):
    """Get the appropriate agent based on mode and user session."""
    if mode == "document":
        session = _user_sessions.get(email)
        if session and session.get("doc_agent"):
            return session["doc_agent"], MODE_DOCUMENT
        else:
            return None, MODE_DOCUMENT
    else:
        return create_general_agent(), MODE_GENERAL


def _detect_credit_score_html(prompt: str, response: str) -> Optional[str]:
    """Generate credit score bar HTML if the response is about credit scores."""
    if "credit score" in prompt.lower() and any(score in response for score in ["620", "650", "700"]):
        return """
        <div style="background:#ffffff;border:1px solid #e4e9f2;border-radius:12px;padding:1.2rem;margin-top:1rem;box-shadow:0 2px 10px rgba(0,0,0,0.02)">
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;font-weight:700;font-size:0.85rem">
                <span style="color:#e04a4a;flex:1;text-align:center">Poor</span>
                <span style="color:#f59e0b;flex:1;text-align:center">Fair</span>
                <span style="color:#d4a84e;flex:1;text-align:center">Good</span>
                <span style="color:#6c7a9c;flex:1;text-align:center">Very Good</span>
                <span style="color:#6c7a9c;flex:1;text-align:center">Excellent</span>
            </div>
            <div style="display:flex;height:12px;border-radius:6px;overflow:hidden;margin-bottom:8px;position:relative">
                <div style="background:#e04a4a;flex:1"></div>
                <div style="background:#f59e0b;flex:1"></div>
                <div style="background:#eab308;flex:1;position:relative">
                    <div style="position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);width:6px;height:6px;background:#1a2540;border-radius:50%"></div>
                </div>
                <div style="background:#5ba0e0;flex:1"></div>
                <div style="background:#2a9d8f;flex:1"></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#6c7a9c">
                <span style="flex:1;text-align:center">300-579</span>
                <span style="flex:1;text-align:center">580-660</span>
                <span style="flex:1;text-align:center;position:relative">
                    <span style="background:#d4a84e;color:#fff;padding:2px 8px;border-radius:6px;font-weight:700;font-size:0.85rem;display:inline-block">650</span>
                </span>
                <span style="flex:1;text-align:center">670-739</span>
                <span style="flex:1;text-align:center">800-850</span>
            </div>
        </div>
        """
    return None


@router.post("/send", response_model=ChatResponse)
def chat_send(req: ChatRequest, email: str = Depends(get_current_user)):
    """Send a message and get a non-streaming response."""
    if not req.message:
        raise HTTPException(status_code=400, detail="Message is required.")

    agent_objects, active_mode = _get_agent_for_mode(email, req.mode)
    
    if agent_objects is None:
        return ChatResponse(
            response="⚠️ **Document-Based Analysis requires uploaded documents.** Please upload files first."
        )

    history_dicts = [{"role": m.role, "content": m.content} for m in req.chat_history]
    
    response_text = get_response(agent_objects, req.message, history_dicts, active_mode)
    
    credit_score_html = _detect_credit_score_html(req.message, response_text) if req.mode == "general" else None

    # Save to DB
    messages = history_dicts.copy()
    messages.append({"role": "user", "content": req.message})
    messages.append({"role": "assistant", "content": response_text, "credit_score_html": credit_score_html})
    save_chat_history(email, messages)

    return ChatResponse(response=response_text, credit_score_html=credit_score_html)


@router.post("/stream")
def chat_stream(req: ChatRequest, email: str = Depends(get_current_user)):
    """Send a message and get a Server-Sent Events stream."""
    import json
    if not req.message:
        raise HTTPException(status_code=400, detail="Message is required.")

    agent_objects, active_mode = _get_agent_for_mode(email, req.mode)
    
    if agent_objects is None:
        def error_gen():
            yield f"data: {json.dumps({'text': '⚠️ **Document-Based Analysis requires uploaded documents.** Please upload files first.'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    history_dicts = [{"role": m.role, "content": m.content} for m in req.chat_history]

    def event_generator():
        collected = []
        try:
            for chunk in get_response_stream(agent_objects, req.message, history_dicts, active_mode):
                collected.append(chunk)
                yield f"data: {json.dumps({'text': chunk})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'text': f'⚠️ Error: {str(e)}'})}\n\n"
        
        # Save complete response to DB
        full_response = "".join(collected)
        if full_response:
            credit_score_html = _detect_credit_score_html(req.message, full_response) if req.mode == "general" else None
            messages = history_dicts.copy()
            messages.append({"role": "user", "content": req.message})
            messages.append({"role": "assistant", "content": full_response, "credit_score_html": credit_score_html})
            save_chat_history(email, messages)
        
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/history")
def chat_history(email: str = Depends(get_current_user)):
    """Get chat history for a user."""
    messages = get_chat_history(email.strip().lower())
    return {"messages": messages}


@router.post("/clear")
def chat_clear(email: str = Depends(get_current_user)):
    """Clear chat history for a user."""
    success = clear_chat_history(email.strip().lower())
    # Also clear in-memory agent session
    _user_sessions.pop(email.strip().lower(), None)
    return {"success": success}


def set_user_session(email: str, session_data: dict):
    """Called by documents route to store doc agent in user session."""
    _user_sessions[email.strip().lower()] = session_data
