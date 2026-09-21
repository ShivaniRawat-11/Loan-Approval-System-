"""
Agent Service — Core LangChain logic for the Loan Approval System.
Ported from the original Streamlit agent.py with all st.* dependencies removed.
Uses module-level singletons and functools.lru_cache instead of st.cache_resource.
"""

import os
import sys
import time
import re
import hashlib
import threading
from typing import List, Optional
from functools import lru_cache
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Ensure custom package overrides are loaded
# custom_pkg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "custom_packages")
# if os.path.exists(custom_pkg_path) and custom_pkg_path not in sys.path:
#     sys.path.insert(0, custom_pkg_path)

load_dotenv(override=True)

from utils.helpers import get_logger, sanitize_input, load_and_split_files

logger = get_logger("agent")

# ── AI Provider Setup ────────────────────────────────────────────────────────
def detect_provider():
    """Detect which AI provider is configured based on available environment variables."""
    nvidia_key = os.getenv("NVIDIA_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    
    if nvidia_key:
        if nvidia_key.startswith("AIzaSy"):
            if not google_key:
                os.environ["GOOGLE_API_KEY"] = nvidia_key
            return "google"
        return "nvidia"
    elif google_key:
        return "google"
    elif os.getenv("OPENAI_API_KEY"):
        return "openai"
    elif os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"):
        return "huggingface"
    else:
        return None

def validate_api_key():
    """Validates that at least one supported API key is set."""
    provider = detect_provider()
    if not provider:
        return False, "No valid API key found. Please set NVIDIA_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY, or HUGGINGFACEHUB_ACCESS_TOKEN in .env"
    return True, None

def get_llm(temperature=0.1, max_retries=0, timeout=15, **kwargs):
    """Returns the appropriate ChatModel based on the available API keys."""
    provider = detect_provider()
    
    if provider == "nvidia":
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        try:
            return ChatNVIDIA(
                model="meta/llama3-70b-instruct", 
                temperature=temperature,
                max_retries=max_retries,
                timeout=timeout,
                **kwargs
            )
        except TypeError:
            return ChatNVIDIA(
                model="meta/llama3-70b-instruct", 
                temperature=temperature,
                **kwargs
            )
        
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        try:
            return ChatGoogleGenerativeAI(
                model="gemini-3.6-flash", 
                temperature=temperature,
                max_retries=max_retries,
                timeout=timeout,
                **kwargs
            )
        except TypeError:
            return ChatGoogleGenerativeAI(
                model="gemini-3.6-flash", 
                temperature=temperature,
                **kwargs
            )
        
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            temperature=temperature,
            max_retries=max_retries,
            timeout=timeout,
            **kwargs
        )
        
    elif provider == "huggingface":
        from langchain_community.llms import HuggingFaceEndpoint
        try:
            return HuggingFaceEndpoint(
                repo_id="mistralai/Mistral-7B-Instruct-v0.2", 
                temperature=max(temperature, 0.01),
                max_new_tokens=1024,
                timeout=timeout,
                **kwargs
            )
        except TypeError:
            return HuggingFaceEndpoint(
                repo_id="mistralai/Mistral-7B-Instruct-v0.2", 
                temperature=max(temperature, 0.01),
                max_new_tokens=1024,
                **kwargs
            )
    else:
        raise ValueError("Cannot initialize LLM: No valid provider configured.")

def get_embeddings(model_type="passage"):
    """Returns the appropriate Embeddings model based on available API keys."""
    provider = detect_provider()
    
    if provider == "nvidia":
        from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
        try:
            return NVIDIAEmbeddings(model="NV-Embed-QA", model_type=model_type)
        except TypeError:
            return NVIDIAEmbeddings(model="NV-Embed-QA")
            
    elif provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        
    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings()
        
    elif provider == "huggingface":
        from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
        return HuggingFaceInferenceAPIEmbeddings(
            api_key=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    else:
        raise ValueError("Cannot initialize Embeddings: No valid provider configured.")

# ── Module-level Singletons (replacing st.cache_resource) ────────────────────
_singleton_lock = threading.Lock()
_passage_embeddings = None
_query_embeddings = None
_llm_instance = None

def _get_passage_embeddings():
    global _passage_embeddings
    if _passage_embeddings is None:
        with _singleton_lock:
            if _passage_embeddings is None:
                _passage_embeddings = get_embeddings("passage")
    return _passage_embeddings

def _get_query_embeddings():
    global _query_embeddings
    if _query_embeddings is None:
        with _singleton_lock:
            if _query_embeddings is None:
                _query_embeddings = get_embeddings("query")
    return _query_embeddings

def _get_llm():
    global _llm_instance
    if _llm_instance is None:
        with _singleton_lock:
            if _llm_instance is None:
                from langchain_core.globals import set_llm_cache
                from langchain_core.caches import InMemoryCache
                set_llm_cache(InMemoryCache())
                _llm_instance = get_llm(
                    temperature=0.1,
                    max_retries=0,
                    timeout=15,
                )
    return _llm_instance

# ── FAISS Vector Store Building & Retrieval ──────────────────────────────────
_vector_store_cache = {}
_vs_cache_lock = threading.Lock()

def _build_cached_vector_store(files_hash, splits):
    """Builds and caches the FAISS vector store database."""
    with _vs_cache_lock:
        if files_hash in _vector_store_cache:
            return _vector_store_cache[files_hash]
    
    from langchain_community.vectorstores import FAISS
    passage_embeddings = _get_passage_embeddings()

    # Dimension checks for NVIDIA
    test_emb = passage_embeddings.embed_query("verification")
    if detect_provider() == "nvidia" and len(test_emb) != 4096:
        raise ValueError(
            f"NVIDIA nv-embed-v1 dimension mismatch: expected 4096, got {len(test_emb)}"
        )

    vector_store = FAISS.from_documents(splits, passage_embeddings)

    if detect_provider() == "nvidia":
        query_embeddings = _get_query_embeddings()
        if hasattr(vector_store, "_embedding_function"):
            vector_store._embedding_function = query_embeddings.embed_query
        
    logger.info("FAISS vector store built successfully.")
    
    with _vs_cache_lock:
        _vector_store_cache[files_hash] = vector_store
    
    return vector_store

def get_retriever(vector_store, k=6):
    """Returns a retriever from the given vector store."""
    return vector_store.as_retriever(search_kwargs={"k": k})

def search_documents(query: str, retriever) -> str:
    """Retrieves relevant document chunks and returns a formatted string."""
    try:
        docs = retriever.invoke(query)
    except Exception as e:
        logger.error(f"Document retrieval failed: {e}")
        return "Error: Could not search the uploaded documents. Please try again."

    if not docs:
        return "No relevant information found in the uploaded documents."

    results = []
    for doc in docs:
        source = doc.metadata.get("source_file", "Unknown")
        results.append(f"[Source: {source}]\n{doc.page_content}")

    return "RETRIEVED DOCUMENT CONTENT:\n\n" + "\n\n---\n\n".join(results)

# ── Structured Document Extraction ───────────────────────────────────────────

class ResumeAnalysis(BaseModel):
    name: str = Field(description="Full name of the candidate")
    email: str = Field(description="Email address")
    phone: str = Field(description="Phone number")
    location: str = Field(description="Location or address")
    education: List[str] = Field(description="List of degrees and colleges")
    skills: List[str] = Field(description="List of all skills")
    projects: List[str] = Field(description="List of project names and short descriptions")
    experience: List[str] = Field(description="List of internships or work experience")
    certifications: List[str] = Field(description="List of certifications")
    links: List[str] = Field(description="GitHub, LinkedIn, or portfolio links")
    candidate_overview: str = Field(description="A brief 2-3 sentence overview")
    key_skills_identified: List[str] = Field(description="Top 3-5 key skills identified")
    strengths: List[str] = Field(description="Top strengths of the candidate")
    missing_skills: List[str] = Field(description="Missing skills or improvement areas")
    recommended_roles: List[str] = Field(description="Recommended job roles")
    resume_score: int = Field(description="Overall resume score from 0 to 100")

class LoanApplicationAnalysis(BaseModel):
    applicant_name: str = Field(description="Applicant's full name")
    age: Optional[int] = Field(description="Applicant's age if mentioned")
    employment_status: str = Field(description="Employment status")
    income: Optional[float] = Field(description="Monthly or annual income")
    credit_score: Optional[int] = Field(description="Credit score or CIBIL score")
    loan_amount: Optional[float] = Field(description="Requested loan amount")
    loan_purpose: str = Field(description="Purpose of the loan")
    existing_liabilities: List[str] = Field(description="Any existing debts, loans, or EMIs")
    eligibility_summary: str = Field(description="Summary of eligibility")
    factors_supporting_approval: List[str] = Field(description="Factors supporting approval")
    risk_factors: List[str] = Field(description="Risk factors or red flags")
    loan_recommendation: str = Field(description="Final recommendation: Approve, Reject, or Need More Info")

class BankPolicyAnalysis(BaseModel):
    eligibility_rules: List[str] = Field(description="General eligibility rules")
    credit_score_requirements: str = Field(description="Credit score requirements")
    income_requirements: str = Field(description="Income requirements")
    loan_limits: str = Field(description="Minimum and maximum loan limits")
    approval_conditions: List[str] = Field(description="Specific conditions for approval")
    policy_summary: str = Field(description="A brief summary of the policy document")
    key_requirements: List[str] = Field(description="The most critical requirements")
    important_restrictions: List[str] = Field(description="Important restrictions or exclusions")

def _invoke_analyzer_with_retry(chain, inputs, max_retries=3, base_wait=5.0):
    """Invokes structured extractor with backoff for rate limits."""
    for attempt in range(max_retries):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "resource_exhausted" in error_msg or "quota" in error_msg:
                if attempt == max_retries - 1:
                    raise
                wait_time = base_wait * (2 ** attempt)
                logger.warning(f"Rate limit hit. Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise

def analyze_document(text: str, filename: str) -> dict:
    """Classifies document content and extracts relevant structured data."""
    if not text or len(text.strip()) == 0:
        return {"type": "ERROR", "error": "Empty text provided."}

    truncated_text = text[:15000]
    llm = _get_llm()

    classification_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert document classifier for a Bank Loan Approval System. Classify the following document text into EXACTLY ONE of these categories: IRRELEVANT (if the document is a Resume, CV, academic syllabus, code file, book chapter, or any document not related to banking policy or a loan application), LOAN_APPLICATION (if it contains personal financial information, credit score, income details for loan assessment), BANK_POLICY (if it describes loan criteria, interest rate matrices, or bank guidelines). Respond ONLY with the category name: IRRELEVANT, LOAN_APPLICATION, or BANK_POLICY."),
        ("human", "Document text:\n\n{text}")
    ])
    
    try:
        classification_result = _invoke_analyzer_with_retry(classification_prompt | llm, {"text": truncated_text})
        doc_type = classification_result.content.strip().upper()
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        doc_type = "IRRELEVANT"

    logger.info(f"Classified document '{filename}' as {doc_type}")

    try:
        if "IRRELEVANT" in doc_type or "RESUME" in doc_type:
            return {
                "type": "IRRELEVANT",
                "filename": filename,
                "data": {
                    "warning": f"⚠️ Irrelevant Document Detected: The uploaded file '{filename}' appears to be a Resume, Syllabus, or another non-financial document. To evaluate loan eligibility, I require relevant financial documents such as Salary Slips, Bank Statements, ITR, or Identity Proofs. Please upload the correct documents."
                }
            }
        elif doc_type == "LOAN_APPLICATION":
            extractor = llm.with_structured_output(LoanApplicationAnalysis)
            result = _invoke_analyzer_with_retry(extractor, f"Extract all relevant information and evaluate this loan application:\n\n{truncated_text}")
            return {"type": "LOAN_APPLICATION", "filename": filename, "data": result.dict()}
        elif doc_type == "BANK_POLICY":
            extractor = llm.with_structured_output(BankPolicyAnalysis)
            result = _invoke_analyzer_with_retry(extractor, f"Extract all relevant information and summarize this bank policy document:\n\n{truncated_text}")
            return {"type": "BANK_POLICY", "filename": filename, "data": result.dict()}
        else:
            return {
                "type": "IRRELEVANT",
                "filename": filename,
                "data": {
                    "warning": f"⚠️ Irrelevant Document Detected: The uploaded file '{filename}' is not recognized as a financial or policy document."
                }
            }
    except Exception as e:
        logger.error(f"Extraction failed for {filename}: {e}")
        return {"type": "ERROR", "filename": filename, "error": str(e)}

# ── Constants & Configs ──────────────────────────────────────────────────────
MAX_CHAT_HISTORY_TURNS = 6
MAX_RETRIES = 2
MIN_REQUEST_INTERVAL = 1.0

_last_api_call_time = 0.0

def _throttle():
    """Throttle outgoing API calls."""
    global _last_api_call_time
    now = time.time()
    elapsed = now - _last_api_call_time
    if elapsed < MIN_REQUEST_INTERVAL:
        wait = MIN_REQUEST_INTERVAL - elapsed
        time.sleep(wait)
    _last_api_call_time = time.time()

def _extract_retry_delay(error_msg: str) -> float:
    error_str = error_msg.lower()
    match = re.search(r"retry in (\d+(\.\d+)?)s", error_str)
    if match:
        return float(match.group(1))
    match2 = re.search(r"retrydelay':\s*'(\d+)s?'", error_str)
    if match2:
        return float(match2.group(1))
    return 10.0

# ── Response Cache ───────────────────────────────────────────────────────────
_response_cache = {}
_CACHE_MAX_SIZE = 50

def _cache_key(user_input: str, active_mode: str) -> str:
    normalized = user_input.strip().lower()
    return hashlib.md5(f"{active_mode}:{normalized}".encode()).hexdigest()

def _get_cached_response(user_input: str, active_mode: str):
    key = _cache_key(user_input, active_mode)
    return _response_cache.get(key)

def _set_cached_response(user_input: str, active_mode: str, response: str):
    if len(_response_cache) >= _CACHE_MAX_SIZE:
        oldest_key = next(iter(_response_cache))
        del _response_cache[oldest_key]
    key = _cache_key(user_input, active_mode)
    _response_cache[key] = response

# ── Modes ────────────────────────────────────────────────────────────────────
MODE_GENERAL = "General Guidance"
MODE_DOCUMENT = "Document-Based Analysis"

# ── Prompts ──────────────────────────────────────────────────────────────────
DOCUMENT_SYSTEM_PROMPT = """You are a strict, professional Bank Loan Approval Officer. You NEVER evaluate resumes, coding skills, or act as an HR recruiter. Your job is to analyze the retrieved document context and assess loan eligibility or answer banking questions strictly from a financial perspective.

STRICT RELEVANCE VALIDATION (GATEKEEPER):
The very first step of your analysis MUST be validating the document type. 
If the uploaded document context is a Resume, CV, academic syllabus, code file, book, or any non-financial document, you must immediately STOP analysis and return this exact warning message, nothing else:
"⚠️ Irrelevant Document Detected: This document appears to be a [Resume/CV/Syllabus/etc.]. To evaluate loan eligibility, I require relevant financial documents such as Salary Slips, Bank Statements, ITR, or Identity Proofs. Please upload the documents."

STRICT CREDIT HISTORY MAPPING RULES & CHAIN-OF-THOUGHT (UNBREAKABLE):
1. FORCE STEP-BY-STEP EXTRACTION: Before generating any decision or output, you MUST perform a step-by-step extraction. You MUST explicitly scan the user's input and context for the exact phrases "Credit History: 0" (or 0.0) or "Credit History: 1" (or 1.0) and identify the exact value of Credit History.
2. THE UNBREAKABLE GOLDEN RULE:
   - IF the input/context contains "Credit History: 0" (or 0.0) anywhere, you MUST IMMEDIATELY map this to a "Poor Credit Score (below 600)" (specifically 550) and the final loan Status MUST ALWAYS BE 'Rejected'.
   - You are STRICTLY FORBIDDEN from hallucinating, assuming, or generating any credit score above 600 if "Credit History: 0" is present. No exceptions.
   - IF the input/context contains "Credit History: 1" (or 1.0), map it to a Good Credit Score (e.g., 750) and evaluate normally.
3. OUTPUT FORMAT ENFORCEMENT: If "Credit History: 0" is detected, your reasoning must explicitly state: "The user has a Credit History of 0, which translates to a poor credit score. Therefore, the loan is rejected."

ELIGIBILITY ASSESSMENT LOGIC:
If the user asks "Am I eligible for a loan?" or if you are evaluating loan eligibility:
- If relevant financial data (Income, Credit Score, Debt-to-Income ratio, or Employment history) is found in the context: Provide a professional approval/rejection analysis with clear reasons.
- If this necessary financial data is missing or incomplete: State clearly, and nothing else:
  "I cannot determine your loan eligibility because the provided document lacks necessary financial information (e.g., monthly income, credit history). To check your eligibility, please upload your recent financial documents."

OUTPUT FORMATTING:
If the document is relevant and contains financial data, always format your response using these exact Markdown headers:

### 💼 Financial Profile
(Provide a brief summary of the extracted financial metrics.)

### 💰 Income Verification
(State if the income is verified, the source, and if it meets baseline bank policies.)

### ⚠️ Risk Factors
(Detail any risks like low credit score, high DTI ratio, or missing key parameters.)

### 📊 Loan Eligibility Status
(State the final decision: Approved, Rejected, Conditionally Approved, or Indeterminate due to missing data.)

IMPORTANT RULES:
- Base your answers ONLY on the retrieved document context provided.
- Do NOT invent or assume any financial metrics if they are not explicitly present.
- Maintain a highly formal, strict bank officer persona.

FORMATTING & COMPLIANCE RULES:
1. BITE-SIZED READABILITY: Ensure the Financial Profile and Eligibility Status sections are extremely concise and use bullet points.
2. LEGAL DISCLAIMER: You MUST append this exact text at the very bottom: "*Disclaimer: This automated document analysis is for preliminary informational purposes. Final loan sanctioning requires manual bank verification.*"
"""

DOCUMENT_HUMAN_TEMPLATE = """Retrieved Document Context:

{context}

User Question:

{question}"""

GENERAL_SYSTEM_PROMPT = """You are an expert Loan and Banking Assistant. Your job is to answer general banking knowledge questions, loan eligibility criteria, typical interest rates, credit score rules, and application steps.

Provide direct, professional, and clear answers using your general core knowledge. Do not mention any uploaded documents or context unless explicitly asked.

STRICT CREDIT HISTORY MAPPING RULES & CHAIN-OF-THOUGHT (UNBREAKABLE):
1. FORCE STEP-BY-STEP EXTRACTION: Before generating any decision or output, you MUST perform a step-by-step extraction. You MUST explicitly scan the user's input for the exact phrases "Credit History: 0" (or 0.0) or "Credit History: 1" (or 1.0) and identify the exact value of Credit History.
2. THE UNBREAKABLE GOLDEN RULE:
   - IF the input contains "Credit History: 0" (or 0.0) anywhere, you MUST IMMEDIATELY map this to a "Poor Credit Score (below 600)" (specifically 550) and the final Status MUST ALWAYS BE 'Rejected'.
   - You are STRICTLY FORBIDDEN from hallucinating, assuming, or generating any credit score above 600 if "Credit History: 0" is present. No exceptions.
   - IF the input contains "Credit History: 1" (or 1.0), map it to a Good Credit Score (e.g., 750) and evaluate normally.
3. OUTPUT FORMAT ENFORCEMENT: If "Credit History: 0" is detected, your reasoning must explicitly state: "The user has a Credit History of 0, which translates to a poor credit score. Therefore, the loan is rejected."

When discussing credit scores, use these standard policy benchmarks:
- Minimum credit score required: 650
- 650-700: May be approved with higher interest rates
- Above 750: Eligible for premium rates and higher loan amounts
- Below 600: Generally not eligible for unsecured loans

When discussing interest rates, use these typical ranges:
- Base rate: 8.5% per annum
- Credit score above 750: Base rate
- Credit score 700-750: Base rate + 1%
- Credit score 650-700: Base rate + 2.5%

CRITICAL INSTRUCTIONS FOR USER INTERACTION:
1. Always answer the user's question directly and fully first using your baseline knowledge.
2. If the user asks a general question about loan eligibility or credit score criteria, explain the general requirements first.
3. Never respond to a general query with only a question asking for details.
4. Avoid getting stuck in a counter-questioning loop.
5. Act as a math calculator when the user provides numbers for EMI, interest, or term calculations.
6. Strictly use the provided `chat_history` for context awareness.
7. CALCULATIONS & MATH: If the user provides numbers and asks for a calculation, execute the mathematical formula step-by-step.
8. CONTEXT AWARENESS: You are a conversational AI. Use the provided chat history to remember context.
9. INDEPENDENT EVALUATION: If evaluating a new applicant's eligibility, evaluate them strictly on the current prompt's numbers.

FORMATTING & COMPLIANCE RULES:
1. BITE-SIZED READABILITY: Always format your answers using short, scannable bullet points and bold headings.
2. CALL TO ACTION (CTA): Always end your response by guiding the user to the next step.
3. LEGAL DISCLAIMER: You MUST append this exact text at the very bottom: "*Disclaimer: This is AI-generated guidance based on standard banking policies. Final approval is subject to official bank verification.*"
"""

# ── RAG Helpers ───────────────────────────────────────────────────────────────
def _build_document_analysis_markdown(document_analyses) -> str:
    if not document_analyses:
        return ""
    sections = ["=== AUTOMATIC DOCUMENT ANALYSIS (Pre-Extracted) ===", ""]
    for analysis in document_analyses:
        doctype = analysis.get("type", "UNKNOWN")
        fname = analysis.get("filename", "Document")
        data = analysis.get("data", {})
        sections.append(f"**Document:** {fname}")
        sections.append(f"**Detected Type:** {doctype}")
        sections.append("**Extracted Fields:**")
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list):
                    value_text = ", ".join(str(item) for item in value) if value else "N/A"
                else:
                    value_text = str(value) if value is not None else "N/A"
                label = key.replace("_", " ").title()
                sections.append(f"- {label}: {value_text}")
        else:
            sections.append(f"- {data}")
        sections.append("")
    return "\n".join(sections).strip()

def _build_rag_context(question: str, retriever, document_analyses=None) -> str:
    parts = []
    if retriever is not None:
        parts.append(search_documents(question, retriever))
    analysis_md = _build_document_analysis_markdown(document_analyses)
    if analysis_md:
        parts.append(analysis_md)
    return "\n\n".join(parts) if parts else "No document context available."

def _build_document_agent_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", DOCUMENT_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", DOCUMENT_HUMAN_TEMPLATE),
    ])

# ── Agent Creation ────────────────────────────────────────────────────────────
_general_agent = None
_general_agent_lock = threading.Lock()

def create_general_agent():
    global _general_agent
    if _general_agent is None:
        with _general_agent_lock:
            if _general_agent is None:
                llm = _get_llm()
                prompt = ChatPromptTemplate.from_messages([
                    ("system", GENERAL_SYSTEM_PROMPT),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}")
                ])
                chain = prompt | llm
                _general_agent = (chain, llm, None, None)
    return _general_agent

def create_loan_agent(retriever=None, document_analyses=None):
    """Instantiate a loan analysis agent chain."""
    llm = _get_llm()
    prompt = _build_document_agent_prompt()
    chain = prompt | llm
    return chain, llm, retriever, document_analyses

def truncate_chat_history(chat_history: list, max_turns: int = MAX_CHAT_HISTORY_TURNS) -> list:
    max_messages = max_turns * 2
    if len(chat_history) > max_messages:
        return chat_history[-max_messages:]
    return chat_history

def _convert_chat_history(history_dicts: list) -> list:
    """Convert list of {role, content} dicts to LangChain message objects."""
    messages = []
    for msg in history_dicts:
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg.get("role") == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages

# ── Retry Logic ──────────────────────────────────────────────────────────────
def invoke_with_retry(runnable, input_data, max_retries=3):
    _throttle()
    attempt = 0
    base_wait = 2.0
    while attempt <= max_retries:
        try:
            return runnable.invoke(input_data)
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = any(term in error_str for term in ["429", "resource_exhausted", "quota", "rate limit"])
            is_network_error = any(term in error_str for term in ["11001", "getaddrinfo", "connection", "timeout", "ssl"])
            
            if is_rate_limit:
                attempt += 1
                if attempt > max_retries:
                    return None
                wait_time = _extract_retry_delay(str(e)) + 1.5
                logger.warning(f"Rate limit hit. Retrying in {wait_time}s... (Attempt {attempt}/{max_retries})")
                time.sleep(wait_time)
            elif is_network_error:
                attempt += 1
                if attempt > max_retries:
                    return None
                wait_time = 2.0 * (2 ** (attempt - 1))
                logger.warning(f"Network error. Retrying in {wait_time}s... (Attempt {attempt}/{max_retries})")
                time.sleep(wait_time)
            else:
                logger.error(f"Non-retryable error: {e}")
                raise

def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(content)

# ── Main Interfaces called by API Routes ─────────────────────────────────────
def get_response(agent_objects, user_input: str, chat_history_dicts: list, active_mode: str = MODE_GENERAL) -> str:
    chain, base_llm, retriever, document_analyses = agent_objects
    is_document_mode = active_mode == MODE_DOCUMENT

    user_input = sanitize_input(user_input)
    if not user_input:
        return "Please enter a valid question."

    if is_document_mode and retriever is None:
        return "⚠️ **Document-Based Analysis requires uploaded documents.**\nPlease upload loan-related files."

    chat_history = _convert_chat_history(chat_history_dicts)
    safe_history = truncate_chat_history(chat_history)
    
    try:
        if is_document_mode:
            context = _build_rag_context(user_input, retriever, document_analyses)
            response = invoke_with_retry(chain, {"context": context, "question": user_input, "chat_history": safe_history})
        else:
            response = invoke_with_retry(chain, {"input": user_input, "chat_history": safe_history})

        if response is None:
            return "Our system is currently handling high traffic. Please wait a moment and try again."

        result_text = _extract_text(response.content) if hasattr(response, "content") else str(response)
        _set_cached_response(user_input, active_mode, result_text)
        return result_text
    except Exception as e:
        logger.error(f"get_response error: {e}")
        error_str = str(e).lower()
        if any(term in error_str for term in ["429", "resource_exhausted", "quota", "rate limit"]):
            return "Our system is currently handling high traffic. Please wait a moment and try again."
        return f"⚠️ An error occurred: {str(e)}"

def get_response_stream(agent_objects, user_input: str, chat_history_dicts: list, active_mode: str = MODE_GENERAL):
    """Generator that yields text chunks for streaming responses."""
    chain, base_llm, retriever, document_analyses = agent_objects
    is_document_mode = active_mode == MODE_DOCUMENT

    user_input = sanitize_input(user_input)
    if not user_input:
        yield "Please enter a valid question."
        return

    if is_document_mode and retriever is None:
        yield "⚠️ **Document-Based Analysis requires uploaded documents.**\nPlease upload loan-related files."
        return

    chat_history = _convert_chat_history(chat_history_dicts)
    safe_history = truncate_chat_history(chat_history)
    
    _throttle()
    if is_document_mode:
        payload = {
            "context": _build_rag_context(user_input, retriever, document_analyses),
            "question": user_input,
            "chat_history": safe_history
        }
    else:
        payload = {"input": user_input, "chat_history": safe_history}
    
    collected = []
    attempt = 0
    max_retries = 3
    while attempt <= max_retries:
        try:
            stream_iterator = chain.stream(payload)
            first_chunk = next(stream_iterator, None)
            
            if first_chunk is not None:
                text = _extract_text(first_chunk.content) if hasattr(first_chunk, "content") else ""
                if text:
                    collected.append(text)
                    yield text
                for chunk in stream_iterator:
                    text = _extract_text(chunk.content) if hasattr(chunk, "content") else ""
                    if text:
                        collected.append(text)
                        yield text
            break
        except Exception as e:
            error_str = str(e).lower()
            if any(term in error_str for term in ["429", "resource_exhausted", "quota", "rate limit"]):
                attempt += 1
                if attempt > max_retries:
                    yield "⏳ **High Traffic Alert:** Our servers are currently handling a large volume of requests. Please wait about 1 minute and try again."
                    return
                wait_time = _extract_retry_delay(str(e)) + 1.5
                logger.warning(f"Rate limit hit in stream. Retrying in {wait_time}s... (Attempt {attempt}/{max_retries})")
                time.sleep(wait_time)
            else:
                logger.error(f"Streaming error: {e}")
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    yield "⏳ **High Traffic Alert:** Please wait about 1 minute and try again."
                else:
                    yield f"⚠️ An error occurred: {error_msg}"
                return
                
    if collected:
        _set_cached_response(user_input, active_mode, "".join(collected))

def process_documents(files_data: list) -> dict:
    """
    Process uploaded documents: load, split, build FAISS, extract structured analysis.
    files_data: list of {"filename": str, "content": bytes}
    """
    if not files_data:
        return None

    # Generate a stable cache key
    hasher = hashlib.md5()
    for f in sorted(files_data, key=lambda x: x["filename"]):
        hasher.update(f["filename"].encode('utf-8'))
        hasher.update(str(len(f["content"])).encode('utf-8'))
    files_hash = hasher.hexdigest()

    splits, file_names, raw_texts = load_and_split_files(files_data)

    if not splits:
        return None

    # Retrieve or build cached vector store
    vector_store = _build_cached_vector_store(files_hash, splits)
    retriever = get_retriever(vector_store)

    # Document intelligence extraction
    document_analyses = []
    for fname, text in raw_texts.items():
        analysis = analyze_document(text, fname)
        document_analyses.append(analysis)

    doc_agent = create_loan_agent(retriever, document_analyses)

    return {
        "doc_agent": doc_agent,
        "file_names": file_names,
        "document_analyses": document_analyses
    }

# ── Validate API Key on load ─────────────────────────────────────────────────
_key_ok, _key_error = validate_api_key()
if not _key_ok:
    logger.error(_key_error)
