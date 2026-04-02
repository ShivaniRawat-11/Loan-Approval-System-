import os
import sys
import time
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from dotenv import load_dotenv

from tools.loan_tool import predict_loan_approval
from vector_store import search_documents

load_dotenv()

# ── System Prompt: Document-Based Evaluation (Mode 2) ──────────────────────
DOCUMENT_SYSTEM_PROMPT = """You are an AI Loan Approval Assistant operating in DOCUMENT-BASED ANALYSIS mode.

Your task is to evaluate loan eligibility strictly based on the information extracted from uploaded documents and loan policy rules.

IMPORTANT RULES:
1. Do NOT assume or invent any applicant information.
2. Only use information explicitly present in retrieved documents.
3. If required financial or applicant details are missing or documents are unrelated to loan evaluation, DO NOT approve or reject the loan.
4. In such cases, return a clear and user-friendly response stating that eligibility cannot be determined.
5. Your response must be concise, professional, and easy for non-technical users to understand.
6. Do NOT use default values for any missing information.

REQUIRED OUTPUT FORMAT:

If sufficient information is NOT available, respond EXACTLY in the following structure:

⚠️ Loan eligibility cannot be determined.

Reason:
Clearly explain that the uploaded documents do not contain the financial or applicant information required for loan assessment.

Please upload:
• Income proof (salary slips or tax returns)
• Bank statements
• Credit report or credit score details
• Employment verification or job details
• Loan amount and loan purpose information

If sufficient information IS available, then provide:

✅ Loan Eligibility Decision

Applicant Summary:
- Age:
- Credit Score:
- Monthly Income:
- Employment Status:
- Debt-to-Income Ratio:
- Loan Amount Requested:
- Loan Purpose:

Policy Evaluation:
Explain how each policy requirement is satisfied or failed.

Final Decision:
- Loan Status: APPROVED / REJECTED / NEED_MORE_INFORMATION
- Clear Reasoning:
Explain decision strictly using policy-based reasoning.

Always prioritize clarity, correctness, and safety over giving a decision.
"""

# ── System Prompt: General Conversation (Mode 1) ───────────────────────────
GENERAL_SYSTEM_PROMPT = """You are an AI Loan Assistant operating in GENERAL GUIDANCE mode.

No documents have been uploaded yet, so you should answer the user's questions using your general knowledge about loans, banking, credit, eligibility rules, and financial terms.

IMPORTANT RULES:
1. Provide helpful, accurate, and easy-to-understand explanations.
2. Be concise and professional.
3. If the user asks about their personal loan eligibility, politely explain that you need documents to perform a personalized evaluation, and suggest they upload documents via the sidebar.
4. Never invent applicant-specific financial data.
5. Clearly indicate that your answers are general guidance and not based on any uploaded documents.
6. You may discuss topics like: types of loans, interest rates, credit score ranges, documentation requirements, eligibility criteria, EMI calculations, debt-to-income ratios, and general banking terms.

💡 Tip: If the user wants a personalized eligibility assessment, suggest uploading loan-related documents (income proof, bank statements, credit reports, etc.) using the sidebar.
"""


def _get_llm():
    """Shared LLM instance factory."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.1
    )


def create_general_agent():
    """
    Creates a lightweight agent for Mode 1 (no documents).
    No tools are bound — the LLM answers from general knowledge.
    """
    llm = _get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", GENERAL_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    chain = prompt | llm
    # Return in the same tuple shape so get_response works for both modes
    return chain, llm, []


def create_loan_agent(retriever=None):
    """
    Creates the document-based loan agent with tools bound to the LLM.
    If a retriever is provided, creates a dynamic search tool bound to it.
    """
    llm = _get_llm()

    # Build tool list
    agent_tools = [predict_loan_approval]

    if retriever is not None:
        # Create a dynamic search tool bound to this session's retriever
        @tool("search_uploaded_documents")
        def search_uploaded_documents(query: str) -> str:
            """Search the uploaded loan documents for relevant information.
            Use this for questions about bank policies, applicant details,
            income proofs, credit reports, or any loan-related information.

            Args:
                query: Question or topic to search for in the documents

            Returns:
                Relevant content retrieved from uploaded documents
            """
            return search_documents(query, retriever)

        agent_tools.append(search_uploaded_documents)

    llm_with_tools = llm.bind_tools(agent_tools)

    prompt = ChatPromptTemplate.from_messages([
        ("system", DOCUMENT_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    chain = prompt | llm_with_tools
    return chain, llm, agent_tools


def invoke_with_retry(runnable, input_data, max_retries=3):
    """
    Invokes a runnable with retry logic for RESOURCE_EXHAUSTED errors.
    Parses the wait time from the error message.
    """
    attempt = 0
    while attempt <= max_retries:
        try:
            return runnable.invoke(input_data)
        except Exception as e:
            error_str = str(e)
            if "RESOURCE_EXHAUSTED" in error_str:
                attempt += 1
                if attempt > max_retries:
                    raise e

                match = re.search(r"Please retry in (\d+(\.\d+)?)s", error_str)
                if match:
                    wait_time = float(match.group(1)) + 1.0
                    print(f"⚠️ Rate limit hit. Waiting {wait_time:.1f}s (retry {attempt}/{max_retries})...", file=sys.stderr)
                    time.sleep(wait_time)
                else:
                    print(f"⚠️ Rate limit hit. Waiting 10s (retry {attempt}/{max_retries})...", file=sys.stderr)
                    time.sleep(10)
            else:
                raise e


def _extract_text(content) -> str:
    """Safely extract text from LLM response content (may be str or list of parts)."""
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


def get_response(agent_objects, user_input: str, chat_history: list) -> str:
    chain, base_llm, agent_tools = agent_objects

    # Build a name→tool lookup for dispatching
    tool_lookup = {t.name: t for t in agent_tools}

    try:
        # Step 1: Initial call to decide tool usage
        response = invoke_with_retry(chain, {"input": user_input, "chat_history": chat_history})

        # Step 2: Handle tool calls if any
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_outputs = []
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']

                if tool_name in tool_lookup:
                    result = tool_lookup[tool_name].invoke(tool_args)
                else:
                    result = f"Unknown tool: {tool_name}"

                tool_outputs.append(f"Tool '{tool_name}' output: {str(result)}")

            # Step 3: Re-invoke LLM to explain the tool output using the strict prompt template
            combined_context = f"""
            User's original question: {user_input}

            The tool(s) returned the following results:
            {chr(10).join(tool_outputs)}

            Based on the tool results, provide a response STRICTLY following these instructions:
            {DOCUMENT_SYSTEM_PROMPT}

            IMPORTANT:
            - ONLY use the tool results above.
            - If details are missing, use the "⚠️ Loan eligibility cannot be determined" format.
            """

            final_response = invoke_with_retry(base_llm, [
                HumanMessage(content=combined_context)
            ])
            return _extract_text(final_response.content)

        else:
            return _extract_text(response.content) if hasattr(response, 'content') else str(response)

    except Exception as e:
        return f"Error: {str(e)}"
