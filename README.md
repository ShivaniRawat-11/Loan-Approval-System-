# 🏦 AI Loan Approval System

A production-ready RAG-based Loan Approval System built with LangChain, Streamlit, and Google Gemini.

## Features

- **Document Upload** — Upload PDF and TXT loan documents (policies, applications, credit reports, income proofs)
- **RAG Pipeline** — Documents are chunked, embedded, and indexed in FAISS for retrieval
- **ML Prediction** — Trained sklearn model predicts loan approval with confidence scores
- **Policy Enforcement** — Decisions are strictly based on retrieved document content
- **Structured Output** — Professional format: Applicant Summary → Policy Evaluation → Final Decision

## Project Structure

```
loan_approval_agent/
├── app.py                  # Streamlit UI with document upload
├── agent.py                # LangChain agent with RAG + tools
├── vector_store.py         # Dynamic FAISS vector store from uploads
├── tools/
│   └── loan_tool.py        # ML prediction tool (sklearn)
├── ML model/
│   ├── ML_part.py          # Model training script
│   ├── loan_model.pkl      # Trained model
│   ├── scaler.pkl          # Feature scaler
│   ├── le_education.pkl    # Label encoder
│   ├── ohe.pkl             # One-hot encoder
│   └── feature_names.pkl   # Feature names
├── data/
│   └── bank_policy.txt     # Sample bank policy document
├── requirements.txt
├── .env                    # GOOGLE_API_KEY goes here
└── README.md
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API Key

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Run the Application

```bash
streamlit run app.py
```

## Usage

1. **Upload Documents** — Use the sidebar to upload loan policy PDFs, application forms, credit reports, etc.
2. **Process** — Click "Process Documents" to ingest and index them
3. **Ask Questions** — Use the chat to ask about policies or evaluate loan eligibility
4. **Get Decisions** — Receive structured, policy-driven loan decisions

## Sample Prompts

- "What is the minimum credit score required?"
- "Check eligibility: income $5000, loan $20000, credit score 720"
- "What documents are needed for a home loan?"
- "Why might a loan be rejected?"

## Tech Stack

| Component | Technology |
|-----------|-----------|
| UI | Streamlit |
| LLM | Google Gemini 2.5 Flash |
| RAG | LangChain + FAISS |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| ML Model | scikit-learn |
| Document Loading | PyPDF, TextLoader |
