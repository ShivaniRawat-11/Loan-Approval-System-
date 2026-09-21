# AI Loan Approval System Architecture

This document provides a high-level overview of the system architecture for the AI Loan Approval Agent.

## System Components

```mermaid
graph TD
    %% Define Styles
    classDef frontend fill:#3b82f6,stroke:#1d4ed8,stroke-width:2px,color:#fff;
    classDef backend fill:#10b981,stroke:#047857,stroke-width:2px,color:#fff;
    classDef agent fill:#8b5cf6,stroke:#6d28d9,stroke-width:2px,color:#fff;
    classDef db fill:#f59e0b,stroke:#b45309,stroke-width:2px,color:#fff;
    classDef ext fill:#ef4444,stroke:#b91c1c,stroke-width:2px,color:#fff;

    %% Client Layer
    subgraph Client ["Client Layer"]
        UI["Next.js Frontend UI<br/>(React, CSS)"]:::frontend
    end

    %% API Layer
    subgraph APILayer ["API Layer (FastAPI)"]
        Router["API Routers<br/>(/auth, /chat, /documents)"]:::backend
        Auth["Auth Service<br/>(JWT)"]:::backend
    end

    %% Agent Core
    subgraph AgentCore ["Agent & ML Core"]
        AgentService["Agent Service<br/>(LangChain)"]:::agent
        Classifier["Document Classifier"]:::agent
        StructuredExtractor["Structured Extractor<br/>(Pydantic Models)"]:::agent
        RAG["RAG Engine"]:::agent
    end

    %% Data Layer
    subgraph DataLayer ["Data Layer"]
        SQLite[("SQLite DB<br/>(Users, Chat History)")]:::db
        FAISS[("FAISS Vector Store<br/>(Document Embeddings)")]:::db
    end

    %% External APIs
    LLM["Google Gemini API<br/>(gemini-3.6-flash)"]:::ext
    Embedder["Embedding Model API<br/>(passage & query)"]:::ext

    %% Connections
    UI -- "REST / SSE" --> Router
    Router -- "Validates User" --> Auth
    Auth -- "Reads/Writes" --> SQLite
    
    Router -- "Routes Requests" --> AgentService
    AgentService -- "Text Query" --> LLM
    AgentService -- "Builds/Queries" --> FAISS
    FAISS -- "Generates Vectors" --> Embedder
    
    AgentService -- "Classifies Type" --> Classifier
    AgentService -- "Extracts Data" --> StructuredExtractor
    AgentService -- "Retrieves Context" --> RAG
```

## Description

1. **Frontend**: A modern Next.js application providing a chat interface, document upload capabilities, and a dashboard for viewing structured loan evaluations.
2. **Backend**: A Python FastAPI server that handles routing, authentication (JWT), and streaming Server-Sent Events (SSE) back to the client.
3. **Agent Service**: Powered by LangChain, this service manages the core logic:
   - **Document Classification**: Determines if uploaded files are Bank Policies, Loan Applications, or Irrelevant.
   - **Structured Extraction**: Uses Pydantic models to enforce structured JSON output from the LLM.
   - **RAG Engine**: Chunks documents and uses FAISS to retrieve relevant context for answering questions.
4. **Data Layer**: Local SQLite database for persistent user accounts and chat history, and an in-memory FAISS vector store for semantic document search.
5. **External LLM**: Uses Google's Gemini models for intelligent reasoning, text generation, and embeddings.
