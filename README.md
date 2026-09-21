# 🏦 AI Loan Approval System (Decoupled Architecture)

A production-ready RAG-based Loan Approval System. The project has recently been refactored from a monolithic Streamlit app into a modern, decoupled 3-tier architecture using **Next.js** (Frontend), **FastAPI** (Backend), and **SQLite** (Database), all containerized with **Docker**.

## Features

- **JWT Authentication** — Secure login and signup with JSON Web Tokens.
- **Modern UI** — A premium, responsive user interface built with Next.js and Tailwind-like Vanilla CSS.
- **Document Upload** — Upload PDF and TXT loan documents (policies, applications, credit reports, income proofs).
- **RAG Pipeline** — Documents are chunked, embedded, and indexed in FAISS for retrieval.
- **ML Prediction** — Trained sklearn model predicts loan approval with confidence scores.
- **Dockerized Setup** — Fully containerized for seamless cross-platform deployment.

## Project Structure

```text
loan_approval_agent/
├── backend/                # FastAPI Backend Service
│   ├── main.py             # Entry point
│   ├── routes/             # API Endpoints (auth, chat, documents)
│   ├── services/           # LangChain RAG logic & LLM agents
│   ├── tools/              # ML model and sklearn tools
│   ├── utils/              # Security (JWT) and helpers
│   ├── database/           # SQLite DB connection
│   └── requirements.txt    # Python dependencies
├── frontend/               # Next.js Frontend Service
│   ├── src/app/            # App router pages (Login, Dashboard)
│   ├── src/lib/api.js      # API fetch client
│   └── package.json        # Node.js dependencies
├── .env                    # Environment variables (Backend)
├── docker-compose.yml      # Docker orchestration
└── README.md
```

## Setup & Running (Using Docker)

The easiest way to run the application is using Docker Compose. This ensures you don't need to install Python or Node.js directly on your host machine.

### 1. Set API Keys
Create or update the `.env` file in the project root:
```env
NVIDIA_API_KEY=your_nvidia_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
# Note: Provide whichever API keys you configured your LangChain models to use.
```

### 2. Build and Start Containers
Make sure Docker Desktop is running, then execute:
```bash
docker compose up --build -d
```

### 3. Access the App
- **Frontend UI:** [http://localhost:3000](http://localhost:3000)
- **Backend API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🛠 Troubleshooting: "Red Lines" in VS Code

If you open the project in VS Code and see red squiggly lines (e.g., unresolved imports in `frontend/` or `backend/`), **do not worry! There are no actual syntax errors.**

### Why does this happen?
Because the project runs entirely inside Docker, the required local dependencies (`node_modules` for Next.js, and the Python virtual environment for FastAPI) are installed *inside the containers*, not on your host machine. VS Code's local linting tools cannot see them.

### How to fix it (Optional):
If you want IDE autocomplete and linting to work flawlessly on your host machine, you can install the dependencies locally just for the IDE's sake:
1. **Frontend:** Open terminal, `cd frontend`, and run `npm install`.
2. **Backend:** Open terminal, `cd backend`, create a venv `python -m venv venv`, activate it, and run `pip install -r requirements.txt`.

Otherwise, you can safely ignore the red lines, as Docker handles everything successfully during runtime.
