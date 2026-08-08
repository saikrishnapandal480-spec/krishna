# AI Interview Agent

An enterprise-grade, personalized technical interview application built for the **31-Day Enterprise AI Engineering Cohort Hackathon**.

The **AI Interview Agent** conducts realistic, multi-turn technical interviews tailored to a candidate's learning history, progress signals, passed missions, and skipped topics from the 31-day Enterprise AI curriculum.

---

## Technical Features

1. **Strict API Specification**: Exposes exact required endpoint `POST /api/interview`.
2. **Session Persistence**: Maintains conversation state across multi-turn requests using `sessionId`.
3. **Adaptive Difficulty**: Evaluates candidate technical reasoning, correctness, and practical depth on every turn to adjust question complexity (`easy` -> `medium` -> `hard`).
4. **Curriculum Coverage**: Ensures at least **8 questions** are asked and at least **4 distinct curriculum days** are covered before concluding.
5. **Personalization**: Analyzes candidate profile datasets (first-try passes vs. multiple attempts vs. skipped topics) to tailor question selection.
6. **Context-Aware Follow-ups**: Generates intelligent follow-ups referencing specific technical concepts from prior candidate responses.
7. **Dual AI Provider**:
   - `MockAIProvider`: Zero-dependency, offline-ready mock provider that simulates an experienced AI lead.
   - `LLMAIProvider`: OpenAI-compatible API provider configurable via environment variables.
8. **Structured Final Feedback**: Concludes with structured feedback matching exact schema (`summary`, `strengths`, `gaps`, `next`).

---

## Project Architecture

```text
SAIKRISHNA FLODER/
├── backend/
│   ├── main.py                  # FastAPI application entry point & static file server
│   ├── api/
│   │   └── interview.py          # POST /api/interview & helper endpoints
│   ├── agents/
│   │   ├── interviewer.py        # Question selection & curriculum day orchestration
│   │   ├── evaluator.py          # Candidate response evaluation
│   │   └── feedback.py           # Final feedback synthesis
│   ├── services/
│   │   ├── curriculum.py         # Curriculum JSON loader & query service
│   │   ├── candidates.py         # Candidate dataset parser & profile analyzer
│   │   ├── sessions.py           # In-memory session manager
│   │   └── ai_provider.py        # BaseAIProvider, MockAIProvider, LLMAIProvider
│   ├── models/
│   │   └── schemas.py            # Pydantic request/response & feedback models
│   ├── data/
│   │   ├── curriculum.json       # 31-Day Enterprise AI Curriculum dataset
│   │   └── candidates.json       # 20 Candidate profiles dataset
│   └── tests/
│       └── test_api.py           # Pytest automated test suite
├── frontend/
│   ├── index.html                # High-performance glassmorphism React web application
│   ├── package.json              # Vite + React + TypeScript configuration
│   └── src/
│       ├── App.tsx               # Main React application
│       └── types.ts              # TypeScript interface definitions
├── implementation_plan.md
└── README.md
```

---

## Environment Variables

Create a `.env` file in the root directory (optional, app defaults to mock mode out-of-the-box):

```env
# AI Provider Choice: 'mock' (default) or 'openai'
AI_PROVIDER=mock

# OpenAI API Settings (Required only if AI_PROVIDER=openai)
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4o-mini
OPENAI_API_BASE=https://api.openai.com/v1
```

---

## Installation & Running

### 1. Backend Setup

Ensure Python 3.10+ is installed.

```powershell
# Install backend dependencies
pip install fastapi uvicorn pydantic pytest httpx

# Run Backend Server
python -m backend.main
```

The backend server starts on `http://127.0.0.1:8000`.

### 2. Frontend Application

The application serves the complete interactive frontend at:
👉 `http://127.0.0.1:8000/`

Alternatively, if running Vite:
```powershell
cd frontend
npm install
npm run dev
```

---

## API Documentation

### `POST /api/interview`

#### 1. Start Interview Request
```json
POST /api/interview
Content-Type: application/json

{
  "sessionId": "abc-123",
  "candidate": {
    "member": {
      "id": "CAND-001",
      "name": "Sarah Johnson",
      "jobRole": "Senior Data Engineer",
      "yearsExperience": 9,
      "education": "MS Computer Science"
    },
    "missions": [
      { "day": 7, "title": "Embeddings Explained", "passed": true, "attempts": 1 }
    ],
    "signals": { "commitDays": 28, "missionsCompleted": 30, "missionsFirstTry": 20 }
  }
}
```

**Response (`done: false`)**:
```json
{
  "reply": "Welcome. Let's begin your interview.\n\nQuestion 1 (Day 7 - Embeddings):\nIn Day 7, you covered text embeddings. Can you explain how textual data is converted into high-dimensional vector representations?",
  "done": false,
  "questionNumber": 1,
  "currentTopic": "Embeddings",
  "curriculumDay": 7,
  "coveredDaysCount": 1,
  "difficulty": "medium"
}
```

#### 2. Conversation Turn Request
```json
POST /api/interview
Content-Type: application/json

{
  "sessionId": "abc-123",
  "message": "We convert text into vector embeddings using Sentence Transformers and store them in ChromaDB using HNSW indexes for cosine similarity search."
}
```

**Response (`done: false`)**:
```json
{
  "reply": "Question 2 (Day 8 - Vector Databases):\nIn your response regarding Embeddings, you referenced 'vector'. Building on that: How do indexing algorithms like HNSW balance search recall speed with memory overhead?",
  "done": false,
  "questionNumber": 2,
  "currentTopic": "Vector Databases",
  "curriculumDay": 8,
  "coveredDaysCount": 2,
  "difficulty": "hard"
}
```

#### 3. Final Completion Response (After 8+ questions and 4+ days)
```json
{
  "reply": "Interview completed.",
  "done": true,
  "feedback": {
    "summary": "Sarah Johnson demonstrated solid enterprise AI engineering competencies suited for a Senior Data Engineer...",
    "strengths": [
      "Strong architectural knowledge in Vector Databases",
      "Solid comprehension of RAG & Vector Database pipelines",
      "Effective technical communication"
    ],
    "gaps": [
      "Needs further hands-on experience with production monitoring and metrics",
      "Could refine edge-case error handling in multi-agent workflows"
    ],
    "next": [
      "Review vector search retrieval re-ranking and reciprocal rank fusion techniques",
      "Practice building end-to-end agentic workflows with MCP servers",
      "Implement automated LLM evaluation and observability with Prometheus & Grafana"
    ]
  }
}
```

---

## Automated Testing

Run the full pytest suite:

```powershell
python -m pytest backend/tests/ -v
```

### Test Coverage:
- `POST /api/interview` start request initialization
- Session creation & persistence across turns
- Adaptive difficulty progression & context-aware follow-ups
- Enforcement of 8-question minimum & 4 curriculum days minimum
- Feedback schema structural validation
- Invalid input & error handling (missing session, empty messages)

---

## Hackathon Requirements Checklist

- [x] Expose HTTP endpoint `POST /api/interview`
- [x] Maintain session state using `sessionId`
- [x] Support Start Interview request (`sessionId` + `candidate`)
- [x] Support Conversation turns (`sessionId` + `message`)
- [x] Support Final completion response with exact feedback schema (`summary`, `strengths`, `gaps`, `next`)
- [x] Ask at least 8 questions
- [x] Cover at least 4 different curriculum days
- [x] Personalize questions using candidate profile
- [x] Adapt difficulty based on candidate answers
- [x] Intelligent follow-ups referencing prior answers
- [x] Configurable AI provider (`mock` mode out-of-the-box)
- [x] Modern responsive frontend interface
- [x] Automated pytest test suite passing
