# 🏢 AI-Powered HR Policy Assistant using Agentic RAG

> **Agentic AI Hands-On Course 2026 — Capstone Project**
> Instructor: Dr. Kanthi Kiran Sirra | Sr. AI Engineer

---

## 📌 Problem Statement

Employees frequently ask repetitive questions about HR policies — leave, salary, attendance, WFH, and benefits. This overloads the HR team and causes delays. This project builds a **24/7 AI-powered HR Policy Assistant** that answers employee queries accurately from company policy documents, maintains conversation memory, and never fabricates information.

---

## ✅ Six Mandatory Capabilities

| # | Capability | Implementation |
|---|-----------|----------------|
| 1 | **LangGraph StateGraph** | 8-node graph: memory → router → retrieve/skip/tool → answer → eval → save |
| 2 | **ChromaDB RAG** | 12 HR policy documents, SentenceTransformer all-MiniLM-L6-v2, top-3 retrieval |
| 3 | **MemorySaver + thread_id** | Sliding window of 6 messages, employee name extraction |
| 4 | **Self-reflection eval node** | Faithfulness scored 0.0–1.0, retry if < 0.70, MAX_EVAL_RETRIES=2 |
| 5 | **Tool use — datetime** | Current date, time, days left in month/year, financial year |
| 6 | **Streamlit deployment** | Full chat UI with sidebar, memory persistence, New Conversation button |

---

## 🏗️ Architecture

```
User Question
      ↓
[memory_node] → sliding window + name extraction
      ↓
[router_node] → retrieve / memory_only / tool
      ↓
[retrieval_node] OR [skip_node] OR [tool_node]
      ↓
[answer_node] → grounded LLM response (ONLY from context)
      ↓
[eval_node] → faithfulness score → retry if < 0.70
      ↓
[save_node] → END
```

---

## 📚 Knowledge Base (12 Documents)

| Document | Topic |
|----------|-------|
| doc_001 | Annual Leave Policy |
| doc_002 | Sick Leave and Medical Leave |
| doc_003 | Casual Leave and Compensatory Off |
| doc_004 | Attendance and Working Hours |
| doc_005 | Salary and Payroll |
| doc_006 | Work From Home Policy |
| doc_007 | Performance Review and Appraisal |
| doc_008 | Grievance Redressal Policy |
| doc_009 | Travel and Expense Reimbursement |
| doc_010 | Employee Benefits and Health Insurance |
| doc_011 | Onboarding and Probation |
| doc_012 | Exit and Offboarding Process |

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/HR-Agentic-AI-Capstone.git
cd HR-Agentic-AI-Capstone
```

### 2. Install dependencies
```bash
pip install langgraph langchain-groq langchain-core chromadb sentence-transformers streamlit python-dotenv ragas datasets
```

### 3. Create `.env` file
```
GROQ_API_KEY=your_groq_api_key_here
```
Get your free API key at [console.groq.com](https://console.groq.com)

### 4. Run the Streamlit UI
```bash
python -m streamlit run capstone_streamlit.py
```
Open your browser at `http://localhost:8501`

---

## 💬 Sample Questions to Try

```
My name is Priya. How many annual leave days do I get?
Can I carry forward unused leave to next year?
What is today's date?
What is the WFH policy?
When is salary credited each month?
What health insurance does the company provide?
What happens if I get a performance rating of 1?
How do I raise a grievance?
```

---

## 📊 Evaluation Results

| Metric | Score | Method |
|--------|-------|--------|
| Faithfulness | **0.90** | Manual LLM scoring via Groq (official fallback) |
| Test Questions | **10/10 passed** | Domain + red-team tests |
| Red-Team Tests | **2/2 passed** | Out-of-scope + false premise |

---

## 🗂️ Project Structure

```
HR-Agentic-AI-Capstone/
├── agent.py                          # Shared agent module (KB, nodes, graph)
├── capstone_streamlit.py             # Streamlit chat UI
├── day13_capstone_completed.ipynb    # Completed notebook (all 8 parts)
├── HR_Capstone_Report.docx           # Full project report
├── .env                              # API keys (not uploaded to GitHub)
└── README.md                         # This file
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.13-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-1.1.8-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5.8-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.56-red)
![Groq](https://img.shields.io/badge/Groq-LLaMA--3.3--70b-purple)

- **LangGraph** — Agentic graph orchestration
- **ChromaDB** — Vector database for HR policy retrieval
- **SentenceTransformers** — all-MiniLM-L6-v2 embeddings
- **LangChain Groq** — LLaMA-3.3-70b-versatile LLM
- **Streamlit** — Web UI deployment
- **Python-dotenv** — API key management

---

## 🔴 Red-Team Tests

| Test | Question | Expected | Result |
|------|----------|----------|--------|
| Out-of-scope | What is the policy on cryptocurrency trading? | Admit it doesn't know, refer to HR | ✅ PASS |
| False premise | I heard we get 30 annual leave days — is that right? | Correct the error (actual: 18 days) | ✅ PASS |

---

## 🔮 Future Improvements

- Load real HR policy PDFs using PyMuPDF with 512-token chunking
- BM25 hybrid retrieval (BM25 + vector search with RRF fusion)
- WhatsApp bot deployment
- Multilingual support (Hindi, Telugu)
- Integration with real HR APIs

---

## 📄 License

This project was built as part of the **Agentic AI Hands-On Course 2026** by Dr. Kanthi Kiran Sirra.

---

*Built with ❤️ using LangGraph + ChromaDB + Streamlit*
