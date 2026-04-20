"""
agent.py — HR Policy Assistant (Shared Agent Module)
Agentic AI Capstone | Dr. Kanthi Kiran Sirra | 2026

This module is imported by both the notebook and capstone_streamlit.py.
It contains: DOCUMENTS, CapstoneState, all node functions, and graph assembly.
"""

import os
import datetime
from typing import TypedDict, List

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ── LLM & Embedder ────────────────────────────────────────────────────────────
llm      = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ── Knowledge Base: 12 HR Policy Documents ───────────────────────────────────
DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "Annual Leave Policy",
        "text": """Annual Leave Policy — TechCorp India Pvt. Ltd.

All full-time employees are entitled to 18 days of paid annual leave (AL) per calendar year.
Leave is accrued at 1.5 days per month starting from the date of joining.
Employees must have completed a probation period of 3 months before they can avail annual leave.
Annual leave can be carried forward to the next calendar year, with a maximum carry-forward cap of 9 days.
Any unused leave beyond 9 days at the end of December will lapse and will not be encashed.
Leave encashment is permitted only at the time of resignation or retirement, calculated at basic salary per day.
Annual leave must be applied for at least 3 working days in advance through the HR portal (hrportal.techcorp.in).
Emergency exceptions require manager approval within 24 hours of the absence.
Annual leave cannot be split into half-days. The minimum unit is 1 full working day.
Employees on probation are eligible for only casual leave and sick leave, not annual leave.
Leave during notice period is subject to manager and HR approval and is not guaranteed.
"""
    },
    {
        "id": "doc_002",
        "topic": "Sick Leave and Medical Leave",
        "text": """Sick Leave (SL) and Medical Leave Policy — TechCorp India Pvt. Ltd.

All employees are entitled to 12 days of paid sick leave per calendar year.
Sick leave is credited in full at the beginning of each calendar year (January 1st).
New joiners receive sick leave on a pro-rata basis based on their month of joining.
For sick leave up to 2 consecutive days, self-declaration via email to HR and manager is sufficient.
For sick leave of 3 or more consecutive days, a medical certificate from a registered physician is mandatory.
Medical certificates must be submitted within 3 working days of returning to the office.
Sick leave cannot be carried forward to the next year or encashed under any circumstances.
Extended medical leave (beyond 12 days) may be granted as loss-of-pay leave with HR approval.
Maternity leave is separate from sick leave: 26 weeks of paid maternity leave as per the Maternity Benefit Act, 2017.
Paternity leave: 5 working days of paid paternity leave within 6 months of childbirth or adoption.
"""
    },
    {
        "id": "doc_003",
        "topic": "Casual Leave and Compensatory Off",
        "text": """Casual Leave (CL) and Compensatory Off (Comp-Off) Policy — TechCorp India Pvt. Ltd.

All employees are entitled to 6 days of casual leave per calendar year.
Casual leave is for short-term personal needs and cannot exceed 3 consecutive days.
Casual leave can be taken as half-days (morning or afternoon session).
CL must be applied in advance except in genuine emergencies. Retrospective CL must be approved by the manager within 1 working day.
Casual leave cannot be combined with annual leave to form a continuous stretch exceeding 7 days without special HR approval.
Casual leave cannot be carried forward to the next year and is not encashable.

Compensatory Off (Comp-Off):
Employees who are required to work on declared public holidays or weekends are entitled to a compensatory off.
Comp-off must be availed within 60 days of the date on which extra work was done.
Comp-off must be applied through the HR portal and approved by the reporting manager.
Comp-offs cannot be encashed and lapse after 60 days if not availed.
"""
    },
    {
        "id": "doc_004",
        "topic": "Attendance and Working Hours",
        "text": """Attendance and Working Hours Policy — TechCorp India Pvt. Ltd.

Standard working hours: 9 hours per day including a 1-hour lunch break (net working: 8 hours/day).
Official working hours: 9:00 AM to 6:00 PM, Monday to Friday.
Employees are required to mark attendance in/out using the biometric system or the mobile app.
A grace period of 15 minutes is allowed (9:15 AM). Arrival after 9:15 AM is marked as late.
3 late arrivals in a calendar month count as 1 half-day loss of pay.
Employees working from office must achieve a minimum of 85% in-office attendance per month.
Habitual late arrivals (more than 5 in a month) may result in a formal warning letter.
For remote-working employees, login must be recorded in the attendance system by 9:15 AM.
Overtime beyond standard hours is not mandated; however, project requirements may require it with manager agreement.
Saturday is a working day for the first and third weekends of the month (9:00 AM to 1:00 PM).
Second and fourth Saturdays are off. All five Sundays in a month are always off.
"""
    },
    {
        "id": "doc_005",
        "topic": "Salary and Payroll",
        "text": """Salary and Payroll Policy — TechCorp India Pvt. Ltd.

Salaries are processed on the last working day of every month.
In case the last working day is a public holiday, salary is credited on the previous working day.
Salary is credited directly to the employee's registered bank account via NEFT/IMPS.
Salary slips are generated on the 1st of every month and are available on the HR portal (hrportal.techcorp.in).
CTC (Cost to Company) includes: Basic Salary (40% of CTC), HRA (20% of CTC), Special Allowance, PF contribution, and medical allowance.
Provident Fund (PF): 12% of basic salary is deducted and matched by the company as per EPF Act.
Professional Tax is deducted as per the applicable state slab.
Income tax (TDS) is deducted monthly based on the investment declarations submitted at the start of the year.
Investment proof submission deadline: January 31st of each financial year.
Employees failing to submit proofs will have TDS recalculated in March, which may result in a higher deduction.
Annual salary increments are processed in April following the performance review cycle in March.
Increment letters are issued by April 15th each year.
For salary discrepancies, employees should raise a ticket on the HR portal within 5 working days of salary credit.
"""
    },
    {
        "id": "doc_006",
        "topic": "Work From Home Policy",
        "text": """Work From Home (WFH) Policy — TechCorp India Pvt. Ltd.

Employees are eligible for up to 2 WFH days per week, subject to manager approval and project requirements.
WFH eligibility begins after the completion of 6 months of employment and the probation period.
WFH requests must be submitted on the HR portal by 9:00 PM the previous day.
Ad-hoc or same-day WFH requires manager approval via email or messaging before 9:00 AM.
During WFH, employees are expected to be reachable on all official communication channels (email, Teams, Slack) during working hours.
Core collaboration hours are 10:00 AM to 4:00 PM — employees must be available for meetings during this window.
Internet and power supply are the employee's responsibility during WFH. The company does not provide reimbursement for home internet.
Laptop and required equipment will be provided by IT. Any damage or loss must be reported immediately.
WFH privilege can be suspended for employees with poor performance, attendance issues, or policy violations.
Full remote work arrangements require special approval from the Department Head and HR Director for a maximum of 3 months at a time.
"""
    },
    {
        "id": "doc_007",
        "topic": "Performance Review and Appraisal",
        "text": """Performance Review and Appraisal Process — TechCorp India Pvt. Ltd.

TechCorp follows an annual performance review cycle.
Review cycle timeline:
  - January: Goal-setting for the new year; carry-over goals reviewed.
  - February: Mid-year check-in (informal).
  - March 1–15: Employee self-assessment submitted on the HR portal.
  - March 15–31: Manager assessment and rating submission.
  - April 1–10: Calibration meetings across departments by HRBPs.
  - April 15: Increment letters issued; performance rating communicated to employees.

Performance ratings scale:
  5 — Outstanding (top 5% of performers; 20–30% increment)
  4 — Exceeds Expectations (15–20% increment)
  3 — Meets Expectations (8–12% increment)
  2 — Partially Meets Expectations (0–5% increment; PIP may follow)
  1 — Does Not Meet Expectations (no increment; Performance Improvement Plan mandatory)

Performance Improvement Plan (PIP):
Employees rated 1 or 2 may be placed on a 90-day PIP with clear, measurable targets.
Failure to meet PIP targets may result in separation from the company.
Employees can discuss their rating with their manager. Appeals must be raised within 15 days of rating communication.
"""
    },
    {
        "id": "doc_008",
        "topic": "Grievance Redressal Policy",
        "text": """Grievance Redressal Policy — TechCorp India Pvt. Ltd.

TechCorp is committed to a fair and transparent workplace. Any employee who faces a workplace grievance has the right to raise it formally.

Types of grievances covered:
- Workplace harassment (including sexual harassment, as per POSH Act 2013)
- Discrimination based on gender, religion, caste, or disability
- Salary or benefit discrepancies
- Unfair treatment or performance review disputes
- Violation of company policies by peers or management

How to raise a grievance:
Step 1: Raise the concern informally with your direct manager.
Step 2: If unresolved within 5 working days, raise a formal grievance via the HR portal under "Grievance Submission."
Step 3: HR will acknowledge the grievance within 2 working days and assign an HR Business Partner (HRBP).
Step 4: Investigation completed within 21 working days. Employee is informed of the outcome.
Step 5: Appeal: If dissatisfied, the employee may escalate to the HR Director within 10 working days.

For POSH complaints, the Internal Complaints Committee (ICC) handles all matters confidentially.
ICC contact: icc@techcorp.in | Helpline: 1800-XXX-9090 (toll-free, confidential).
Retaliation against any employee who raises a grievance is a serious disciplinary offence.
"""
    },
    {
        "id": "doc_009",
        "topic": "Travel and Expense Reimbursement",
        "text": """Travel and Expense Reimbursement Policy — TechCorp India Pvt. Ltd.

Business travel reimbursement is applicable for client meetings, training, and inter-office travel approved in advance.

Travel approval: All business travel must be pre-approved by the reporting manager and HR via the travel request form on the HR portal.

Reimbursable expenses:
- Local travel: Auto-rickshaw, cab (Ola/Uber), or metro fares are reimbursable with receipts or app screenshots.
- Outstation travel: Economy class air travel or AC train travel (3-Tier) is standard. Business class requires VP-level approval.
- Accommodation: Hotel up to Rs. 3,500/night in Tier-1 cities; Rs. 2,500/night in Tier-2 cities.
- Meals: Daily meal allowance of Rs. 600 for full-day outstation travel.
- Fuel reimbursement: Rs. 8/km for personal vehicle use (approved route only).

Non-reimbursable expenses: Alcohol, personal entertainment, fines or penalties, and flight upgrades.

Expense claim submission:
- Claims must be submitted within 15 working days of travel completion.
- All expenses above Rs. 500 require a valid receipt/invoice.
- Claims submitted after 30 days will not be processed.
- Reimbursement is processed in the next salary cycle after claim approval.

International travel: Requires approval from the Department Head and CFO. Per-diem rates vary by country (contact HR for country-specific rates).
"""
    },
    {
        "id": "doc_010",
        "topic": "Employee Benefits and Health Insurance",
        "text": """Employee Benefits and Health Insurance — TechCorp India Pvt. Ltd.

Health Insurance:
All full-time employees are covered under a group health insurance policy from the date of joining.
Coverage: Rs. 5,00,000 per annum (floater basis covering employee + spouse + 2 children).
Parents can be added at a subsidised premium of Rs. 8,000/year (deducted from salary in April).
Pre-existing conditions are covered after 1 year of continuous employment.
Cashless hospitalisation is available at 500+ network hospitals across India.
For non-network hospitals, reimbursement must be claimed within 30 days of discharge.
Insurance card and policy document are available on the HR portal under "My Benefits."

Additional Benefits:
- Life Insurance: Term life cover of 3x annual CTC (company-paid).
- Personal Accident Insurance: Coverage up to Rs. 10,00,000 (company-paid).
- Gratuity: Applicable after 5 years of continuous service as per the Payment of Gratuity Act.
- Employee Stock Option Plan (ESOP): Eligible after 2 years; vesting schedule: 4 years (25% per year).
- Flexi Benefits: Rs. 15,000/year for books, internet, gym, or skill development (claim via HR portal).
- Employee Referral Bonus: Rs. 25,000 for successful referral (paid after 6 months of the referred employee's joining).
"""
    },
    {
        "id": "doc_011",
        "topic": "Onboarding and Probation",
        "text": """Onboarding and Probation Policy — TechCorp India Pvt. Ltd.

Onboarding process:
Day 1: IT setup (laptop, accounts, email), HR induction, and ID card issuance.
Week 1: Department orientation, buddy assignment, and introduction to key stakeholders.
Month 1: Mandatory online compliance training (POSH, Code of Conduct, Data Security) to be completed in the first 30 days.
Month 3: First check-in with HR and manager to assess integration and comfort.

Probation period: 3 months for most roles; 6 months for senior roles (Deputy Manager and above).
During probation:
- Employees are on monthly performance monitoring.
- Notice period is 15 days (vs. 60 days post-confirmation).
- Annual leave is not available; casual leave (CL) and sick leave (SL) are available.
- Health insurance and life insurance begin from Day 1.

Confirmation:
At the end of the probation period, the manager submits a confirmation recommendation to HR.
HR issues a confirmation letter within 10 working days of probation completion.
If performance is unsatisfactory, probation may be extended by up to 3 additional months with written communication.
If performance continues to be unsatisfactory after extension, employment may be terminated with 15 days' notice.

IT assets provided during onboarding must be returned at exit. Damage beyond normal wear is chargeable.
"""
    },
    {
        "id": "doc_012",
        "topic": "Exit and Offboarding Process",
        "text": """Exit and Offboarding Process — TechCorp India Pvt. Ltd.

Resignation:
Employees wishing to resign must submit a formal resignation letter to their manager and HR via the HR portal.
Notice period: 60 days for confirmed employees; 15 days during probation.
Notice period can be waived or shortened by mutual agreement between the employee and the company.
Serving notice in full is mandatory unless a buyout is agreed upon. Notice buyout cost = remaining notice days x per-day salary.

Exit process steps:
1. Resignation submitted → HR acknowledges within 2 working days.
2. Last working day (LWD) confirmed and communicated.
3. Knowledge transfer (KT) plan submitted to manager within the first 2 weeks of notice.
4. Exit interview scheduled with HR in the final week.
5. IT asset return: Laptop, access card, and all company equipment returned on LWD.
6. All pending expense claims must be submitted at least 10 working days before LWD.
7. Full and Final (F&F) settlement is processed within 45 days of LWD.

F&F settlement includes:
- Remaining salary for the month
- Leave encashment (annual leave balance)
- Gratuity (if eligible)
- ESOP: Unvested options are forfeited; vested options must be exercised within 90 days of exit.
Experience letter and relieving letter are issued digitally within 7 working days of F&F clearance.

Re-hire: Former employees who left in good standing may apply for rehire after 6 months. Previous service period will not count for leave or gratuity purposes.
"""
    },
]

# ── Build ChromaDB ────────────────────────────────────────────────────────────
def build_collection():
    client = chromadb.Client()
    try:
        client.delete_collection("hr_kb")
    except Exception:
        pass
    col = client.create_collection("hr_kb")
    texts      = [d["text"]  for d in DOCUMENTS]
    ids        = [d["id"]    for d in DOCUMENTS]
    metadatas  = [{"topic": d["topic"]} for d in DOCUMENTS]
    embeddings = embedder.encode(texts).tolist()
    col.add(documents=texts, embeddings=embeddings,
            ids=ids, metadatas=metadatas)
    return col

collection = build_collection()

# ── State ─────────────────────────────────────────────────────────────────────
class CapstoneState(TypedDict):
    question:     str
    messages:     List[dict]
    route:        str
    retrieved:    str
    sources:      List[str]
    tool_result:  str
    answer:       str
    faithfulness: float
    eval_retries: int
    user_name:    str          # HR-specific: remember employee name

# ── Thresholds ────────────────────────────────────────────────────────────────
FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES       = 2

# ── Node 1: Memory ────────────────────────────────────────────────────────────
def memory_node(state: CapstoneState) -> dict:
    msgs = state.get("messages", [])
    msgs = msgs + [{"role": "user", "content": state["question"]}]
    if len(msgs) > 6:
        msgs = msgs[-6:]
    # Extract user name if mentioned
    user_name = state.get("user_name", "")
    q_lower = state["question"].lower()
    if "my name is" in q_lower:
        parts = q_lower.split("my name is")
        if len(parts) > 1:
            name_part = parts[1].strip().split()[0]
            user_name = name_part.capitalize()
    return {"messages": msgs, "user_name": user_name}

# ── Node 2: Router ────────────────────────────────────────────────────────────
def router_node(state: CapstoneState) -> dict:
    question = state["question"]
    messages = state.get("messages", [])
    recent   = "; ".join(
        f"{m['role']}: {m['content'][:60]}" for m in messages[-3:-1]
    ) or "none"

    prompt = f"""You are a router for an HR Policy chatbot at TechCorp India.

Available routes:
- retrieve: the user is asking about HR policy, leave, salary, attendance, WFH, benefits, performance review, travel, onboarding, exit/offboarding, or any company policy topic — search the knowledge base.
- memory_only: the user is making small talk, asking what you just said, or asking something that can be answered purely from the recent conversation (e.g. "what did you just tell me?", "say that again").
- tool: the user needs the current date or time (e.g. "what is today's date?", "what day is it?", "how many days until December 31?").

Recent conversation: {recent}
Current question: {question}

Reply with ONLY one word: retrieve / memory_only / tool"""

    response = llm.invoke(prompt)
    decision = response.content.strip().lower()
    if "memory" in decision:
        decision = "memory_only"
    elif "tool" in decision:
        decision = "tool"
    else:
        decision = "retrieve"
    return {"route": decision}

# ── Node 3: Retrieval ─────────────────────────────────────────────────────────
def retrieval_node(state: CapstoneState) -> dict:
    q_emb   = embedder.encode([state["question"]]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=3)
    chunks  = results["documents"][0]
    topics  = [m["topic"] for m in results["metadatas"][0]]
    context = "\n\n---\n\n".join(
        f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks))
    )
    return {"retrieved": context, "sources": topics}

def skip_retrieval_node(state: CapstoneState) -> dict:
    return {"retrieved": "", "sources": []}

# ── Node 4: Tool (Date/Time) ──────────────────────────────────────────────────
def tool_node(state: CapstoneState) -> dict:
    """Returns current date and time in a human-readable format.
    Also computes useful HR-related dates (days left in year, month, etc.)"""
    try:
        now   = datetime.datetime.now()
        today = now.date()
        year_end    = datetime.date(today.year, 12, 31)
        month_end   = (datetime.date(today.year, today.month % 12 + 1, 1)
                       - datetime.timedelta(days=1)) if today.month < 12 \
                       else datetime.date(today.year, 12, 31)
        days_in_year_left  = (year_end - today).days
        days_in_month_left = (month_end - today).days

        result = (
            f"Current date: {today.strftime('%A, %d %B %Y')}\n"
            f"Current time: {now.strftime('%I:%M %p')}\n"
            f"Current month: {today.strftime('%B %Y')}\n"
            f"Days remaining in current month: {days_in_month_left} days\n"
            f"Days remaining in current calendar year: {days_in_year_left} days\n"
            f"Current financial year: {'FY ' + str(today.year) + '-' + str(today.year+1)[2:] if today.month >= 4 else 'FY ' + str(today.year-1) + '-' + str(today.year)[2:]}\n"
        )
    except Exception as e:
        result = f"Unable to retrieve date/time: {str(e)}"
    return {"tool_result": result}

# ── Node 5: Answer ────────────────────────────────────────────────────────────
def answer_node(state: CapstoneState) -> dict:
    question     = state["question"]
    retrieved    = state.get("retrieved", "")
    tool_result  = state.get("tool_result", "")
    messages     = state.get("messages", [])
    eval_retries = state.get("eval_retries", 0)
    user_name    = state.get("user_name", "")

    context_parts = []
    if retrieved:
        context_parts.append(f"HR POLICY KNOWLEDGE BASE:\n{retrieved}")
    if tool_result:
        context_parts.append(f"CURRENT DATE/TIME INFORMATION:\n{tool_result}")
    context = "\n\n".join(context_parts)

    name_greeting = f" You may address the employee by first name ({user_name}) if known." if user_name else ""

    if context:
        system_content = f"""You are a professional HR Policy Assistant for TechCorp India Pvt. Ltd.{name_greeting}
Your job is to answer employee questions about company HR policies clearly, accurately, and helpfully.

STRICT RULES:
1. Answer ONLY using the information provided in the context below. Do NOT use your training knowledge.
2. If the answer is not in the context, clearly say: "I don't have that specific information in my knowledge base. Please contact HR at hr@techcorp.in or call the HR helpline."
3. Always be polite, empathetic, and professional.
4. If a question is about a medical, legal, or financial decision, recommend the employee consult the appropriate professional.
5. Never make up numbers, dates, or policy details.

{context}"""
    else:
        system_content = f"""You are a professional HR Policy Assistant for TechCorp India Pvt. Ltd.{name_greeting}
Answer based on the conversation history. Be helpful and professional."""

    if eval_retries > 0:
        system_content += "\n\nIMPORTANT: Your previous answer did not meet faithfulness standards. Answer using ONLY information explicitly stated in the context above. Do not add anything from your own knowledge."

    lc_msgs = [SystemMessage(content=system_content)]
    for msg in messages[:-1]:
        if msg["role"] == "user":
            lc_msgs.append(HumanMessage(content=msg["content"]))
        else:
            lc_msgs.append(AIMessage(content=msg["content"]))
    lc_msgs.append(HumanMessage(content=question))

    response = llm.invoke(lc_msgs)
    return {"answer": response.content}

# ── Node 6: Eval ──────────────────────────────────────────────────────────────
def eval_node(state: CapstoneState) -> dict:
    answer  = state.get("answer", "")
    context = state.get("retrieved", "")[:500]
    retries = state.get("eval_retries", 0)

    if not context:
        return {"faithfulness": 1.0, "eval_retries": retries + 1}

    prompt = f"""Rate faithfulness: does this answer use ONLY information from the context?
Reply with ONLY a number between 0.0 and 1.0.
1.0 = fully faithful to context. 0.5 = some content not in context. 0.0 = mostly hallucinated.

Context: {context}
Answer: {answer[:300]}"""

    result = llm.invoke(prompt).content.strip()
    try:
        score = float(result.split()[0].replace(",", "."))
        score = max(0.0, min(1.0, score))
    except Exception:
        score = 0.5

    gate = "✅ PASS" if score >= FAITHFULNESS_THRESHOLD else "⚠️ RETRY"
    print(f"  [eval_node] Faithfulness: {score:.2f} — {gate}")
    return {"faithfulness": score, "eval_retries": retries + 1}

# ── Node 7: Save ──────────────────────────────────────────────────────────────
def save_node(state: CapstoneState) -> dict:
    messages = state.get("messages", [])
    messages = messages + [{"role": "assistant", "content": state["answer"]}]
    return {"messages": messages}

# ── Routing Functions ─────────────────────────────────────────────────────────
def route_decision(state: CapstoneState) -> str:
    route = state.get("route", "retrieve")
    if route == "tool":        return "tool"
    if route == "memory_only": return "skip"
    return "retrieve"

def eval_decision(state: CapstoneState) -> str:
    score   = state.get("faithfulness", 1.0)
    retries = state.get("eval_retries", 0)
    if score >= FAITHFULNESS_THRESHOLD or retries >= MAX_EVAL_RETRIES:
        return "save"
    return "answer"

# ── Graph Assembly ────────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(CapstoneState)

    graph.add_node("memory",   memory_node)
    graph.add_node("router",   router_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("skip",     skip_retrieval_node)
    graph.add_node("tool",     tool_node)
    graph.add_node("answer",   answer_node)
    graph.add_node("eval",     eval_node)
    graph.add_node("save",     save_node)

    graph.set_entry_point("memory")
    graph.add_edge("memory", "router")

    graph.add_conditional_edges(
        "router", route_decision,
        {"retrieve": "retrieve", "skip": "skip", "tool": "tool"}
    )

    graph.add_edge("retrieve", "answer")
    graph.add_edge("skip",     "answer")
    graph.add_edge("tool",     "answer")
    graph.add_edge("answer",   "eval")

    graph.add_conditional_edges(
        "eval", eval_decision,
        {"answer": "answer", "save": "save"}
    )
    graph.add_edge("save", END)

    checkpointer = MemorySaver()
    compiled_app = graph.compile(checkpointer=checkpointer)
    return compiled_app

# Build and export
app = build_graph()
print("✅ HR Policy Assistant agent loaded successfully.")
print(f"   Knowledge base: {collection.count()} documents")
print(f"   Topics: {[d['topic'] for d in DOCUMENTS]}")

# ── Helper ────────────────────────────────────────────────────────────────────
def ask(question: str, thread_id: str = "default") -> dict:
    """Run a single question through the agent."""
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"question": question}, config=config)
    return result
