"""
capstone_streamlit.py — HR Policy Assistant (Streamlit UI)
Agentic AI Capstone | Dr. Kanthi Kiran Sirra | 2026

Run:  streamlit run capstone_streamlit.py
"""

import streamlit as st
import uuid
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

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TechCorp HR Assistant",
    page_icon="🏢",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Load agent (cached so it doesn't reload on every rerun)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_agent():
    llm      = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    DOCUMENTS = [
        {"id": "doc_001", "topic": "Annual Leave Policy",
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
Leave during notice period is subject to manager and HR approval and is not guaranteed."""},

        {"id": "doc_002", "topic": "Sick Leave and Medical Leave",
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
Paternity leave: 5 working days of paid paternity leave within 6 months of childbirth or adoption."""},

        {"id": "doc_003", "topic": "Casual Leave and Compensatory Off",
         "text": """Casual Leave (CL) and Compensatory Off (Comp-Off) Policy — TechCorp India Pvt. Ltd.
All employees are entitled to 6 days of casual leave per calendar year.
Casual leave is for short-term personal needs and cannot exceed 3 consecutive days.
Casual leave can be taken as half-days (morning or afternoon session).
CL must be applied in advance except in genuine emergencies.
Casual leave cannot be combined with annual leave to form a continuous stretch exceeding 7 days without special HR approval.
Casual leave cannot be carried forward to the next year and is not encashable.
Compensatory Off (Comp-Off): Employees who work on declared public holidays or weekends are entitled to a comp-off.
Comp-off must be availed within 60 days of the extra work date.
Comp-offs cannot be encashed and lapse after 60 days if not availed."""},

        {"id": "doc_004", "topic": "Attendance and Working Hours",
         "text": """Attendance and Working Hours Policy — TechCorp India Pvt. Ltd.
Standard working hours: 9 hours per day including a 1-hour lunch break (net working: 8 hours/day).
Official working hours: 9:00 AM to 6:00 PM, Monday to Friday.
Employees are required to mark attendance in/out using the biometric system or the mobile app.
A grace period of 15 minutes is allowed (9:15 AM). Arrival after 9:15 AM is marked as late.
3 late arrivals in a calendar month count as 1 half-day loss of pay.
Employees working from office must achieve a minimum of 85% in-office attendance per month.
Saturday is a working day for the first and third weekends of the month (9:00 AM to 1:00 PM).
Second and fourth Saturdays are off. All Sundays are off."""},

        {"id": "doc_005", "topic": "Salary and Payroll",
         "text": """Salary and Payroll Policy — TechCorp India Pvt. Ltd.
Salaries are processed on the last working day of every month.
Salary is credited directly to the employee's registered bank account via NEFT/IMPS.
Salary slips are generated on the 1st of every month and are available on the HR portal.
CTC includes: Basic Salary (40% of CTC), HRA (20% of CTC), Special Allowance, PF contribution, and medical allowance.
Provident Fund (PF): 12% of basic salary is deducted and matched by the company as per EPF Act.
Professional Tax is deducted as per the applicable state slab.
Income tax (TDS) is deducted monthly based on investment declarations submitted at the start of the year.
Investment proof submission deadline: January 31st of each financial year.
Annual salary increments are processed in April following the performance review cycle in March.
Increment letters are issued by April 15th each year.
For salary discrepancies, raise a ticket on the HR portal within 5 working days of salary credit."""},

        {"id": "doc_006", "topic": "Work From Home Policy",
         "text": """Work From Home (WFH) Policy — TechCorp India Pvt. Ltd.
Employees are eligible for up to 2 WFH days per week, subject to manager approval.
WFH eligibility begins after the completion of 6 months of employment and the probation period.
WFH requests must be submitted on the HR portal by 9:00 PM the previous day.
During WFH, employees must be reachable on all official channels during working hours.
Core collaboration hours: 10:00 AM to 4:00 PM — employees must be available for meetings.
Internet and power supply are the employee's responsibility during WFH.
WFH privilege can be suspended for employees with poor performance or attendance issues.
Full remote work arrangements require special approval from the Department Head and HR Director."""},

        {"id": "doc_007", "topic": "Performance Review and Appraisal",
         "text": """Performance Review and Appraisal Process — TechCorp India Pvt. Ltd.
TechCorp follows an annual performance review cycle.
Timeline: January (goal-setting), March 1-15 (self-assessment), March 15-31 (manager assessment), April 15 (increment letters).
Performance ratings: 5-Outstanding (20-30% increment), 4-Exceeds Expectations (15-20%), 3-Meets Expectations (8-12%), 2-Partially Meets (0-5%), 1-Does Not Meet (no increment, PIP mandatory).
PIP (Performance Improvement Plan): 90-day plan with clear measurable targets for employees rated 1 or 2.
Failure to meet PIP targets may result in separation from the company.
Appeals must be raised within 15 days of rating communication."""},

        {"id": "doc_008", "topic": "Grievance Redressal Policy",
         "text": """Grievance Redressal Policy — TechCorp India Pvt. Ltd.
Any employee who faces a workplace grievance has the right to raise it formally.
Types of grievances covered: harassment, discrimination, salary discrepancies, unfair treatment, policy violations.
Step 1: Raise concern informally with direct manager.
Step 2: If unresolved in 5 working days, raise formal grievance via HR portal under Grievance Submission.
Step 3: HR acknowledges within 2 working days and assigns an HRBP.
Step 4: Investigation completed within 21 working days.
Step 5: Appeal to HR Director within 10 working days if dissatisfied.
For POSH complaints, the Internal Complaints Committee (ICC) handles matters confidentially.
ICC contact: icc@techcorp.in | Helpline: 1800-XXX-9090 (toll-free, confidential).
Retaliation against anyone who raises a grievance is a serious disciplinary offence."""},

        {"id": "doc_009", "topic": "Travel and Expense Reimbursement",
         "text": """Travel and Expense Reimbursement Policy — TechCorp India Pvt. Ltd.
All business travel must be pre-approved via the travel request form on the HR portal.
Reimbursable: local cab/auto fares with receipts, economy air or 3-Tier AC train, hotel up to Rs. 3,500/night (Tier-1 cities), Rs. 2,500/night (Tier-2 cities), meal allowance Rs. 600/day outstation, fuel Rs. 8/km for personal vehicle.
Non-reimbursable: alcohol, personal entertainment, fines, flight upgrades.
Expense claims must be submitted within 15 working days of travel completion.
All expenses above Rs. 500 require a valid receipt.
Claims submitted after 30 days will not be processed.
Reimbursement is processed in the next salary cycle after claim approval."""},

        {"id": "doc_010", "topic": "Employee Benefits and Health Insurance",
         "text": """Employee Benefits and Health Insurance — TechCorp India Pvt. Ltd.
Health Insurance: Rs. 5,00,000 per annum (floater: employee + spouse + 2 children) from Day 1.
Parents can be added at a subsidised premium of Rs. 8,000/year (deducted from salary in April).
Pre-existing conditions covered after 1 year of continuous employment.
Cashless hospitalisation at 500+ network hospitals across India.
Life Insurance: Term life cover of 3x annual CTC (company-paid).
Personal Accident Insurance: Coverage up to Rs. 10,00,000 (company-paid).
Gratuity: Applicable after 5 years of continuous service per the Payment of Gratuity Act.
ESOP: Eligible after 2 years; 4-year vesting schedule (25% per year).
Flexi Benefits: Rs. 15,000/year for books, internet, gym, or skill development.
Employee Referral Bonus: Rs. 25,000 for successful referral (paid after 6 months of referred employee joining)."""},

        {"id": "doc_011", "topic": "Onboarding and Probation",
         "text": """Onboarding and Probation Policy — TechCorp India Pvt. Ltd.
Day 1: IT setup (laptop, accounts, email), HR induction, and ID card issuance.
Week 1: Department orientation, buddy assignment, introduction to key stakeholders.
Month 1: Mandatory online compliance training (POSH, Code of Conduct, Data Security) within first 30 days.
Probation period: 3 months for most roles; 6 months for Deputy Manager and above.
During probation: monthly monitoring, 15-day notice period, no annual leave (only CL and SL).
Health insurance and life insurance begin from Day 1.
Confirmation letter issued within 10 working days of probation completion.
Probation may be extended by up to 3 months if performance is unsatisfactory."""},

        {"id": "doc_012", "topic": "Exit and Offboarding Process",
         "text": """Exit and Offboarding Process — TechCorp India Pvt. Ltd.
Resignation: Submit formal resignation via HR portal.
Notice period: 60 days for confirmed employees; 15 days during probation.
Notice buyout cost = remaining notice days x per-day salary.
Exit steps: Resignation submitted → LWD confirmed → Knowledge transfer plan in first 2 weeks → Exit interview in final week → IT asset return on LWD → F&F settlement within 45 days of LWD.
F&F settlement includes: remaining salary, leave encashment, gratuity (if eligible).
ESOP: Unvested options forfeited; vested options must be exercised within 90 days of exit.
Experience letter and relieving letter issued within 7 working days of F&F clearance.
Re-hire: Former employees may apply after 6 months if they left in good standing."""},
    ]

    # Build ChromaDB
    client = chromadb.Client()
    try:
        client.delete_collection("hr_kb_ui")
    except Exception:
        pass
    col = client.create_collection("hr_kb_ui")
    texts = [d["text"] for d in DOCUMENTS]
    col.add(
        documents=texts,
        embeddings=embedder.encode(texts).tolist(),
        ids=[d["id"] for d in DOCUMENTS],
        metadatas=[{"topic": d["topic"]} for d in DOCUMENTS],
    )

    # ── State ────────────────────────────────────────────────
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
        user_name:    str

    FAITHFULNESS_THRESHOLD = 0.7
    MAX_EVAL_RETRIES       = 2

    # ── Nodes ────────────────────────────────────────────────
    def memory_node(state):
        msgs = state.get("messages", [])
        msgs = msgs + [{"role": "user", "content": state["question"]}]
        if len(msgs) > 6:
            msgs = msgs[-6:]
        user_name = state.get("user_name", "")
        q_lower = state["question"].lower()
        if "my name is" in q_lower:
            parts = q_lower.split("my name is")
            if len(parts) > 1:
                user_name = parts[1].strip().split()[0].capitalize()
        return {"messages": msgs, "user_name": user_name}

    def router_node(state):
        question = state["question"]
        messages = state.get("messages", [])
        recent   = "; ".join(f"{m['role']}: {m['content'][:60]}" for m in messages[-3:-1]) or "none"
        prompt = f"""You are a router for an HR Policy chatbot at TechCorp India.
Routes: retrieve (HR policy questions), memory_only (small talk or referencing what was just said), tool (needs today's date or time).
Recent conversation: {recent}
Current question: {question}
Reply with ONLY one word: retrieve / memory_only / tool"""
        decision = llm.invoke(prompt).content.strip().lower()
        if "memory" in decision:   decision = "memory_only"
        elif "tool" in decision:   decision = "tool"
        else:                      decision = "retrieve"
        return {"route": decision}

    def retrieval_node(state):
        q_emb   = embedder.encode([state["question"]]).tolist()
        results = col.query(query_embeddings=q_emb, n_results=3)
        chunks  = results["documents"][0]
        topics  = [m["topic"] for m in results["metadatas"][0]]
        context = "\n\n---\n\n".join(f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks)))
        return {"retrieved": context, "sources": topics}

    def skip_retrieval_node(state):
        return {"retrieved": "", "sources": []}

    def tool_node(state):
        try:
            now   = datetime.datetime.now()
            today = now.date()
            year_end      = datetime.date(today.year, 12, 31)
            days_left_yr  = (year_end - today).days
            result = (
                f"Current date: {today.strftime('%A, %d %B %Y')}\n"
                f"Current time: {now.strftime('%I:%M %p')}\n"
                f"Days remaining in calendar year: {days_left_yr} days\n"
                f"Current financial year: {'FY ' + str(today.year) + '-' + str(today.year+1)[2:] if today.month >= 4 else 'FY ' + str(today.year-1) + '-' + str(today.year)[2:]}"
            )
        except Exception as e:
            result = f"Date/time error: {str(e)}"
        return {"tool_result": result}

    def answer_node(state):
        question     = state["question"]
        retrieved    = state.get("retrieved", "")
        tool_result  = state.get("tool_result", "")
        messages     = state.get("messages", [])
        eval_retries = state.get("eval_retries", 0)
        user_name    = state.get("user_name", "")
        context_parts = []
        if retrieved:    context_parts.append(f"HR POLICY KNOWLEDGE BASE:\n{retrieved}")
        if tool_result:  context_parts.append(f"DATE/TIME:\n{tool_result}")
        context = "\n\n".join(context_parts)
        name_note = f" Address the employee as {user_name} if their name is known." if user_name else ""
        if context:
            sys = f"""You are a professional HR Policy Assistant for TechCorp India Pvt. Ltd.{name_note}
Answer ONLY using the context below. If information is not in the context, say: "I don't have that information. Please contact HR at hr@techcorp.in."
Do NOT use your training knowledge. Be polite and professional.

{context}"""
        else:
            sys = f"You are a professional HR Policy Assistant for TechCorp India.{name_note} Answer from the conversation history. Be helpful."
        if eval_retries > 0:
            sys += "\n\nIMPORTANT: Previous answer was not faithful to context. Use ONLY the context provided."
        lc_msgs = [SystemMessage(content=sys)]
        for msg in messages[:-1]:
            lc_msgs.append(HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]))
        lc_msgs.append(HumanMessage(content=question))
        return {"answer": llm.invoke(lc_msgs).content}

    def eval_node(state):
        answer  = state.get("answer", "")
        context = state.get("retrieved", "")[:500]
        retries = state.get("eval_retries", 0)
        if not context:
            return {"faithfulness": 1.0, "eval_retries": retries + 1}
        prompt = f"Rate faithfulness 0.0-1.0 (reply with ONLY a number). Does the answer use ONLY context info?\nContext: {context}\nAnswer: {answer[:300]}"
        try:
            score = float(llm.invoke(prompt).content.strip().split()[0].replace(",", "."))
            score = max(0.0, min(1.0, score))
        except Exception:
            score = 0.5
        return {"faithfulness": score, "eval_retries": retries + 1}

    def save_node(state):
        msgs = state.get("messages", [])
        return {"messages": msgs + [{"role": "assistant", "content": state["answer"]}]}

    # ── Graph ────────────────────────────────────────────────
    def route_decision(state):
        r = state.get("route", "retrieve")
        if r == "tool":        return "tool"
        if r == "memory_only": return "skip"
        return "retrieve"

    def eval_decision(state):
        if state.get("faithfulness", 1.0) >= FAITHFULNESS_THRESHOLD or state.get("eval_retries", 0) >= MAX_EVAL_RETRIES:
            return "save"
        return "answer"

    g = StateGraph(CapstoneState)
    for name, fn in [("memory", memory_node), ("router", router_node), ("retrieve", retrieval_node),
                     ("skip", skip_retrieval_node), ("tool", tool_node), ("answer", answer_node),
                     ("eval", eval_node), ("save", save_node)]:
        g.add_node(name, fn)

    g.set_entry_point("memory")
    g.add_edge("memory", "router")
    g.add_conditional_edges("router", route_decision, {"retrieve": "retrieve", "skip": "skip", "tool": "tool"})
    g.add_edge("retrieve", "answer")
    g.add_edge("skip",     "answer")
    g.add_edge("tool",     "answer")
    g.add_edge("answer",   "eval")
    g.add_conditional_edges("eval", eval_decision, {"answer": "answer", "save": "save"})
    g.add_edge("save", END)

    agent_app = g.compile(checkpointer=MemorySaver())
    topics    = [d["topic"] for d in DOCUMENTS]
    return agent_app, topics

# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────
try:
    agent_app, KB_TOPICS = load_agent()
    agent_ready = True
except Exception as e:
    agent_ready = False
    agent_error = str(e)

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]
if "faithfulness_scores" not in st.session_state:
    st.session_state.faithfulness_scores = []

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/organization.png", width=60)
    st.title("TechCorp HR")
    st.caption("AI-Powered Policy Assistant")
    st.divider()

    st.markdown("**📋 Topics I can help with:**")
    for topic in KB_TOPICS:
        st.markdown(f"• {topic}")

    st.divider()
    st.markdown(f"**Session ID:** `{st.session_state.thread_id}`")

    if st.session_state.faithfulness_scores:
        avg_faith = sum(st.session_state.faithfulness_scores) / len(st.session_state.faithfulness_scores)
        st.metric("Avg Faithfulness", f"{avg_faith:.2f}", help="How grounded the answers are to the policy KB")

    st.divider()
    if st.button("🗑️ New Conversation", use_container_width=True):
        st.session_state.messages          = []
        st.session_state.thread_id         = str(uuid.uuid4())[:8]
        st.session_state.faithfulness_scores = []
        st.rerun()

    st.divider()
    st.caption("📧 HR Helpdesk: hr@techcorp.in")
    st.caption("☎️ Helpline: 1800-XXX-9090")
    st.caption("🌐 Portal: hrportal.techcorp.in")

# ─────────────────────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("🏢 TechCorp HR Policy Assistant")
st.caption("Ask me anything about company policies — leave, salary, attendance, WFH, benefits, and more.")

if not agent_ready:
    st.error(f"❌ Agent failed to load: {agent_error}")
    st.info("Make sure your GROQ_API_KEY is set in .env and all dependencies are installed.")
    st.stop()

# Suggested questions
if not st.session_state.messages:
    st.markdown("#### 💡 Try asking:")
    cols = st.columns(2)
    suggestions = [
        "How many annual leave days do I get?",
        "What is the WFH policy?",
        "When is salary credited each month?",
        "What are the health insurance benefits?",
        "What is today's date?",
        "How do I raise a grievance?",
    ]
    for i, s in enumerate(suggestions):
        if cols[i % 2].button(s, key=f"sug_{i}", use_container_width=True):
            st.session_state._pending_question = s
            st.rerun()

# Display conversation history
for msg in st.session_state.messages:
    icon = "🧑‍💼" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=icon):
        st.markdown(msg["content"])

# Handle pending suggestion click
pending = st.session_state.pop("_pending_question", None)

# Chat input
user_input = st.chat_input("Ask about any HR policy…") or pending

if user_input:
    # Show user message
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(user_input)

    # Run agent
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Looking up policy…"):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = agent_app.invoke({"question": user_input}, config=config)

        answer      = result.get("answer", "Sorry, I couldn't generate a response.")
        route       = result.get("route", "?")
        faithfulness = result.get("faithfulness", 1.0)
        sources     = result.get("sources", [])

        st.markdown(answer)

        # Metadata expander
        with st.expander("🔍 Response details", expanded=False):
            col1, col2 = st.columns(2)
            col1.metric("Route", route)
            col2.metric("Faithfulness", f"{faithfulness:.2f}")
            if sources:
                st.markdown("**Sources retrieved:**")
                for src in sources:
                    st.markdown(f"  • {src}")

    # Persist
    st.session_state.messages.append({"role": "user",      "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.faithfulness_scores.append(faithfulness)
