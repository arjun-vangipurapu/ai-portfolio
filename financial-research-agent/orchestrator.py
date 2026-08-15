# orchestrator.py
import asyncio
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from db_agent import run_db_agent
from pdf_agent import run_pdf_agent
from dotenv import load_dotenv
import os
import time

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

VALID_AGENTS = ["pdf_agent", "db_agent", "both"]

# --- Step 1: Query Analyzer ---
analyzer_prompt = PromptTemplate.from_template("""
You are a query router for a financial research system.
Decide which agents are needed to answer the question.

Agents available:
- pdf_agent: searches earnings call transcripts for management commentary, guidance, strategy, quotes
- db_agent: queries structured financial data — revenue, stock price, PE ratio, growth metrics
- both: question needs data AND commentary together

Reply with ONLY one of: pdf_agent | db_agent | both

Examples:
"What did Tim Cook say about iPhone?" → pdf_agent
"What is Apple's market cap?" → db_agent
"Did Apple's revenue match their guidance?" → both
"Which company has highest PE ratio?" → db_agent
"What is Microsoft's AI strategy?" → pdf_agent
"Compare revenue growth vs what management promised" → both

Question: {question}
Agent:
""")

# --- Step 2: Reflect Node ---
reflect_prompt = PromptTemplate.from_template("""
You are a financial research quality checker.
Review the answer below and check:
1. Does it actually answer the question?
2. Are there any contradictions between sources?
3. Is the confidence level appropriate?

Question: {question}
Answer: {answer}

Reply with:
APPROVED: <brief reason>
or
NEEDS_REVISION: <what's missing or contradictory>
""")

# --- Step 3: Synthesis ---
synthesis_prompt = PromptTemplate.from_template("""
You are a senior financial analyst synthesizing research from multiple sources.
Combine the results below into a clear, concise answer.
Always cite which source provided which fact.
End with a confidence level: High / Medium / Low and why.

Question: {question}

PDF Agent Result (earnings call transcripts):
{pdf_result}

DB Agent Result (financial metrics):
{db_result}

Synthesized Answer:
""")

single_source_prompt = PromptTemplate.from_template("""
You are a senior financial analyst.
Answer the question clearly and concisely using the data below.
Cite the source. End with confidence level: High / Medium / Low.

Question: {question}
Data: {result}

Answer:
""")

# --- Async agent execution ---
async def run_pdf_async(question: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, run_pdf_agent, question)

async def run_db_async(question: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, run_db_agent, question)

# --- Main orchestrator ---
async def orchestrate(question: str) -> dict:
    start = time.time()
    result = {
        "question": question,
        "agents_used": [],
        "pdf_result": None,
        "db_result": None,
        "answer": None,
        "reflection": None,
        "latency_ms": 0
    }

    # Step 1: Route
    print(f"\n🔍 Analyzing query...")
    route_response = llm.invoke(analyzer_prompt.format(question=question))
    route = route_response.content.strip().lower()

    if route not in VALID_AGENTS:
        route = "both"

    print(f"📡 Routing to: {route}")
    result["agents_used"] = [route] if route != "both" else ["pdf_agent", "db_agent"]

    # Step 2: Execute agents (parallel if both needed)
    if route == "both":
        print(f"⚡ Running agents in parallel...")
        pdf_result, db_result = await asyncio.gather(
            run_pdf_async(question),
            run_db_async(question)
        )
        result["pdf_result"] = pdf_result
        result["db_result"] = db_result

        # Synthesize
        answer_response = llm.invoke(synthesis_prompt.format(
            question=question,
            pdf_result=pdf_result,
            db_result=db_result
        ))
        result["answer"] = answer_response.content

    elif route == "pdf_agent":
        print(f"📄 Running PDF agent...")
        pdf_result = await run_pdf_async(question)
        result["pdf_result"] = pdf_result

        answer_response = llm.invoke(single_source_prompt.format(
            question=question,
            result=pdf_result
        ))
        result["answer"] = answer_response.content

    else:  # db_agent
        print(f"🗄️ Running DB agent...")
        db_result = await run_db_async(question)
        result["db_result"] = db_result

        answer_response = llm.invoke(single_source_prompt.format(
            question=question,
            result=db_result
        ))
        result["answer"] = answer_response.content

    # Step 3: Reflect
    print(f"🔎 Reflecting on answer quality...")
    reflect_response = llm.invoke(reflect_prompt.format(
        question=question,
        answer=result["answer"]
    ))
    result["reflection"] = reflect_response.content

    result["latency_ms"] = int((time.time() - start) * 1000)
    return result

def run(question: str) -> dict:
    return asyncio.run(orchestrate(question))

if __name__ == "__main__":
    questions = [
        "What did Tim Cook say about iPhone revenue?",
        "What is the PE ratio of all three companies?",
        "Did Apple's revenue match what management guided?"
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = run(q)
        print(f"\n🤖 Answer:\n{result['answer']}")
        print(f"\n🔎 Reflection: {result['reflection']}")
        print(f"\n⏱️ Latency: {result['latency_ms']}ms")
        print(f"🔧 Agents used: {result['agents_used']}")