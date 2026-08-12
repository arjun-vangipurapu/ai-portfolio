# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from tools import semantic_search, action_tracker, conflict_check
from ingest import ingest_transcript
from dotenv import load_dotenv
import os, json
from datetime import datetime

load_dotenv()

app = FastAPI(
    title="Meeting Intelligence Agent",
    description="RAG-powered agent for meeting transcripts",
    version="1.0.0"
)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

VALID_TOOLS = ["semantic_search", "action_tracker", "conflict_check"]
KNOWN_NAMES = ["sai", "priya", "rahul", "kiran", "ananya"]

router_prompt = PromptTemplate.from_template("""
You are an agent with three tools. Pick one or more tools needed.
Reply with a comma-separated list only. No explanation.

Tools:
- semantic_search: questions about discussions, decisions, topics
- action_tracker: questions about tasks, owners, overdue items
- conflict_check: checking if a new decision contradicts past ones

Question: {question}
Tools:
""")

synthesize_prompt = PromptTemplate.from_template("""
Answer the user's question using the tool results below.
Be concise. Mention meeting title and date where relevant.
End with "Confidence: High/Medium/Low".

Question: {question}
Tool results:
{result}
Answer:
""")

# --- Request/Response Models ---

class IngestRequest(BaseModel):
    transcript: str
    meeting_date: str   # format: YYYY-MM-DD
    meeting_title: str

class IngestResponse(BaseModel):
    status: str
    meeting_title: str
    decisions_count: int
    action_items_count: int

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    tools_used: list[str]
    tokens_used: int
    timestamp: str

class ActionItem(BaseModel):
    task: str
    owner: str
    due: str
    meeting: str
    meeting_date: str

class MeetingSummary(BaseModel):
    title: str
    date: str
    summary: str
    decisions: list[str]

# --- Helper ---

def extract_owner(question: str) -> str:
    for word in question.split():
        if word.lower() in KNOWN_NAMES:
            return word
    return None

def execute_tool(tool: str, question: str) -> str:
    if tool == "action_tracker":
        return action_tracker(owner=extract_owner(question))
    elif tool == "conflict_check":
        return conflict_check(question)
    return semantic_search(question)

# --- Routes ---

@app.get("/")
def root():
    return {
        "name": "Meeting Intelligence Agent",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/ingest", "/query", "/meetings", "/actions", "/reset"]
    }

@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    try:
        structured = ingest_transcript(
            request.transcript,
            request.meeting_date,
            request.meeting_title
        )
        return IngestResponse(
            status="indexed",
            meeting_title=request.meeting_title,
            decisions_count=len(structured.get("decisions", [])),
            action_items_count=len(structured.get("action_items", []))
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        total_tokens = 0

        # Route
        response = llm.invoke(router_prompt.format(question=request.question))
        total_tokens += response.response_metadata.get("usage", {}).get("total_tokens", 0)

        raw_tools = response.content.strip().lower()
        tools_needed = [
            t.strip() for t in raw_tools.split(",")
            if t.strip() in VALID_TOOLS
        ]
        if not tools_needed:
            tools_needed = ["semantic_search"]

        # Execute tools
        results = []
        for tool in tools_needed:
            tool_result = execute_tool(tool, request.question)
            results.append(f"[{tool}]\n{tool_result}")

        combined = "\n\n".join(results)

        # Synthesize
        answer_response = llm.invoke(synthesize_prompt.format(
            question=request.question,
            result=combined
        ))
        total_tokens += answer_response.response_metadata.get("usage", {}).get("total_tokens", 0)

        return QueryResponse(
            question=request.question,
            answer=answer_response.content,
            tools_used=tools_needed,
            tokens_used=total_tokens,
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/meetings", response_model=list[MeetingSummary])
def list_meetings():
    try:
        vs = Chroma(
            persist_directory="./meeting_index",
            embedding_function=embeddings
        )
        all_docs = vs.get()
        meetings = []
        for metadata in all_docs["metadatas"]:
            meetings.append(MeetingSummary(
                title=metadata.get("title", ""),
                date=metadata.get("date", ""),
                summary=metadata.get("summary", ""),
                decisions=json.loads(metadata.get("decisions", "[]"))
            ))
        return meetings
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/actions", response_model=list[ActionItem])
def list_actions(owner: str = None):
    try:
        vs = Chroma(
            persist_directory="./meeting_index",
            embedding_function=embeddings
        )
        all_docs = vs.get()
        action_items = []

        for metadata in all_docs["metadatas"]:
            items = json.loads(metadata.get("action_items", "[]"))
            for item in items:
                if item.get("resolved"):
                    continue
                if owner and owner.lower() not in item.get("owner", "").lower():
                    continue
                action_items.append(ActionItem(
                    task=item.get("task", ""),
                    owner=item.get("owner", ""),
                    due=item.get("due", "not set"),
                    meeting=metadata.get("title", ""),
                    meeting_date=metadata.get("date", "")
                ))
        return action_items
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/reset")
def reset_index():
    try:
        import shutil
        shutil.rmtree("./meeting_index", ignore_errors=True)
        return {"status": "index cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))