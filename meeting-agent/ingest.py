from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
import json, re, os

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

extract_prompt = PromptTemplate.from_template("""
Extract structured information from this meeting transcript.
Return ONLY valid JSON, nothing else. No markdown, no explanation.

{{
  "decisions": ["list of decisions made"],
  "action_items": [
    {{"task": "...", "owner": "...", "due": "...", "resolved": false}}
  ],
  "summary": "2 sentence summary"
}}

Transcript:
{transcript}

JSON:
""")

def extract_structure(transcript: str) -> dict:
    response = llm.invoke(extract_prompt.format(transcript=transcript))
    raw = response.content
    clean = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(clean)
    except Exception as e:
        print(f"Parse error: {e}")
        print(f"Raw output: {raw}")
        return {"decisions": [], "action_items": [], "summary": raw}

def ingest_transcript(transcript: str, meeting_date: str, meeting_title: str):
    print(f"\nIngesting: {meeting_title}...")
    structured = extract_structure(transcript)
    print(f"Extracted: {len(structured['decisions'])} decisions, "
          f"{len(structured['action_items'])} action items")

    doc = Document(
        page_content=transcript,
        metadata={
            "date": meeting_date,
            "title": meeting_title,
            "decisions": json.dumps(structured["decisions"]),
            "action_items": json.dumps(structured["action_items"]),
            "summary": structured["summary"]
        }
    )

    vectorstore = Chroma(
        persist_directory="./meeting_index",
        embedding_function=embeddings
    )
    vectorstore.add_documents([doc])
    print(f"Indexed: {meeting_title}")
    return structured