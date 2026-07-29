\# Meeting Intelligence Agent



A multi-tool RAG agent that answers questions across meeting transcripts.

Built with a manual agent loop — no frameworks hiding the internals.



\## The Problem

Teams lose institutional memory. Decisions made 3 weeks ago get forgotten.

Action items fall through. New decisions contradict old ones.

Nobody has a good tool for this.



\## What It Does



| Question type | Example | Tool used |

|---|---|---|

| Semantic search | "What did we decide about GraphQL?" | `semantic\_search` |

| Action tracking | "What's overdue for Sai?" | `action\_tracker` |

| Conflict detection | "We want to use REST — any conflicts?" | `conflict\_check` |



\## Demo



```

You: What did we decide about GraphQL?

🔧 Tool: semantic\_search

🤖 In the \[2025-05-10 — API Strategy Review] meeting, it was decided 

&#x20;  to evaluate GraphQL only for mobile BFF. GraphQL will not be used 

&#x20;  for internal services.



You: We want to use REST for all new services — any conflicts?

🔧 Tool: conflict\_check

🤖 Yes — conflicts with two past decisions: gRPC migration planned for 

&#x20;  Q3/Q4, and all new services explicitly set to use gRPC from 2025-05-20.

```



\## Architecture



```

User question

&#x20;     ↓

LLM router (Mistral) — decides which tool to call

&#x20;     ↓

├── semantic\_search  → ChromaDB vector search

├── action\_tracker   → ChromaDB metadata filter  

└── conflict\_check   → ChromaDB + LLM reasoning

&#x20;     ↓

LLM synthesizes final answer with meeting references

```



\## What I Learned Building This



\- \*\*Tool routing without frameworks\*\* — the LLM decides which tool

&#x20; to call based on intent. Understanding this manually makes LangGraph

&#x20; and ADK make sense.

\- \*\*Metadata-enriched RAG\*\* — storing structured data (decisions,

&#x20; action items) alongside embeddings unlocks hybrid retrieval.

\- \*\*Prompt chaining\*\* — three prompts in sequence:

&#x20; extract → route → synthesize. Each output feeds the next.

\- \*\*Where agents actually fail\*\* — malformed JSON from LLM extraction,

&#x20; zero retrieval results, wrong tool routing. Happy path is easy.

&#x20; Reliability is hard.



\## Stack



\- \*\*LLM:\*\* Ollama + Mistral 7B (local, free)

\- \*\*Embeddings:\*\* Sentence Transformers `all-MiniLM-L6-v2`

\- \*\*Vector store:\*\* ChromaDB

\- \*\*Orchestration:\*\* LangChain

\- \*\*Language:\*\* Python 3.11



\## Run It Yourself



```bash

\# 1. Install dependencies

pip install langchain langchain-community langchain-ollama \\

langchain-chroma langchain-core chromadb sentence-transformers



\# 2. Pull the model

ollama pull mistral



\# 3. Index sample transcripts

python seed\_data.py



\# 4. Run the agent

python agent.py

```



\## What's Next



\- Rebuild with Google ADK + Gemini 2.0 Flash

\- Add streaming responses

\- Add Streamlit UI for demo

\- Eval harness to measure routing accuracy



\## Part of



\[ai-portfolio](https://github.com/arjun-vangipurapu/ai-portfolio) — 

a series of production AI systems built while transitioning 

from senior full-stack to AI engineer.

