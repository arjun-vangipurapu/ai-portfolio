from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from tools import semantic_search, action_tracker, conflict_check
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

VALID_TOOLS = ["semantic_search", "action_tracker", "conflict_check"]

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

test_cases = [
    {
        "question": "What did we decide about GraphQL?",
        "expected_tool": "semantic_search",
        "expected_keywords": ["graphql", "mobile", "bff"]
    },
    {
        "question": "What action items are open for Sai?",
        "expected_tool": "action_tracker",
        "expected_keywords": ["sai", "grpc", "benchmark"]
    },
    {
        "question": "We want to use REST for all new services — any conflicts?",
        "expected_tool": "conflict_check",
        "expected_keywords": ["conflict", "grpc"]
    },
    {
        "question": "What did we decide and what is pending for Priya?",
        "expected_tool": "semantic_search,action_tracker",
        "expected_keywords": ["priya", "api", "doc"]
    }
]

def run_evals():
    print("\n🧪 Running evals...\n")
    passed = 0

    for case in test_cases:
        response = llm.invoke(router_prompt.format(question=case["question"]))
        tool = response.content.strip().lower()

        tool_correct = all(
            t in tool for t in case["expected_tool"].split(",")
        )

        # check answer quality
        if "action_tracker" in tool:
            result = action_tracker()
        elif "conflict_check" in tool:
            result = conflict_check(case["question"])
        else:
            result = semantic_search(case["question"])

        keyword_hits = sum(
            1 for kw in case["expected_keywords"]
            if kw.lower() in result.lower()
        )
        keyword_score = keyword_hits / len(case["expected_keywords"])

        status = "✅" if tool_correct else "❌"
        print(f"{status} [{case['question'][:45]}...]")
        print(f"   Tool: expected={case['expected_tool']} got={tool.strip()}")
        print(f"   Keywords: {keyword_hits}/{len(case['expected_keywords'])} found")
        print()

        if tool_correct:
            passed += 1

    print(f"─" * 50)
    print(f"Routing score: {passed}/{len(test_cases)}")

if __name__ == "__main__":
    run_evals()