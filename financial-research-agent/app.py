# app.py
import streamlit as st
from orchestrator import run
import time

st.set_page_config(
    page_title="Financial Research Agent",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Financial Research Agent")
st.caption("Multi-agent system — searches earnings transcripts + financial metrics simultaneously")

# Sidebar
with st.sidebar:
    st.markdown("### Companies Covered")
    st.markdown("- 🍎 Apple (AAPL)")
    st.markdown("- 🪟 Microsoft (MSFT)")
    st.markdown("- 🔵 Infosys (INFY)")
    st.markdown("### Data Sources")
    st.markdown("- 📄 Earnings call transcripts (PDF)")
    st.markdown("- 🗄️ Financial metrics (SQLite)")
    st.markdown("### Agents")
    st.markdown("- PDF Agent — hybrid search + reranker")
    st.markdown("- DB Agent — LLM-generated SQL")
    st.markdown("- Orchestrator — parallel execution + reflect")

# Example questions
st.markdown("### Try These Questions")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🍎 Apple iPhone commentary"):
        st.session_state.question = "What did Tim Cook say about iPhone revenue?"
    if st.button("📊 PE ratio comparison"):
        st.session_state.question = "What is the PE ratio of all three companies?"

with col2:
    if st.button("🤖 Microsoft AI strategy"):
        st.session_state.question = "What was Microsoft's AI strategy discussed in the earnings call?"
    if st.button("📈 Apple revenue trend"):
        st.session_state.question = "What is Apple's quarterly revenue trend?"

with col3:
    if st.button("🔵 Infosys guidance"):
        st.session_state.question = "What guidance did Infosys give for revenue growth?"
    if st.button("💰 Market cap comparison"):
        st.session_state.question = "What is the market cap of all three companies?"
        
st.markdown("---")

# Query input
question = st.text_input(
    "Ask anything about Apple, Microsoft, or Infosys:",
    value=st.session_state.get("question", ""),
    placeholder="e.g. What did Tim Cook say about iPhone revenue?"
)

if question:
    with st.spinner("🔍 Analyzing query and running agents..."):
        start = time.time()
        result = run(question)
        elapsed = time.time() - start

    # Show agents used
    agents = result.get("agents_used", [])
    agent_labels = {
        "pdf_agent": "📄 PDF Agent",
        "db_agent": "🗄️ DB Agent"
    }
    cols = st.columns(len(agents) + 2)
    for i, agent in enumerate(agents):
        cols[i].success(agent_labels.get(agent, agent))
    cols[-2].info(f"⏱️ {result['latency_ms']}ms")
    cols[-1].info(f"🔧 {len(agents)} agent(s)")

    st.markdown("---")

    # Main answer
    st.markdown("### 🤖 Answer")
    st.markdown(result["answer"])

    # Reflection
    reflection = result.get("reflection", "")
    if "APPROVED" in reflection:
        st.success(f"✅ Quality Check: {reflection}")
    else:
        st.warning(f"⚠️ Quality Check: {reflection}")

    # Show raw agent outputs in expanders
    if result.get("pdf_result"):
        with st.expander("📄 PDF Agent Raw Output"):
            st.text(result["pdf_result"])

    if result.get("db_result"):
        with st.expander("🗄️ DB Agent Raw Output"):
            st.text(result["db_result"])