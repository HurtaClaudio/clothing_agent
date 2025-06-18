import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from bible_rag import get_graph

st.set_page_config(page_title="📜 Bible Wisdom Chat", layout="centered")
st.title("📜 Ask the Bible")

# Load LangGraph
qa_graph = get_graph()

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
question = st.text_input("Ask a question about the Bible")

if st.button("Send") and question:
    # Rebuild messages from history
    messages = []
    for q, a in st.session_state.chat_history:
        messages.append(HumanMessage(content=q))
        messages.append(AIMessage(content=a))

    # Add current question
    messages.append(HumanMessage(content=question))

    # Invoke LangGraph
    result = qa_graph.invoke({"messages": messages})
    answer = result["answer"]

    # Update chat history
    st.session_state.chat_history.append((question, answer))

    # Display the answer
    st.markdown(f"**Answer:** {answer}")

# Show full chat history
if st.session_state.chat_history:
    with st.expander("📚 Chat History", expanded=False):
        for i, (q, a) in enumerate(st.session_state.chat_history):
            st.markdown(f"**Q{i+1}:** {q}")
            st.markdown(f"**A{i+1}:** {a}")