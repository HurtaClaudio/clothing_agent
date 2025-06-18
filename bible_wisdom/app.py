import streamlit as st
from langchain_core.messages import HumanMessage
from bible_rag import get_graph

st.set_page_config(page_title="📜 Bible Wisdom Chat", layout="centered")
st.title("📜 Ask the Bible")

qa_graph = get_graph()

# UI for user input
question = st.text_input("Ask a question about the Bible")

if st.button("Send") and question:
    print(question)
    print(HumanMessage(content=question))
    result = qa_graph.invoke({"messages": HumanMessage(content=question)})
    st.markdown(f"**Answer:** {result['answer']}")

if hasattr(qa_graph, "memory") and hasattr(qa_graph.memory, "buffer"):
    with st.expander("📚 Chat History"):
        st.markdown(qa_graph.memory.buffer)
