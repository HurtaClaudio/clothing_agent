import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from graph import get_graph

config = {"configurable": {"thread_id": "1"}}

st.set_page_config(page_title="♟️ Multi-Turn Chess/Friend Chatbot", layout="centered")
st.title("♟️ Chess Expert & Friendly Chatbot")

# Load LangGraph
if 'graph' not in st.session_state:
    st.session_state.graph = get_graph()

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.text_input("Ask me anything (chess or life questions!)")

if st.button("Send") and user_input:
    # Build the full message history
    messages = []
    for q, a in st.session_state.chat_history:
        messages.append(HumanMessage(content=q))
        messages.append(AIMessage(content=a))
    messages.append(HumanMessage(content=user_input))

    # Invoke LangGraph with the full history
    output = st.session_state.graph.invoke({"messages": messages}, config)
    answer = output['messages'][-1:][0].content

    # Update chat history
    st.session_state.chat_history.append((user_input, answer))

    # Display the answer
    st.markdown(f"**Answer:** {answer}")

# Show full chat history
if st.session_state.chat_history:
    with st.expander("📝 Chat History", expanded=False):
        for i, (q, a) in enumerate(st.session_state.chat_history):
            st.markdown(f"**Q{i+1}:** {q}")
            st.markdown(f"**A{i+1}:** {a}")


# aprender a usar streamlit mejor
# aprender un framework de java
# que es mas rápido? xd
