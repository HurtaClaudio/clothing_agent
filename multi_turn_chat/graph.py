from langgraph.graph import END
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START

model = ChatOpenAI(model="gpt-4o",temperature=0)

class State(MessagesState):
    summary: str
    category: str

def is_about_chess_or_life_or_else(state: State):
    """Return the category of the latest user message: 'chess', 'life', or 'other'."""
    messages = state["messages"]
    if not messages:
        return "other"
    user_message = messages[-1].content
    # Use the LLM to classify the message
    prompt = (
        "Classify the following user message as one of: 'chess', 'life', or 'other'. "
        "Only return the category name.\n"
        f"User message: {user_message}"
    )
    category = model.invoke(prompt).content.strip().lower()

    if category in {"chess", "life", "other"}:
        return {"category": category}
    return {"category": "other"}


# Define the logic to call the model for chess

def call_model_chess(state: State):
    summary = state.get("summary", "")
    system_prompt = (
        "You are a world-class chess expert. Answer the user's question with deep chess knowledge, "
        "using precise terminology and offering strategic insights."
    )
    system_message = SystemMessage(content=system_prompt)
    messages = [system_message]
    if summary:
        messages.append(SystemMessage(content=f"Summary of conversation earlier: {summary}"))
    messages += state["messages"]
    response = model.invoke(messages)
    return {"messages": response}

# Define the logic to call the model for life advice

def call_model_life(state: State):
    summary = state.get("summary", "")
    system_prompt = (
        "You are a supportive, understanding friend. Answer the user's question with empathy, "
        "warmth, and encouragement."
    )
    system_message = SystemMessage(content=system_prompt)
    messages = [system_message]
    if summary:
        messages.append(SystemMessage(content=f"Summary of conversation earlier: {summary}"))
    messages += state["messages"]
    response = model.invoke(messages)
    return {"messages": response}

# Define the logic to call the model for other topics

def call_model_other(state: State):
    summary = state.get("summary", "")
    system_prompt = (
        "You are a helpful and knowledgeable assistant. Answer the user's question clearly and helpfully, "
        "drawing on general knowledge and best practices."
    )
    system_message = SystemMessage(content=system_prompt)
    messages = [system_message]
    if summary:
        messages.append(SystemMessage(content=f"Summary of conversation earlier: {summary}"))
    messages += state["messages"]
    response = model.invoke(messages)
    return {"messages": response}

def summarize_conversation(state: State):
    
    # First, we get any existing summary
    summary = state.get("summary", "")

    # Create our summarization prompt 
    if summary:
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
        
    else:
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    
    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}

# Determine whether to end or summarize the conversation
def should_summarize(state: State):
    
    """Return the next node to execute."""
    
    messages = state["messages"]
    
    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"
    
    # Otherwise we can just end
    return END

def route_by_category(state: State):
    """
    Determines the next node in the workflow based on the 'category' field in the state.
    """
    category = state["category"]
    return category


def neutral_node(state: State):
    """A neutral node that just receives input and passes state through unchanged."""
    return {}

def get_graph():
    workflow = StateGraph(State)
    workflow.add_node("router", is_about_chess_or_life_or_else)
    workflow.add_node("chess", call_model_chess)
    workflow.add_node("life", call_model_life)
    workflow.add_node("other", call_model_other)
    workflow.add_node("neutral_node", neutral_node)
    workflow.add_node("summarize_conversation", summarize_conversation)

    # Start the graph with the router node
    workflow.add_edge(START, "router")

    # Route from router to either chess, life, or other
    workflow.add_conditional_edges("router", route_by_category)

    workflow.add_edge("chess", "neutral_node")
    workflow.add_edge("life", "neutral_node")
    workflow.add_edge("other", "neutral_node")


    # After each, go to should_summarize
    workflow.add_conditional_edges("neutral_node", should_summarize)
    workflow.add_edge("summarize_conversation", END)

    # Compile
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    return graph