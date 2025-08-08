# app.py
import streamlit as st
from langchain.schema import HumanMessage, AIMessage
from agent_factory import create_agent
import PyPDF2

# Initialize agent once
# @st.cache_resource
def get_agent():
    return create_agent()

agent = get_agent()

st.title("Rent Agreement Q&A Chatbot")
st.markdown("""
Welcome! Upload your rent agreement PDFs and ask any questions about them.
""")

# PDF upload
uploaded_files = st.file_uploader(
    "Upload rent agreement PDFs", type=["pdf"], accept_multiple_files=True
)
pdf_text = ""
if uploaded_files:
    st.session_state["pdfs"] = uploaded_files
    st.success(f"{len(uploaded_files)} PDF(s) uploaded.")
    # Extract text from all uploaded PDFs
    for pdf_file in uploaded_files:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            pdf_text += page.extract_text() or ""
    st.session_state["pdf_text"] = pdf_text

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Render past messages
for entry in st.session_state.history:
    role, text = entry
    if role == "user":
        st.chat_message("user").write(text)
    else:
        st.chat_message("assistant").write(text)

# Input box
if prompt := st.chat_input("Your message"):
    # Record user turn
    st.session_state.history.append(("user", prompt))
    with st.spinner("Thinking…"):
        # Build full chat history for agent
        messages = []
        for role, text in st.session_state.history:
            if role == "user":
                messages.append(HumanMessage(content=text))
            elif role == "bot":
                messages.append(AIMessage(content=text))
        messages.append(HumanMessage(content=prompt))

        # Pass PDF text as context to the agent
        context = st.session_state.get("pdf_text", "")
        resp = agent.invoke(
                {"messages": messages, "context": context},
                config={"configurable": {"thread_id": "user_thread"}}
            )
        # Extract the final AIMessage text
        ai_msg = None
        if isinstance(resp, dict) and "messages" in resp:
            # Filter out AIMessage objects
            ai_msgs = [m for m in resp["messages"] if isinstance(m, AIMessage)]
            if ai_msgs:
                ai_msg = ai_msgs[-1].content
            else:
                ai_msg = ""
        else:
            ai_msg = str(resp)
    # Record bot turn
    st.session_state.history.append(("bot", ai_msg))
    # Rerun to display
    st.rerun()