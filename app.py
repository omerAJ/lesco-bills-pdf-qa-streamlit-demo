# app.py
import streamlit as st
import os
from langchain.schema import HumanMessage, AIMessage
from agent_factory import create_agent
from openai import OpenAI
from utils import crop_roi_to_pdf

st.title("Rent Agreement Q&A Chatbot")
st.markdown("""
Welcome! Upload your rent agreement PDFs and ask any questions about them.
""")

uploaded_files = st.file_uploader(
    "Upload rent agreement PDFs", type=["pdf"], accept_multiple_files=True
)

file_ids = []
if uploaded_files:
    st.session_state["pdfs"] = uploaded_files
    st.success(f"{len(uploaded_files)} PDF(s) uploaded.")

    with st.expander("Show uploaded PDF files"):
        st.write([f.name for f in uploaded_files])

    # Upload PDFs to OpenAI and get file IDs
    if os.environ.get("OPENAI_API_KEY"):
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        ROI_BBOX = (348, 469, 540, 610)  # <-- change these
        for pdf_file in uploaded_files:
            # Save upload to disk first
            with open(pdf_file.name, "wb") as temp:
                temp.write(pdf_file.read())

            # Crop ROI and save to new PDF
            roi_pdf_path = f"roi_{pdf_file.name}.pdf"
            crop_roi_to_pdf(pdf_file.name, roi_pdf_path, page_number=0, bbox=ROI_BBOX, dpi=500)

            # Upload ONLY the ROI-PDF
            file_obj = client.files.create(
                file=open(roi_pdf_path, "rb"),
                purpose="user_data"
            )
            file_ids.append(file_obj.id)
    else:
        file_ids = ["dummy_file_id"]

    st.session_state["file_ids"] = file_ids

# Initialize agent with file_ids as context
agent = None
if "file_ids" in st.session_state:
    agent = create_agent(context=st.session_state["file_ids"])

if "history" not in st.session_state:
    st.session_state.history = []

for entry in st.session_state.history:
    role, text = entry
    if role == "user":
        st.chat_message("user").write(text)
    else:
        st.chat_message("assistant").write(text)

if prompt := st.chat_input("Your message"):
    st.session_state.history.append(("user", prompt))
    with st.spinner("Thinking…"):
        messages = []
        for role, text in st.session_state.history:
            if role == "user":
                messages.append(HumanMessage(content=text))
            elif role == "bot":
                messages.append(AIMessage(content=text))
        messages.append(HumanMessage(content=prompt))

        ai_msg = ""
        if agent:
            resp = agent.invoke({"messages": messages})
            if isinstance(resp, dict) and "messages" in resp:
                ai_msgs = [m for m in resp["messages"] if isinstance(m, AIMessage)]
                if ai_msgs:
                    ai_msg = ai_msgs[-1].content
        st.session_state.history.append(("bot", ai_msg))
        st.rerun()