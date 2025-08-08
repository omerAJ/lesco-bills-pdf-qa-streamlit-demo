# app.py
import streamlit as st
import os
from langchain.schema import HumanMessage, AIMessage
from agent_factory import create_agent
from openai import OpenAI
from utils import crop_roi_to_pdf

st.set_page_config(page_title="Electricity Bill Q&A", page_icon="⚡")
st.title("⚡ Electricity Bill Q&A")
st.markdown(
    "Upload your **bill PDF**. We'll crop the meter area automatically and ask the model:\n"
    "_“What is the reading on the meter?”_  \n"
    "Then you can continue chatting to ask anything else from the same bill."
)

# ---- Session State ----
if "history" not in st.session_state:
    st.session_state.history = []
if "file_ids" not in st.session_state:
    st.session_state.file_ids = []
if "initial_query_done" not in st.session_state:
    st.session_state.initial_query_done = False

# ---- Constants ----
# Change this ROI to match your meter area (PDF coordinate space: points, origin bottom-left)
ROI_BBOX = (348, 469, 540, 610)  # (x0, y0, x1, y1)
INITIAL_QUESTION = "What is the reading on the meter?"

# ---- File Uploader (PDF only) ----
uploaded_file = st.file_uploader(
    "Upload electricity bill PDF", type=["pdf"], accept_multiple_files=False
)

# ---- On Upload: crop -> upload to OpenAI -> auto-ask initial question ----
if uploaded_file and not st.session_state.file_ids:
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("Missing OPENAI_API_KEY in environment. Please set it and refresh.")
        st.stop()

    with st.spinner("Processing PDF (cropping ROI)…"):
        # Save the uploaded PDF to disk
        local_pdf_path = uploaded_file.name
        with open(local_pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        # Crop ROI and save as a new (single-page) PDF for vision
        roi_pdf_path = f"roi_{uploaded_file.name}.pdf"
        crop_roi_to_pdf(local_pdf_path, roi_pdf_path, page_number=0, bbox=ROI_BBOX, dpi=500)

        # Upload the cropped ROI-PDF to OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        file_obj = client.files.create(
            file=open(roi_pdf_path, "rb"),
            purpose="user_data",  # keeps parity with Responses API usage
        )
        st.session_state.file_ids = [file_obj.id]

    # Create agent & auto-ask the initial meter-reading question
    agent = create_agent(context=st.session_state.file_ids)
    st.session_state.history.append(("user", INITIAL_QUESTION))

    with st.spinner("Asking the model for the meter reading…"):
        # Build the LangChain-style message history for the agent
        messages = []
        for role, text in st.session_state.history:
            if role == "user":
                messages.append(HumanMessage(content=text))
            elif role == "bot":
                messages.append(AIMessage(content=text))
        # Final user message (the initial question)
        messages.append(HumanMessage(content=INITIAL_QUESTION))

        resp = agent.invoke({"messages": messages})
        ai_msg = ""
        if isinstance(resp, dict) and "messages" in resp:
            ai_msgs = [m for m in resp["messages"] if isinstance(m, AIMessage)]
            if ai_msgs:
                ai_msg = ai_msgs[-1].content or ""

        # Ensure the follow-up line is present for the very first response
        followup = "Would you like to know anything else from the uploaded bill?"
        if followup.lower() not in ai_msg.lower():
            if ai_msg.strip():
                ai_msg = f"{ai_msg.strip()}\n\n{followup}"
            else:
                ai_msg = followup

        st.session_state.history.append(("bot", ai_msg))
        st.session_state.initial_query_done = True
        st.rerun()

# ---- Show chat history (after upload) ----
if st.session_state.history:
    for entry in st.session_state.history:
        role, text = entry
        if role == "user":
            st.chat_message("user").write(text)
        else:
            st.chat_message("assistant").write(text)

# ---- Chat input (enabled only after a PDF is uploaded) ----
if st.session_state.file_ids:
    if prompt := st.chat_input("Ask anything else from this bill…"):
        st.session_state.history.append(("user", prompt))
        with st.spinner("Thinking…"):
            # Recreate agent each turn to ensure it has the latest file_ids
            agent = create_agent(context=st.session_state.file_ids)

            # Build history for the agent call
            messages = []
            for role, text in st.session_state.history:
                if role == "user":
                    messages.append(HumanMessage(content=text))
                elif role == "bot":
                    messages.append(AIMessage(content=text))
            messages.append(HumanMessage(content=prompt))

            ai_msg = ""
            resp = agent.invoke({"messages": messages})
            if isinstance(resp, dict) and "messages" in resp:
                ai_msgs = [m for m in resp["messages"] if isinstance(m, AIMessage)]
                if ai_msgs:
                    ai_msg = ai_msgs[-1].content or ""

            st.session_state.history.append(("bot", ai_msg))
            st.rerun()
else:
    st.info("Please upload a PDF to begin.")
