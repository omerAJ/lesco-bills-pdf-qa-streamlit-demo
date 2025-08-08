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
    "Upload your **electricity bill PDF**."
)

# ---- Session State ----
if "history" not in st.session_state:
    st.session_state.history = []
if "file_ids" not in st.session_state:
    st.session_state.file_ids = []  # used by agent each turn
if "initial_query_done" not in st.session_state:
    st.session_state.initial_query_done = False
if "roi_file_id" not in st.session_state:
    st.session_state.roi_file_id = None
if "full_pdf_path" not in st.session_state:
    st.session_state.full_pdf_path = None
if "full_file_id" not in st.session_state:
    st.session_state.full_file_id = None

# ---- Constants ----
# Change this ROI to match your meter area (PDF coordinate space: points, origin bottom-left)
ROI_BBOX = (348, 469, 540, 610)  # (x0, y0, x1, y1)
INITIAL_QUESTION = "What is the reading on the meter? Focus on the decimal point."

# ---- Helpers ----
def swap_roi_for_full_pdf():
    """Delete the ROI file from OpenAI and upload the full PDF. Update session_state.file_ids."""
    if not os.environ.get("OPENAI_API_KEY"):
        return
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Delete ROI file if present
    if st.session_state.roi_file_id:
        try:
            client.files.delete(st.session_state.roi_file_id)
        except Exception:
            # Soft-fail; continue even if deletion fails
            pass
        finally:
            st.session_state.roi_file_id = None

    # Upload full PDF and switch context
    if st.session_state.full_pdf_path and os.path.exists(st.session_state.full_pdf_path):
        try:
            with open(st.session_state.full_pdf_path, "rb") as f:
                full_obj = client.files.create(file=f, purpose="user_data")
            st.session_state.full_file_id = full_obj.id
            st.session_state.file_ids = [full_obj.id]
        except Exception as e:
            st.error(f"Failed to upload full PDF: {e}")

# ---- File Uploader (PDF only) ----
uploaded_file = st.file_uploader(
    "Upload electricity bill PDF", type=["pdf"], accept_multiple_files=False
)

# ---- On Upload: crop -> upload ROI to OpenAI -> auto-ask initial question -> swap to full PDF ----
if uploaded_file and not st.session_state.initial_query_done and not st.session_state.file_ids:
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("Missing OPENAI_API_KEY in environment. Please set it and refresh.")
        st.stop()

    with st.spinner("Processing PDF (cropping ROI)…"):
        # Save the uploaded PDF to disk
        local_pdf_path = uploaded_file.name
        with open(local_pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        st.session_state.full_pdf_path = local_pdf_path

        # Crop ROI and save as a new (single-page) PDF for vision
        roi_pdf_path = f"roi_{uploaded_file.name}.pdf"
        crop_roi_to_pdf(local_pdf_path, roi_pdf_path, page_number=0, bbox=ROI_BBOX, dpi=500)

        # Upload the cropped ROI-PDF to OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        with open(roi_pdf_path, "rb") as f:
            roi_obj = client.files.create(file=f, purpose="user_data")
        st.session_state.roi_file_id = roi_obj.id
        st.session_state.file_ids = [roi_obj.id]

    # Create agent & auto-ask the initial meter-reading question (using ROI)
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
        if followup.lower() not in (ai_msg or "").lower():
            if (ai_msg or "").strip():
                ai_msg = f"{ai_msg.strip()}\n\n{followup}"
            else:
                ai_msg = followup

        st.session_state.history.append(("bot", ai_msg))

        # Swap ROI file for full PDF for all subsequent turns
        swap_roi_for_full_pdf()

        st.session_state.initial_query_done = True
        st.rerun()

# ---- Show chat history (after upload) ----
if st.session_state.history:
    for role, text in st.session_state.history:
        if role == "user":
            st.chat_message("user").write(text)
        else:
            st.chat_message("assistant").write(text)

# ---- Chat input (enabled only after a PDF is uploaded & initial turn is done) ----
if st.session_state.full_file_id:
    if prompt := st.chat_input("Ask anything else from this bill…"):
        st.session_state.history.append(("user", prompt))
        with st.spinner("Thinking…"):
            # Recreate agent each turn to ensure it has the latest (full) file_ids
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
