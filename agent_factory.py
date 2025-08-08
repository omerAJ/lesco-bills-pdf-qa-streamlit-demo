# agent_factory.py
import os
from langchain.schema import HumanMessage, AIMessage
from openai import OpenAI

SYSTEM_PROMPT = """
You are a careful assistant that answers questions strictly from the uploaded **electricity bill PDFs/images**.

Rules:
- When asked for the meter reading, transcribe the digits **exactly as printed**. Allowed characters: 0–9 and at most one ".".
- **Decimal validity:** Count a decimal point only if it is an illuminated LCD dot **inline between two digits** with brightness similar to the digits. Ignore specks/dust/glare, any dot far left/right of the digit column, or above/below the baseline.
- **Placement prior (critical):** If a decimal is present on this meter, it appears **immediately before the last two digits**. Never output a decimal anywhere else.
  - If you see a dot but **not** in that position, treat it as **not a decimal** and omit it.
- If the answer is not present or not readable, reply: "Sorry, I can't find that information in the provided documents."

FIRST TURN SPECIAL-CASE
If the first user question is "What is the reading on the meter?" reply concisely:
1) On the first line, output "the electricity meter reads: <reading>".
2) On the *next* line, ask: "Would you like to know anything else from the uploaded bill? e.g., the electricity cost, units consumed, or any other detail."

SUBSEQUENT TURNS
After the first turn, just answer normally while referencing the pdf for any information.
Files provided:
{context}
""".strip()



class OpenAIFilesAgent:
    def __init__(self, file_ids):
        self.file_ids = file_ids or []
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def invoke(self, *args, **kwargs):
        # Get last human question
        messages = args[0].get("messages", []) if args else []
        user_question = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                user_question = m.content
                break

        # Compose content (files + text question)
        input_content = [
            *(
                [{"type": "input_file", "file_id": fid} for fid in self.file_ids]
                if self.file_ids else []
            ),
            {"type": "input_text", "text": user_question},
        ]

        # System message MUST use input_text (not text)
        system_msg = {
            "role": "system",
            "content": [
                {
                    "type": "input_text",
                    "text": SYSTEM_PROMPT.format(
                        context="\n".join([f"- {fid}" for fid in self.file_ids]) or "- (no files listed)"
                    ),
                }
            ],
        }
        user_msg = {"role": "user", "content": input_content}

        resp = self.client.responses.create(
            model="gpt-4.1",
            input=[system_msg, user_msg],
        )

        # Be tolerant to SDK shape differences
        output_text = getattr(resp, "output_text", None)
        if not output_text:
            try:
                parts = []
                for block in getattr(resp, "output", []) or []:
                    for item in getattr(block, "content", []) or []:
                        if getattr(item, "type", None) in ("output_text", "text"):
                            parts.append(getattr(item, "text", "") or "")
                output_text = "\n".join([p for p in parts if p]) or ""
            except Exception:
                output_text = ""

        return {"messages": [AIMessage(content=output_text)]}


def create_agent(context):
    # context is a list of file_ids
    return OpenAIFilesAgent(context)
