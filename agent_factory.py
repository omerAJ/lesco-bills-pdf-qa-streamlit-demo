# agent_factory.py
import os
from langchain.schema import HumanMessage, AIMessage
from openai import OpenAI

SYSTEM_PROMPT = """
You are a careful assistant that answers questions strictly from the uploaded **electricity bill PDFs** (cropped ROI pages).
Rules:
- Use only the information visible in the provided PDFs.
- When asked for the meter reading, read the digits **exactly as printed**, preserving any decimal point. Do not round or normalize.
- If the reading is ambiguous or the decimal point is unclear, say so briefly and give the best reading you see.
- If the answer is not present or not readable, reply: "Sorry, I can't find that information in the provided documents."

FIRST TURN SPECIAL-CASE
If the first user question is "What is the reading on the meter?" reply concisely:
1) On the first line, output only the reading (digits, preserving any decimal point).
2) On the second line, ask: "Would you like to know anything else from the uploaded bill?"

SUBSEQUENT TURNS
After the first turn, just answer normally while following the rules above.
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

        # FIX: use input_text (not text) for system content
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
                for block in resp.output or []:
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
