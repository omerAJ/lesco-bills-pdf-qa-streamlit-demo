# agent_factory.py
import os
from langchain.schema import HumanMessage, AIMessage
from openai import OpenAI

SYSTEM_PROMPT = """
You are a helpful assistant that answers questions strictly based on the provided PDF rent agreements.
Use only the information from the following PDF files:
{context}
If the answer is not present in the PDFs, reply: "Sorry, I can't find that information in the provided documents."
Do not make up information or provide advice beyond what is contained in the PDFs.
"""

class OpenAIFilesAgent:
    def __init__(self, file_ids):
        self.file_ids = file_ids
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def invoke(self, *args, **kwargs):
        messages = args[0].get("messages", []) if args else []
        user_question = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                user_question = m.content
                break
        input_content = [
            *(
                [{"type": "input_file", "file_id": fid} for fid in self.file_ids]
                if self.file_ids else []
            ),
            {"type": "input_text", "text": user_question}
        ]
        resp = self.client.responses.create(
            model="gpt-4.1",
            input=[{"role": "user", "content": input_content}]
        )
        return {"messages": [AIMessage(content=resp.output_text)]}

def create_agent(context):
    # context is a list of file_ids
    return OpenAIFilesAgent(context)