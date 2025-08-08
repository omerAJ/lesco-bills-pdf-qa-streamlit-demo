# agent_factory.py
import os, json, re
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

SYSTEM_PROMPT = """
You are a helpful assistant that answers questions strictly based on the provided PDF rent agreements.
Use only the information from the following PDF content:
{context}
If the answer is not present in the PDFs, reply: "Sorry, I can't find that information in the provided documents."
Do not make up information or provide advice beyond what is contained in the PDFs.
"""

def create_agent() -> object:
    llm = ChatOpenAI(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        model_name="gpt-4.1-nano",
        temperature=0.0
    )
    checkpointer = InMemorySaver()
    agent = create_react_agent(
        model=llm,
        checkpointer=checkpointer,
        prompt=SYSTEM_PROMPT,
        debug=False,
    )
    return agent