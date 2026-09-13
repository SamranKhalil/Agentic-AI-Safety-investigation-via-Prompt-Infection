from langchain_google_genai import ChatGoogleGenerativeAI
from config import (
    MODEL_NAME_1,
    MODEL_NAME_2,
    MODEL_NAME_3,
    MODEL_NAME_4,
    MODEL_NAME_5,
    GEMINI_API_KEY,
    GROQ_API_KEY,
)
from langchain_groq import ChatGroq


# initialize LLM
def get_llm(name, tools=None, groq_api_key=None):
    if name == "gemini":
        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME_1, google_api_key=GEMINI_API_KEY, temperature=0.0
        )
    elif name == "qwen":
        llm = ChatGroq(
            model=MODEL_NAME_2, api_key=groq_api_key or GROQ_API_KEY, temperature=0.0
        )
    elif name == "llama":
        llm = ChatGroq(
            model=MODEL_NAME_3, api_key=groq_api_key or GROQ_API_KEY, temperature=0.0
        )
    elif name == "openai120b":
        llm = ChatGroq(
            model=MODEL_NAME_4, api_key=groq_api_key or GROQ_API_KEY, temperature=0.0
        )
    elif name == "openai20b":
        llm = ChatGroq(
            model=MODEL_NAME_5, api_key=groq_api_key or GROQ_API_KEY, temperature=0.0
        )
    else:
        raise ValueError(f"Unknown model: {name}")

    if tools:
        llm = llm.bind_tools(tools)

    return llm


def extract_content(response) -> str:
    content = response.content
    if isinstance(content, list):
        return " ".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and "text" in block
        )
    return content
