from typing import TypedDict


# State definition
class AgentState(TypedDict):
    messages: list
    messaging_mode: str
    model_name: str
    tool_called: bool
    infected: bool
    defense: str
    groq_api_key: str
