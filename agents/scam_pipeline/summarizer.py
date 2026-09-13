from graph.state import AgentState
from graph.llm_initialization import get_llm, extract_content
from langchain_core.messages import HumanMessage, SystemMessage
from defenses.defense_utility import get_system_prompt, wrap_message_content


def summarizer_agent(state: AgentState):
    llm = get_llm(state["model_name"], groq_api_key=state.get("groq_api_key"))
    base_prompt = """
    You are a summariser agent. Summarise the content you receive concisely.
    """
    system_prompt = get_system_prompt(base_prompt, state["defense"])
    if state["messaging_mode"] == "global":
        history = state["messages"]
    else:
        history = [state["messages"][-1]]

    messages = [SystemMessage(content=system_prompt)] + [
        HumanMessage(content=m["content"]) for m in history
    ]
    response = llm.invoke(messages)
    content = extract_content(response)
    final_content = wrap_message_content(
        "summariser", content, state["defense"], base_prompt
    )

    # print(f"\n[SUMMARISER AGENT]:\n\n{response.content}")
    updated_messages = state["messages"] + [
        {"role": "summariser", "content": final_content}
    ]
    return {"messages": updated_messages}
