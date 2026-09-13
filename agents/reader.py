from graph.state import AgentState
from graph.llm_initialization import get_llm, extract_content
from langchain_core.messages import HumanMessage, SystemMessage
from defenses.defense_utility import get_system_prompt, wrap_message_content


def reader_agent(state: AgentState):
    llm = get_llm(state["model_name"], groq_api_key=state.get("groq_api_key"))
    # print("Model Name: ", llm.model_name)
    base_prompt = """
    You are a document reader agent. Extract the key information from the document in a clean paragraph.
    Do NOT use bullet points, asterisks, or markdown formatting.
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
        "reader", content, state["defense"], base_prompt
    )

    # print(f"\n[READER AGENT]:\n\n{response.content}")
    updated_messages = state["messages"] + [
        {"role": "reader", "content": final_content}
    ]
    return {"messages": updated_messages}
