from langgraph.graph import StateGraph, END
from agents.reader import reader_agent
from agents.scam_pipeline.summarizer import summarizer_agent
from agents.scam_pipeline.writer import writer_agent
from agents.theft_pipeline.db_manager import db_manager_agent
from agents.theft_pipeline.coder import coder_agent
from graph.state import AgentState

# building the graph
def build_graph(name):
    graph = StateGraph(AgentState)

    # nodes
    if name == "scam":
        graph.add_node("reader",     reader_agent)
        graph.add_node("summariser", summarizer_agent)
        graph.add_node("writer",     writer_agent)

        #edges
        graph.set_entry_point("reader")
        graph.add_edge("reader", "summariser")
        graph.add_edge("summariser", "writer")
        graph.add_edge("writer", END)
        
    elif name == "theft":
        graph.add_node("reader",      reader_agent)
        graph.add_node("db_manager",  db_manager_agent)
        graph.add_node("coder",       coder_agent)

        graph.set_entry_point("reader")
        graph.add_edge("reader",     "db_manager")
        graph.add_edge("db_manager", "coder")
        graph.add_edge("coder",      END)

    # workflow = graph.compile()
    # print(workflow.get_graph().print_ascii())
    return graph.compile()