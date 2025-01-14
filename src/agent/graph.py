"""Define a simple chatbot agent.

This agent returns a predefined response without using an actual LLM.
"""

from typing import Any, Dict, Literal, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END, START

from src.agent.configuration import Configuration
from src.agent.state import AgentState, InputState, Router
from src.agent.rag_self_reflection.graph import graph as rag_self_reflection_graph

from src.shared.utils import load_chat_model

async def analyze_and_route_query(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    configuration = Configuration.from_runnable_config(config)
    # configuration = Configuration.from_runnable_config(config)
    print(configuration)
    print(configuration.query_model)
    model = load_chat_model(configuration.query_model)
    print(model)
    messages = [
        {"role": "system", "content": configuration.router_system_prompt}
    ] + state.messages
    print("Messages")
    print(messages)
    response = cast(
        Router, await model.with_structured_output(Router).ainvoke(messages)
    )
    print("Response")
    print(response)
    return {"router": response}

def route_query(state: AgentState) -> Literal["create_research_plan", "ask_for_more_info", "respond_to_general_query"]:
    """Determine the next step based on the query classification.

    Args:
        state (AgentState): The current state of the agent, including the router's classification.

    Returns:
        Literal["create_research_plan", "ask_for_more_info", "respond_to_general_query"]: The next step to take.

    Raises:
        ValueError: If an unknown router type is encountered.
    """
    _type = state.router["type"]
    if _type == "movie":
        return "create_research_plan"
    elif _type == "more-info":
        return "ask_for_more_info"
    elif _type == "general":
        return "respond_to_general_query"
    else:
        raise ValueError(f"Unknown router type {_type}")


async def create_research_plan(state: AgentState, *, config: RunnableConfig) -> dict[str, list[str] | str]:
    print("Creating research plan")
    print(state.messages[0])
    question_content = state.messages[0].content
    result = await rag_self_reflection_graph.ainvoke({"question": question_content})
    print(result)
    return {"steps": "result"}

async def ask_for_more_info(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    print("Asking for more info")
    return {"router": "more-info"}

async def respond_to_general_query(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    print("Responding to general query")
    return {"router": "general"}
# Define a new graph
workflow = StateGraph(AgentState, input=InputState, config_schema=Configuration)

# Add the node to the graph
workflow.add_node(analyze_and_route_query)
workflow.add_node(create_research_plan)
workflow.add_node("ask_for_more_info", ask_for_more_info)
workflow.add_node("respond_to_general_query", respond_to_general_query)
# Set the entrypoint as `call_model`
workflow.add_edge(START, "analyze_and_route_query")
workflow.add_conditional_edges("analyze_and_route_query", route_query)
workflow.add_edge("create_research_plan", END)
workflow.add_edge("ask_for_more_info", END)
workflow.add_edge("respond_to_general_query", END)
# Compile the workflow into an executable graph
graph = workflow.compile()

graph.name = "New Graph"  # This defines the custom name in LangSmith
