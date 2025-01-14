from typing import Any, Dict, Literal, cast, TypedDict

from langchain_core.runnables import RunnableConfig
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, END, START
from langgraph.types import Command
from src.shared.utils import load_chat_model, format_docs
from langchain_core.messages import AIMessage
from src.hierarchical_graph.configuration import Configuration
from src.hierarchical_graph.state import AgentState, InputState

async def research_supervisor_node(
    state: AgentState, *,
    config: RunnableConfig
) -> Command[Literal["search", "web_scraper", "__end__"]]:
    """
    Supervisor node for research tasks.
    """
    # Load configuration and LLM
    configuration = Configuration.from_runnable_config(config)
    print(configuration)
    llm = load_chat_model(configuration.llm_router_model)

    # Define workers managed by the supervisor
    members = ["search", "web_scraper"]

    # Dynamically generate the supervisor function
    supervisor_func = make_supervisor_node(llm, members)

    # Call the supervisor function with the current state
    return supervisor_func(state)


def make_supervisor_node(llm: BaseChatModel, members: list[str]) -> str:
    options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]

    def supervisor_node(state: AgentState) -> Command[Literal[*members, "__end__"]]:
        """An LLM-based router."""
        print("\n[Supervisor Node] Current State:")
        print(state)
        print(state.messages)
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state.messages
        print("\n[Supervisor Node] Messages:")
        print(messages)
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
        if goto == "FINISH":
            print("[Supervisor Node] Finished")
            goto = END

        return Command(goto=goto)

    return supervisor_node

def search_node(state: AgentState, *, config: RunnableConfig) -> Command[Literal["supervisor"]]:
    configuration = Configuration.from_runnable_config(config)
    print(configuration)
    print("\n[Search Node] Current State:")
    print(state)

    # Mock update
    result = {
        "result": "Found top 5 AI research papers of 2023",
        "status": "completed"
    }
    print("[Search Node] Returning Updated Data:")
    print(result)
    # Add the result to the state.messages
    new_message = AIMessage(content=f"Search result: {result['result']}")
    updated_messages = state.messages + [new_message]

    return Command(
        update = {
            "search_response": result,
            "messages": updated_messages
            },
        
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )

def web_scraper_node(state: AgentState, *, config: RunnableConfig) -> Command[Literal["supervisor"]]:
    configuration = Configuration.from_runnable_config(config)
    print(configuration)
    print("\n[Web Scraper Node] Current State:")
    print(state.web_scraper_response)

    # Mock update
    result = {
        "result": "Scraped abstracts: [abstract 1, abstract 2, ...]",
        "status": "completed"
    }
    print("[Web Scraper Node] Returning Updated Data:")
    print(result)

    new_message = AIMessage(content=f"Web Scraper result: {result['result']}")
    updated_messages = state.messages + [new_message]
    return Command(
        update={
            "web_scraper_response": result,
            "messages": updated_messages},
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )

research_builder = StateGraph(AgentState, input=InputState, config_schema=Configuration)
research_builder.add_node("supervisor", research_supervisor_node)
research_builder.add_node("search", search_node)
research_builder.add_node("web_scraper", web_scraper_node)

research_builder.add_edge(START, "supervisor")
research_graph = research_builder.compile()