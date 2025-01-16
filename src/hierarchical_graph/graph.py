from typing import Any, Dict, Literal, cast, TypedDict, List

from langchain_core.runnables import RunnableConfig
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from src.shared.utils import load_chat_model, format_docs

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

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

tavily_tool = TavilySearchResults(max_results=3)

@tool
def scrape_webpages(urls: List[str]) -> str:
    """Use requests and bs4 to scrape the provided web pages for detailed information."""
    print("[scrape_webpages] Called with URLs:", urls)
    loader = WebBaseLoader(urls)
    
    # Optionally wrap in a try/except to see if an error is thrown
    try:
        docs = loader.load()
    except Exception as e:
        print(f"[scrape_webpages] Error while loading pages: {e}")
        return "Error loading pages: " + str(e)

    if not docs:
        print("[scrape_webpages] No documents were loaded!")
        return "No documents loaded."

    print("[scrape_webpages] Loaded Documents:")
    print(f"Count: {len(docs)}")
    for i, doc in enumerate(docs):
        # Print partial content or metadata
        print(f"--- Document #{i} ---")
        print("Metadata:", doc.metadata)
        print("Content (first 500 chars):")
        print(doc.page_content[:500])
        print("------")

    # Finally, return the combined string
    return "\n\n".join(
        [
            f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
            for doc in docs
        ]
    )

def search_node(state: AgentState, *, config: RunnableConfig) -> Command[Literal["supervisor"]]:
    query = state.messages[-1].content
    search_results = tavily_tool.invoke(query)

    result = {
        "result": search_results,
        "status": "completed"
    }
    print("[Search Node] Returning Updated Data:")
    print(result)
    new_message = AIMessage(content=f"Search result: {result['result']}", name="search")
    updated_messages = state.messages + [new_message]
   
    task_history = state.task_history or []
    task_history.append({
        "task": "search_node",
        "result": search_results,
        "status": "completed"
    })


    return Command(
        update = {
            "search_response": result,
            "messages": updated_messages,
            "task_history": task_history,
        },
        goto="supervisor",
    )

def web_scraper_node(state: AgentState, *, config: RunnableConfig) -> Command[Literal["supervisor"]]:
    configuration = Configuration.from_runnable_config(config)
    llm = load_chat_model(configuration.llm_router_model)

    # Create the ReAct agent with your `scrape_webpages` tool
    web_scraper_agent = create_react_agent(llm, tools=[scrape_webpages])

    # Build the prompt that the ReAct agent will see
    urls = [item["url"] for item in state.search_response["result"]]
    user_prompt = (
        f"Scrape the following URLs and return their combined text:\n{urls}\n\n"
        "Use the `scrape_webpages` tool to retrieve the content. Then summarize or return the raw content."
    )

    # Pass the agent *a dict* with messages
    # Here we have just one user message. 
    inputs = {
        "messages": [
            ("user", user_prompt),
        ]
    }

    # Invoke the compiled agent graph with that input
    result_state = web_scraper_agent.invoke(inputs)
    final_message = result_state["messages"][-1]
    print("[Web Scraper Node] Final Message:")
    print(final_message)

    # Now store that text in your own conversation
    new_message = AIMessage(content="finished scraping", name="web_scraper")
    updated_messages = state.messages + [new_message]

    # Return a proper Command with a dict in update
    return Command(
        update={
            "messages": updated_messages,
            "web_scraper_response": final_message.content,
        },
        goto="supervisor",
    )


research_builder = StateGraph(AgentState, input=InputState, config_schema=Configuration)
research_builder.add_node("supervisor", research_supervisor_node)
research_builder.add_node("search", search_node)
research_builder.add_node("web_scraper", web_scraper_node)

research_builder.add_edge(START, "supervisor")
research_graph = research_builder.compile()