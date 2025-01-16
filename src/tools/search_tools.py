from typing import List, Annotated
from googleapiclient.discovery import build
from langchain_core.tools import tool

# Google Custom Search API Configuration
API_KEY = "your_google_api_key"  # Replace with your API key
SEARCH_ENGINE_ID = "your_search_engine_id"  # Replace with your Search Engine ID

def google_search(query: str, max_results: int = 5) -> List[str]:
    """Perform a Google search and return the top results as a list of URLs."""
    service = build("customsearch", "v1", developerKey=API_KEY)
    response = service.cse().list(q=query, cx=SEARCH_ENGINE_ID, num=max_results).execute()

    if "items" not in response:
        return []

    return [item["link"] for item in response["items"]]

@tool
def google_search_tool(query: Annotated[str, "Search query string"]) -> str:
    """Tool for performing a Google search and retrieving the top results."""
    try:
        results = google_search(query)
        if not results:
            return "No results found for the query."
        # Format results for LLM
        return "\n".join(results)
    except Exception as e:
        return f"Error performing search: {e}"