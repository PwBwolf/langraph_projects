"""Define the state structures for the agent."""

from __future__ import annotations


from dataclasses import dataclass, field
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

@dataclass(kw_only=True)
class InputState:
    """Represents the input state for the agent.

    This class defines the structure of the input state, which includes
    the messages exchanged between the user and the agent. It serves as
    a restricted version of the full State, providing a narrower interface
    to the outside world compared to what is maintained internally.
    """

    messages: Annotated[list[AnyMessage], add_messages]

@dataclass
class Router:
    type: Literal["more-info", "movie", "general"]
    logic: str = ""

@dataclass(kw_only=True)
class AgentState(InputState):
    """Defines the input state for the agent, representing a narrower interface to the outside world.

    This class is used to define the initial state and structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    for more information.
    """

    # router: Router = field(default_factory=lambda: Router(type="general", logic=""))

    router: Router = field(default_factory=lambda: Router(type="general"))

