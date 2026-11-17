from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools import TOOLS
from app.config import settings
from app.llm import get_llm


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    iteration_count: int


_llm = None
_llm_with_tools = None


def _get_llm():
    """LLM без инструментов — для финального ответа на последней итерации."""
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm


def _get_llm_with_tools():
    global _llm_with_tools
    if _llm_with_tools is None:
        _llm_with_tools = _get_llm().bind_tools(TOOLS)
    return _llm_with_tools


def agent_node(state: AgentState) -> dict:
    iteration = state.get("iteration_count", 0)
    # На последней разрешённой итерации инструменты не подключаем: модель
    # обязана дать текстовый ответ, а не очередной tool_call без текста.
    is_last_iteration = iteration + 1 >= settings.max_iterations
    llm = _get_llm() if is_last_iteration else _get_llm_with_tools()

    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {
        "messages": [response],
        "iteration_count": iteration + 1,
    }


def route_after_agent(state: AgentState) -> str:
    """Есть tool_calls — идём в tool_executor, иначе заканчиваем.

    Жёсткий предохранитель: по достижении max_iterations всегда END, даже
    если модель всё равно вернула tool_call.
    """
    if state.get("iteration_count", 0) >= settings.max_iterations:
        return END
    return tools_condition(state)  # "tools" или END


tool_executor_node = ToolNode(TOOLS)


def build_agent_graph(checkpointer=None):
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tool_executor", END: END},
    )
    graph.add_edge("tool_executor", "agent")
    return graph.compile(checkpointer=checkpointer)
