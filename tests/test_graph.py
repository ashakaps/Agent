"""Тесты графа агента с поддельной LLM: без сети, ключей и кредитов."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import app.agent.graph as graph_module
from app.agent.graph import build_agent_graph


class FakeLLM:
    """Отдаёт заготовленные ответы по очереди и считает вызовы."""

    def __init__(self, responses: list[AIMessage]):
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if not self.responses:
            raise AssertionError("FakeLLM was called more times than expected")
        return self.responses.pop(0)


def _tool_call(code: str, call_id: str = "call_1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "python_repl", "args": {"code": code}, "id": call_id}],
    )


@pytest.fixture
def patch_llms(monkeypatch):
    def _patch(with_tools: FakeLLM, plain: FakeLLM | None = None) -> None:
        monkeypatch.setattr(graph_module, "_get_llm_with_tools", lambda: with_tools)
        monkeypatch.setattr(graph_module, "_get_llm", lambda: plain or FakeLLM([]))

    return _patch


def test_plain_answer_ends_without_tools(patch_llms):
    with_tools = FakeLLM([AIMessage(content="Hello!")])
    patch_llms(with_tools)

    result = build_agent_graph().invoke(
        {"messages": [HumanMessage(content="Hi")], "iteration_count": 0}
    )

    assert result["messages"][-1].content == "Hello!"
    assert result["iteration_count"] == 1
    assert with_tools.calls == 1


def test_tool_call_is_executed_and_agent_answers(patch_llms):
    with_tools = FakeLLM(
        [_tool_call("print(6 * 7)"), AIMessage(content="The answer is 42.")]
    )
    patch_llms(with_tools)

    result = build_agent_graph().invoke(
        {"messages": [HumanMessage(content="6*7?")], "iteration_count": 0}
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "42" in tool_messages[0].content
    assert result["messages"][-1].content == "The answer is 42."
    assert result["iteration_count"] == 2


def test_iteration_limit_forces_final_answer_without_tools(patch_llms, monkeypatch):
    monkeypatch.setattr("app.agent.graph.settings.max_iterations", 2)
    with_tools = FakeLLM([_tool_call("print(1)"), _tool_call("print(2)", "call_2")])
    plain = FakeLLM([AIMessage(content="Final answer.")])
    patch_llms(with_tools, plain)

    result = build_agent_graph().invoke(
        {"messages": [HumanMessage(content="loop")], "iteration_count": 0}
    )

    # Итерация 1 — с инструментами, итерация 2 (последняя) — без них.
    assert with_tools.calls == 1
    assert plain.calls == 1
    assert result["messages"][-1].content == "Final answer."
    assert not result["messages"][-1].tool_calls


def test_hard_limit_ends_even_if_model_keeps_calling_tools(patch_llms, monkeypatch):
    monkeypatch.setattr("app.agent.graph.settings.max_iterations", 1)
    # Последняя итерация использует llm без инструментов, но даже если он
    # вернёт tool_call, маршрутизация обязана завершить граф.
    plain = FakeLLM([_tool_call("print(1)")])
    patch_llms(FakeLLM([]), plain)

    result = build_agent_graph().invoke(
        {"messages": [HumanMessage(content="x")], "iteration_count": 0}
    )

    assert result["iteration_count"] == 1
    assert plain.calls == 1
