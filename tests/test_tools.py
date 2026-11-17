from unittest.mock import MagicMock, patch

import pytest

import app.agent.tools as tools_module
from app.agent.tools import documentation_search, python_repl, web_search


@pytest.fixture(autouse=True)
def reset_retriever():
    """Кэш retriever'а живёт в модуле — сбрасываем между тестами."""
    tools_module._retriever = None
    yield
    tools_module._retriever = None


def _doc(source: str, text: str) -> MagicMock:
    doc = MagicMock()
    doc.metadata = {"source": source}
    doc.page_content = text
    return doc


# --- documentation_search ---------------------------------------------------


@patch("app.agent.tools.build_retriever")
def test_documentation_search_returns_fragments_and_sources(mock_build):
    retriever = MagicMock()
    retriever.invoke.return_value = [
        _doc("https://scikit-learn.org/ridge.html", "Ridge is L2 regularization.")
    ]
    mock_build.return_value = retriever

    result = documentation_search.invoke({"query": "What is Ridge?"})

    assert "Ridge is L2 regularization." in result
    assert "Sources:" in result
    assert "scikit-learn.org" in result
    # Один поход в Qdrant на вызов, без вложенного вызова LLM.
    assert retriever.invoke.call_count == 1


@patch("app.agent.tools.build_retriever")
def test_documentation_search_handles_empty_result(mock_build):
    retriever = MagicMock()
    retriever.invoke.return_value = []
    mock_build.return_value = retriever

    result = documentation_search.invoke({"query": "nothing"})

    assert "no relevant" in result.lower()


@patch("app.agent.tools.build_retriever")
def test_documentation_search_reports_backend_failure(mock_build):
    mock_build.side_effect = ConnectionError("qdrant is down")

    result = documentation_search.invoke({"query": "Ridge"})

    assert "failed" in result.lower()
    assert "qdrant is down" in result


# --- python_repl ------------------------------------------------------------


def test_python_repl_executes_arithmetic():
    result = python_repl.invoke({"code": "print(7 * 6)"})
    assert "42" in result


def test_python_repl_handles_syntax_error_gracefully():
    result = python_repl.invoke({"code": "print(1 +"})
    assert "error" in result.lower() or "syntax" in result.lower()


def test_python_repl_times_out(monkeypatch):
    monkeypatch.setattr("app.agent.tools.settings.python_repl_timeout_s", 1)
    result = python_repl.invoke({"code": "while True: pass"})
    assert "timed out" in result.lower()


def test_python_repl_does_not_leak_secrets(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "super-secret-value")
    result = python_repl.invoke(
        {"code": "import os; print(os.environ.get('LLM_API_KEY'))"}
    )
    assert "super-secret-value" not in result
    assert "None" in result


def test_python_repl_hints_when_no_output():
    result = python_repl.invoke({"code": "x = 1"})
    assert "no output" in result.lower()


# --- web_search -------------------------------------------------------------


def test_web_search_respects_disable_flag(monkeypatch):
    monkeypatch.setattr("app.agent.tools.settings.enable_web_search", False)
    result = web_search.invoke({"query": "anything"})
    assert "disabled" in result.lower()


def test_web_search_formats_results(monkeypatch):
    monkeypatch.setattr("app.agent.tools.settings.enable_web_search", True)
    with patch(
        "app.agent.tools._ddg_text",
        return_value=[
            {
                "title": "Release notes",
                "body": "scikit-learn 1.8",
                "href": "https://x.org/r",
            }
        ],
    ):
        result = web_search.invoke({"query": "latest scikit-learn release"})

    assert "Release notes" in result
    assert "https://x.org/r" in result


def test_web_search_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr("app.agent.tools.settings.enable_web_search", True)
    monkeypatch.setattr("app.agent.tools.settings.web_search_retries", 3)
    monkeypatch.setattr("app.agent.tools.time.sleep", lambda _: None)
    flaky = MagicMock(
        side_effect=[
            RuntimeError("DNSError: wt.wikipedia.org"),
            [{"title": "OK", "body": "b", "href": "https://x.org"}],
        ]
    )
    with patch("app.agent.tools._ddg_text", flaky):
        result = web_search.invoke({"query": "q"})

    assert "OK" in result
    assert flaky.call_count == 2


def test_web_search_returns_error_text_instead_of_raising(monkeypatch):
    monkeypatch.setattr("app.agent.tools.settings.enable_web_search", True)
    monkeypatch.setattr("app.agent.tools.settings.web_search_retries", 2)
    monkeypatch.setattr("app.agent.tools.time.sleep", lambda _: None)
    with patch("app.agent.tools._ddg_text", side_effect=RuntimeError("DNSError")):
        result = web_search.invoke({"query": "q"})

    assert "failed" in result.lower()
    assert "DNSError" in result
