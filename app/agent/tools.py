import logging
import os
import subprocess
import sys
import tempfile
import time

from ddgs import DDGS
from langchain_core.tools import tool

from app.config import settings
from app.rag.chain import build_retriever, format_docs_with_sources

logger = logging.getLogger(__name__)

_retriever = None


def _ensure_retriever():
    global _retriever
    if _retriever is None:
        _retriever = build_retriever()
    return _retriever


def _truncate(text: str, limit: int | None = None) -> str:
    limit = limit or settings.agent_max_output_chars
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} more chars]"


@tool
def documentation_search(query: str) -> str:
    """Search the scikit-learn documentation corpus for a given query.

    Use this tool when the user asks about scikit-learn classes, methods,
    parameters, or general ML concepts (Ridge, Lasso, decision trees, metrics).

    Args:
        query: natural-language question or keyword search.

    Returns:
        Numbered documentation fragments followed by a list of source URLs.
        Answer the user from these fragments and cite the sources.
    """
    try:
        docs = _ensure_retriever().invoke(query)
    except Exception as exc:
        logger.exception("documentation_search failed")
        return f"Documentation search failed: {type(exc).__name__}: {_truncate(str(exc), 300)}"

    if not docs:
        return "No relevant documentation fragments found."

    sources = [doc.metadata.get("source", "unknown") for doc in docs]
    sources_block = "\n".join(f"- {url}" for url in sources)
    return f"{format_docs_with_sources(docs)}\n\nSources:\n{sources_block}"


# SECURITY NOTE: код выполняется в отдельном процессе с таймаутом, из
# временной папки и без секретов в окружении, но это НЕ полноценная
# песочница: у процесса те же права на файлы и сеть, что и у приложения.
# Для недоверенных пользователей нужна настоящая изоляция (gVisor,
# Firecracker, e2b.dev) или запуск агента только внутри Docker.
_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


def _clean_env() -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if not any(marker in k.upper() for marker in _SECRET_MARKERS)
    }
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run_python(code: str, timeout: int) -> str:
    with tempfile.TemporaryDirectory() as workdir:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", code],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                cwd=workdir,
                env=_clean_env(),
            )
        except subprocess.TimeoutExpired:
            return f"Error: code execution timed out after {timeout} seconds."

    if proc.returncode != 0:
        return _truncate(f"Error (exit code {proc.returncode}):\n{proc.stderr.strip()}")
    output = proc.stdout.strip()
    if not output:
        return "(no output — use print() to show results)"
    return _truncate(output)


@tool
def python_repl(code: str) -> str:
    """Execute Python code and return the printed output.

    Use this tool for arithmetic, computing formulas, or transforming data.
    The code runs in a separate process with a time limit and without access
    to the application's secrets; use `print()` to surface results.

    Args:
        code: Python source code to execute.

    Returns:
        Stdout of the executed code, or an error message.
    """
    return _run_python(code, settings.python_repl_timeout_s)


def _ddg_text(query: str, max_results: int) -> list[dict]:
    return DDGS().text(query, max_results=max_results)


def _format_search_results(results: list[dict]) -> str:
    lines = []
    for item in results:
        title = item.get("title", "").strip()
        body = item.get("body", "").strip()
        url = item.get("href", "").strip()
        lines.append(f"- {title}\n  {body}\n  {url}")
    return "\n".join(lines)


@tool
def web_search(query: str) -> str:
    """Search the web via DuckDuckGo for recent or general-knowledge info.

    Use this tool when the question requires fresh data (release notes,
    latest versions, news) that is NOT in the documentation corpus.

    Args:
        query: free-text search query.

    Returns:
        Top search results as title, snippet and URL.
    """
    if not settings.enable_web_search:
        return "Web search is disabled in this environment."

    attempts = max(1, settings.web_search_retries)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            results = _ddg_text(query, settings.web_search_max_results)
        except Exception as exc:  # ddgs сам выбирает бэкенд — сбои плавающие
            last_error = exc
            logger.warning(
                "web_search attempt %d/%d failed: %s", attempt, attempts, exc
            )
            if attempt < attempts:
                time.sleep(attempt)
            continue
        if not results:
            return "No results found."
        return _truncate(_format_search_results(results))

    return (
        f"Web search failed after {attempts} attempts: "
        f"{type(last_error).__name__}: {_truncate(str(last_error), 300)}"
    )


TOOLS = [documentation_search, python_repl, web_search]
