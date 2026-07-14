from langchain_core.documents import Document

from src.utils import clean_text, format_result, truncate


def test_clean_text_collapses_whitespace():
    assert clean_text("Hello   \n\n world \t!") == "Hello world !"


def test_clean_text_handles_empty_string():
    assert clean_text("") == ""
    assert clean_text(None) == ""  # type: ignore[arg-type]


def test_truncate_short_text_is_unchanged():
    assert truncate("short text", max_chars=50) == "short text"


def test_truncate_long_text_adds_ellipsis():
    text = "a" * 500
    result = truncate(text, max_chars=100)
    assert result.endswith("...")
    assert len(result) == 103  # 100 chars + "..."


def test_format_result_includes_score_and_source():
    doc = Document(page_content="AI is a field of computer science.", metadata={"source": "AI_Ethics.pdf", "page": 0})
    formatted = format_result(1, doc, 0.87)
    assert "Result 1" in formatted
    assert "0.870" in formatted
    assert "AI_Ethics.pdf" in formatted
