from langchain_core.documents import Document

import src.search as search_module


def _fake_results():
    """Mimic Chroma's similarity_search_with_score output: (Document, distance)."""
    return [
        (Document(page_content="AI is the simulation of human intelligence.", metadata={"source": "a.pdf"}), 0.30),
        (Document(page_content="Machine learning is a subset of AI.", metadata={"source": "a.pdf"}), 0.45),
        (Document(page_content="The FIFA World Cup is a football tournament.", metadata={"source": "b.pdf"}), 0.75),
    ]


def test_search_filters_out_low_relevance_results(monkeypatch):
    monkeypatch.setattr(search_module, "similarity_search", lambda query, k=None: _fake_results())

    results = search_module.search("What is AI?", k=3, threshold=0.5)

    # distance 0.75 -> similarity 0.25, below threshold 0.5 -> excluded
    assert len(results) == 2
    assert all(score >= 0.5 for _, score in results)


def test_search_returns_empty_list_when_nothing_relevant(monkeypatch):
    monkeypatch.setattr(
        search_module,
        "similarity_search",
        lambda query, k=None: [(Document(page_content="irrelevant"), 0.9)],
    )

    results = search_module.search("Who won the FIFA World Cup?", threshold=0.5)

    assert results == []


def test_search_empty_query_returns_empty_list():
    assert search_module.search("   ") == []


def test_distance_to_similarity_never_negative():
    assert search_module._distance_to_similarity(1.5) == 0.0
    assert search_module._distance_to_similarity(0.0) == 1.0
