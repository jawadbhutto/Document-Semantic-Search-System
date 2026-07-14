"""
main.py
-------
Entry point. Deliberately thin:

    run pipeline -> ask user for query -> search -> print results (Streamlit)

Run with:
    streamlit run main.py
"""

import streamlit as st

from src.config import settings
from src.pipeline import run_ingestion_pipeline
from src.search import NO_RESULTS_MESSAGE, search

st.set_page_config(page_title="Semantic Search", page_icon="🔎", layout="centered")
st.title("🔎 Document Semantic Search")
st.caption("Semantic search over your indexed PDF documents, powered by ChromaDB.")

# --------------------------------------------------------------------- #
# Step 1: Run the pipeline (idempotent — safe to re-run every load)
# --------------------------------------------------------------------- #
with st.sidebar:
    st.header("Index")
    st.write(f"Source folder: `{settings.raw_data_dir}`")
    if st.button("Rebuild / update index", use_container_width=True):
        with st.spinner("Loading, chunking, and embedding documents..."):
            summary = run_ingestion_pipeline()
        st.success("Index updated.")
        st.json(summary)

    st.divider()
    st.subheader("Search settings")
    top_k = st.slider("Top K results", min_value=1, max_value=10, value=settings.top_k)
    threshold = st.slider(
        "Relevance threshold",
        min_value=0.0,
        max_value=1.0,
        value=settings.relevance_threshold,
        step=0.05,
        help="Minimum similarity score a result must reach to be shown.",
    )

# --------------------------------------------------------------------- #
# Step 2: Ask the user for a query
# --------------------------------------------------------------------- #
query = st.text_input("Ask a question about your documents:", placeholder="e.g. What is Artificial Intelligence?")
run_search = st.button("Search", type="primary")

# --------------------------------------------------------------------- #
# Step 3 & 4: Search, then print results
# --------------------------------------------------------------------- #
if run_search and query:
    with st.spinner("Searching..."):
        results = search(query, k=top_k, threshold=threshold)

    if not results:
        st.warning(NO_RESULTS_MESSAGE)
    else:
        st.subheader(f"Top {len(results)} result(s)")
        for rank, (document, score) in enumerate(results, start=1):
            source = document.metadata.get("source", "unknown")
            page = document.metadata.get("page_label", document.metadata.get("page", "?"))
            with st.container(border=True):
                st.markdown(f"**Result {rank}** &nbsp;·&nbsp; relevance `{score:.3f}` &nbsp;·&nbsp; `{source}` (page {page})")
                st.write(document.page_content)
elif run_search and not query:
    st.info("Type a question above, then click Search.")
