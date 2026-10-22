"""
Hybrid retriever
----------------
Runs a vector (meaning) search AND a keyword (exact word) search,
then merges both result lists with Reciprocal Rank Fusion.
"""

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag.retrievers.rrf import reciprocal_rank_fusion


class HybridRetriever(BaseRetriever):
    vector_retriever: BaseRetriever
    keyword_retriever: BaseRetriever
    k: int = 4

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        vector_results = self.vector_retriever.invoke(query)
        keyword_results = self.keyword_retriever.invoke(query)
        fused = reciprocal_rank_fusion([vector_results, keyword_results])
        return fused[: self.k]